import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench_lse(X, steps, T0, T1, lr, mu=0.9, deadline=None, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    if track:
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        bv = float(G0.max()); bX = X.copy()
    else:
        bv = 1e100; bX = None
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        mk = gn[:, 0] > 2.0
        if np.any(mk):
            Gt[mk] *= (2.0 / gn[mk])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if track and ((s % 150 == 0) or s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _quench_riesz(X, steps, s_pow, lr, mu=0.9, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        # squared distance
        D2 = np.clip(2.0 - 2.0 * G, 1e-6, 4.0)
        # weight s * D2^{-s/2 -1}
        W = np.power(D2, -s_pow * 0.5 - 1.0) * s_pow
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize scale: divide by median norm to keep lr stable
        gn = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / gn
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    return X

def _refine_single(s, R, steps=200, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=8000, n_cand=3):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _refine_single(s0, R, steps=200, T=0.005, lr=0.12)
        s1 = _refine_single(s1, R, steps=200, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cuv[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _row_scores(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, G

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 32.0
    rng = np.random.default_rng(42)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal((n, d))))
    # quick diverse random population to leave the incumbent basin
    elites = []
    elite_scores = []
    for a in cands:
        elites.append(a); elite_scores.append(_max_cos(a))
    # generate fresh random starts with cheap quench
    n_fresh = 6
    for f in range(n_fresh):
        if time.time() > HOP_END - 60:
            break
        X0 = _norm_rows(rng.standard_normal((n, d)))
        T0 = float(10 ** rng.uniform(-2.2, -1.5))
        T1 = float(10 ** rng.uniform(-3.8, -3.0))
        lr = float(10 ** rng.uniform(-1.3, -0.7))
        mu = float(rng.uniform(0.8, 0.93))
        try:
            Xq, sq = _quench_lse(X0, steps=900, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
        except Exception:
            continue
        elites.append(_norm_rows(Xq)); elite_scores.append(float(sq))
    # keep top 5
    if len(elites) > 0:
        od = np.argsort(np.array(elite_scores))
        elites = [elites[i] for i in od[:5]]
        elite_scores = [elite_scores[i] for i in od[:5]]
    best_X = elites[0].copy(); best_score = float(elite_scores[0])
    # main basin hopping with large moves
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        # pick base: 55% best, else random elite
        if rng.random() < 0.55 or len(elites) == 1:
            base = best_X
        else:
            base = elites[int(rng.integers(0, len(elites)))]
        u = rng.random()
        X0 = base.copy()
        if u < 0.30:
            # global shake
            sigma = float(10 ** rng.uniform(-2.6, -0.5))
            X0 = base + rng.standard_normal((n, d)) * sigma
            X0 = _norm_rows(X0)
        elif u < 0.65:
            # cap-kill: remove a whole spherical cap (large topology change)
            k = int(rng.integers(3, 10))  # 3..9
            direction = rng.standard_normal(d)
            direction /= (np.linalg.norm(direction) + 1e-12)
            if rng.random() < 0.5:
                # cap around a random existing point
                direction = base[int(rng.integers(0, n))].copy() + rng.standard_normal(d) * 0.3
                direction /= (np.linalg.norm(direction) + 1e-12)
            proj = base @ direction
            idx = np.argpartition(-proj, k)[:k]
            # replace with far-apart random points biased away from direction or to holes
            R = np.delete(base, idx, axis=0)
            for j in idx:
                # 50% pure random far, 50% hole-optimized
                if rng.random() < 0.4:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                else:
                    try:
                        h = _find_hole(R, rng, n_sample=5000, n_cand=3)
                    except Exception:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            # small noise on survivors
            keep = np.ones(n, dtype=bool); keep[idx] = False
            sigma = float(10 ** rng.uniform(-2.8, -1.2))
            X0[keep] += rng.standard_normal((np.count_nonzero(keep), d)) * sigma
            X0 = _norm_rows(X0)
        else:
            # hub-kill: worst 1-3 by row-max
            k = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
            try:
                rm, _ = _row_scores(base)
                order = np.argsort(-rm + rng.random(n) * 0.001)
                idx = order[:k]
            except Exception:
                idx = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[idx] = False
            R = base[keep]
            for j in idx:
                try:
                    h = _find_hole(R, rng, n_sample=7000, n_cand=3)
                except Exception:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            sigma = float(10 ** rng.uniform(-2.8, -1.0))
            X0[keep] += rng.standard_normal((n - k, d)) * sigma
            X0 = _norm_rows(X0)
        # choose quench type
        if tleft < 6:
            steps = 1000
        elif tleft < 25:
            steps = 1600
        else:
            steps = 1900
            if rng.random() < 0.2:
                steps = 3000
        try:
            if rng.random() < 0.32:
                # Riesz pre-condition then LSE finish (different basin)
                s_pow = float(rng.uniform(2.0, 7.0))
                lr_r = float(10 ** rng.uniform(-1.6, -0.9))
                mu_r = float(rng.uniform(0.8, 0.92))
                Xr = _quench_riesz(X0, steps=max(300, steps // 2), s_pow=s_pow, lr=lr_r, mu=mu_r, deadline=HOP_END)
                T0 = float(10 ** rng.uniform(-2.6, -1.9))
                T1 = float(10 ** rng.uniform(-4.0, -3.2))
                lr2 = float(10 ** rng.uniform(-1.3, -0.8))
                mu2 = float(rng.uniform(0.82, 0.93))
                X1, s1 = _quench_lse(Xr, steps=steps, T0=T0, T1=T1, lr=lr2, mu=mu2, deadline=HOP_END)
            else:
                T0 = float(10 ** rng.uniform(-2.5, -1.6))
                T1 = float(10 ** rng.uniform(-4.2, -3.1))
                lr = float(10 ** rng.uniform(-1.4, -0.7))
                mu = float(rng.uniform(0.82, 0.94))
                X1, s1 = _quench_lse(X0, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        # maintain elite diversity
        if s1 < max(elite_scores) + 1e-9 and s1 < 0.26:
            # insert if different enough
            try:
                dup = False
                for e in elites:
                    if float(np.abs(e - X1).max()) < 1e-6:
                        dup = True; break
                if not dup:
                    elites.append(_norm_rows(X1)); elite_scores.append(float(s1))
                    od = np.argsort(np.array(elite_scores))
                    elites = [elites[i] for i in od[:5]]
                    elite_scores = [elite_scores[i] for i in od[:5]]
            except Exception:
                pass
        if s1 < best_score + 0.0015:
            secs = 1.2
            if tleft < secs + 0.8:
                secs = max(0.4, tleft - 0.8)
            try:
                X2 = _lp_polish(X1, seconds=secs, deadline=HOP_END)
                s2 = _max_cos(X2)
            except Exception:
                continue
            if np.isfinite(s2) and s2 < best_score - 1e-12:
                best_X = _norm_rows(X2); best_score = float(s2)
                # update elites top
                elites[0] = best_X.copy(); elite_scores[0] = best_score
                od = np.argsort(np.array(elite_scores))
                elites = [elites[i] for i in od]
                elite_scores = [elite_scores[i] for i in od]
    # final squeeze
    try:
        trem = HARD_END - time.time() - 0.6
        if trem > 2.0:
            qs = 4500 if trem > 16 else 2200
            try:
                Xf, sf = _quench_lse(best_X, steps=qs, T0=0.0006, T1=0.00008, lr=0.06, mu=0.9, deadline=HARD_END - 8.0)
                if sf < best_score:
                    best_X, best_score = _norm_rows(Xf), float(sf)
            except Exception:
                pass
            trem2 = HARD_END - time.time() - 0.5
            if trem2 > 1.0:
                best_X = _lp_polish(best_X, seconds=min(trem2, 30.0), deadline=HARD_END - 0.4)
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    for i in range(n):
        if not np.all(np.isfinite(best_X[i])) or float(np.linalg.norm(best_X[i])) < 1e-12:
            v = rng.standard_normal(d)
            v = v / (np.linalg.norm(v) + 1e-12)
            best_X[i] = v
    best_X = _norm_rows(best_X)
    return best_X

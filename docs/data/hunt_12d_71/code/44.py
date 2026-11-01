import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _quench_lse(X, steps, T0, T1, lr, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max()); bX = X.copy()
    for s in range(steps):
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
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        mk = gn[:, 0] > 2.0
        if np.any(mk):
            Gt[mk] *= (2.0 / gn[mk])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 120 == 0) or s == steps - 1:
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _quench_pnorm(X, steps, p0, p1, lr, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max()); bX = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        p = p0 + (p1 - p0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m <= 0:
            m = 0.1
        Gn = G / m
        Gn[G < 0] = 0.0
        # weights (Gn)^(p-1), Gn in [0,1]
        W = np.power(np.clip(Gn, 0, 1), p - 1.0)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        # avoid zero rows
        med = float(np.median(rs) + 1e-12)
        rs[rs < 1e-12] = med
        Grad = (W @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / gn
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 120 == 0) or s == steps - 1:
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _tunnel_push(X, steps, step0, step1, noise0, rng):
    X = X.copy()
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max()); bX = X.copy()
    n = X.shape[0]
    for s in range(steps):
        frac = s / max(1, steps - 1)
        step = step0 + (step1 - step0) * frac
        noi = noise0 * (1.0 - frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G.max(axis=1)))
        # partner of k
        G[k, k] = -2.0
        j = int(np.argmax(G[k]))
        # push k away from j and j away from k
        for a, b in ((k, j), (j, k)):
            d = X[b] - float(X[a] @ X[b]) * X[a]
            dn = float(np.linalg.norm(d))
            if dn > 1e-12:
                d = d / dn
                X[a] = X[a] - step * d
        if noi > 0:
            N = rng.standard_normal(X.shape) * noi
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            X = X + N
        X = _norm_rows(X)
        if (s % 50 == 0) or s == steps - 1:
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _refine_single(s, R, steps=150, T=0.003, lr=0.12, mu=0.8):
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

def _find_hole(R, rng, n_sample=4000, n_cand=2):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None; best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _refine_single(s0, R, steps=150, T=0.004, lr=0.12)
        s1 = _refine_single(s1, R, steps=150, T=0.0012, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _rand_orth(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def _lp_polish(x, seconds, deadline=None, r0=3e-3):
    x = _norm_rows(np.array(x, dtype=float, copy=True))
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        band = max(30.0 * radius, 2e-4)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2000:
            order = np.argsort(-cmat[sel])[:2000]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
        else:
            csel = cmat[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 30.0
    rng = np.random.default_rng(12345)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal((n, d))))
    elites = []
    elite_scores = []
    for a in cands:
        elites.append(a.copy()); elite_scores.append(_max_cos(a))
    # diverse fresh starts with p-norm (different basin from LSE)
    for f in range(5):
        if time.time() > HOP_END - 50:
            break
        X0 = _norm_rows(rng.standard_normal((n, d)))
        try:
            p0 = float(rng.uniform(6, 14)); p1 = float(rng.uniform(40, 100))
            lr = float(10 ** rng.uniform(-1.5, -0.9)); mu = float(rng.uniform(0.8, 0.92))
            Xq, sq = _quench_pnorm(X0, steps=700, p0=p0, p1=p1, lr=lr, mu=mu)
            T0 = float(10 ** rng.uniform(-2.4, -1.8)); T1 = float(10 ** rng.uniform(-4.0, -3.2))
            lr2 = float(10 ** rng.uniform(-1.3, -0.8)); mu2 = float(rng.uniform(0.82, 0.93))
            Xq2, sq2 = _quench_lse(Xq, steps=700, T0=T0, T1=T1, lr=lr2, mu=mu2)
            elites.append(_norm_rows(Xq2)); elite_scores.append(float(sq2))
        except Exception:
            continue
    od = np.argsort(np.array(elite_scores))
    elites = [elites[i] for i in od[:6]]
    elite_scores = [float(elite_scores[i]) for i in od[:6]]
    best_X = elites[0].copy(); best_score = float(elite_scores[0])
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        if rng.random() < 0.5 or len(elites) == 1:
            base = best_X
        else:
            base = elites[int(rng.integers(0, len(elites)))]
        u = rng.random()
        X0 = base.copy()
        try:
            if u < 0.22:
                sigma = float(10 ** rng.uniform(-2.6, -0.5))
                X0 = base + rng.standard_normal((n, d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.42:
                k = int(rng.integers(3, 10))
                direction = rng.standard_normal(d)
                direction /= (np.linalg.norm(direction) + 1e-12)
                if rng.random() < 0.5:
                    direction = base[int(rng.integers(0, n))].copy() + rng.standard_normal(d) * 0.3
                    direction /= (np.linalg.norm(direction) + 1e-12)
                proj = base @ direction
                idx = np.argpartition(-proj, k)[:k]
                keep = np.ones(n, dtype=bool); keep[idx] = False
                R = base[keep]
                for j in idx:
                    if rng.random() < 0.35:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                    else:
                        h = _find_hole(R, rng, n_sample=4000, n_cand=2)
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                sigma = float(10 ** rng.uniform(-2.8, -1.2))
                X0[keep] += rng.standard_normal((np.count_nonzero(keep), d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.60:
                # fission rotation: split by random hyperplane, rotate one side
                for _try in range(8):
                    rv = rng.standard_normal(d); rv /= (np.linalg.norm(rv) + 1e-12)
                    pr = base @ rv
                    mask = pr > 0
                    c = int(mask.sum())
                    if 15 <= c <= 56:
                        break
                else:
                    mask = rng.random(n) < 0.5
                Q = _rand_orth(d, rng)
                # interpolate rotation strength: sometimes full, sometimes partial
                if rng.random() < 0.5:
                    X0[mask] = base[mask] @ Q.T
                    # small noise both sides
                    X0 += rng.standard_normal((n, d)) * float(10 ** rng.uniform(-3.0, -1.5))
                    X0 = _norm_rows(X0)
                else:
                    # small rotation: X' = norm((1-a)X + a XQ)
                    a = float(rng.uniform(0.3, 0.9))
                    rot = base[mask] @ Q.T
                    X0[mask] = _norm_rows((1 - a) * base[mask] + a * rot)
                    X0[~mask] += rng.standard_normal((np.count_nonzero(~mask), d)) * 1e-3
                    X0 = _norm_rows(X0)
            elif u < 0.75:
                k = 2 if rng.random() < 0.5 else (3 if rng.random() < 0.7 else 4)
                rm = _row_max(base)
                order = np.argsort(-rm + rng.random(n) * 0.002)
                idx = order[:k]
                keep = np.ones(n, dtype=bool); keep[idx] = False
                R = base[keep]
                for j in idx:
                    h = _find_hole(R, rng, n_sample=5000, n_cand=2)
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                sigma = float(10 ** rng.uniform(-2.8, -1.0))
                X0[keep] += rng.standard_normal((n - k, d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.87:
                # crossover of two elites
                if len(elites) >= 2:
                    a = elites[int(rng.integers(0, len(elites)))]
                    b = elites[int(rng.integers(0, len(elites)))]
                    mask = rng.random(n) < 0.5
                    X0 = np.where(mask[:, None], a, b)
                    X0 = X0 + rng.standard_normal((n, d)) * float(10 ** rng.uniform(-2.5, -1.0))
                    X0 = _norm_rows(X0)
                else:
                    X0 = base + rng.standard_normal((n, d)) * 0.05
                    X0 = _norm_rows(X0)
            else:
                # worst-pair tunnel pre-move
                try:
                    Xt, _ = _tunnel_push(base, steps=250, step0=0.01, step1=0.002, noise0=0.01, rng=rng)
                    X0 = Xt + rng.standard_normal((n, d)) * float(10 ** rng.uniform(-2.8, -1.4))
                    X0 = _norm_rows(X0)
                except Exception:
                    X0 = base + rng.standard_normal((n, d)) * 0.03
                    X0 = _norm_rows(X0)
        except Exception:
            continue
        if tleft < 6:
            steps = 900
        elif tleft < 25:
            steps = 1400
        else:
            steps = 1700
            if rng.random() < 0.15:
                steps = 2600
        try:
            r2 = rng.random()
            if r2 < 0.45:
                s_pow0 = float(rng.uniform(8, 16)); s_pow1 = float(rng.uniform(50, 110))
                lr_r = float(10 ** rng.uniform(-1.6, -0.9)); mu_r = float(rng.uniform(0.8, 0.92))
                Xr, _ = _quench_pnorm(X0, steps=max(400, steps // 2), p0=s_pow0, p1=s_pow1, lr=lr_r, mu=mu_r)
                T0 = float(10 ** rng.uniform(-2.6, -1.9)); T1 = float(10 ** rng.uniform(-4.2, -3.2))
                lr2 = float(10 ** rng.uniform(-1.3, -0.8)); mu2 = float(rng.uniform(0.82, 0.93))
                X1, s1 = _quench_lse(Xr, steps=steps, T0=T0, T1=T1, lr=lr2, mu=mu2)
            elif r2 < 0.75:
                T0 = float(10 ** rng.uniform(-2.5, -1.6)); T1 = float(10 ** rng.uniform(-4.2, -3.1))
                lr = float(10 ** rng.uniform(-1.4, -0.7)); mu = float(rng.uniform(0.82, 0.94))
                X1, s1 = _quench_lse(X0, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu)
            else:
                # tunnel then LSE
                Xt, _ = _tunnel_push(X0, steps=300, step0=0.008, step1=0.001, noise0=0.008, rng=rng)
                T0 = float(10 ** rng.uniform(-2.6, -2.0)); T1 = float(10 ** rng.uniform(-4.2, -3.4))
                lr = float(10 ** rng.uniform(-1.4, -0.8)); mu = float(rng.uniform(0.82, 0.93))
                X1, s1 = _quench_lse(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        if s1 < 0.26:
            # diversity check
            dup = False
            for e in elites:
                try:
                    dd = float(np.linalg.norm(X1 - e) + np.linalg.norm(X1 + e))
                    # cheap: if very close in Frobenius to an elite, skip unless better
                    if min(float(np.linalg.norm(X1 - e)), float(np.linalg.norm(X1 + e))) < 1e-3:
                        dup = True
                        break
                except Exception:
                    pass
            if dup and s1 >= best_score - 2e-7:
                continue
            elites.append(_norm_rows(X1)); elite_scores.append(float(s1))
            od = np.argsort(np.array(elite_scores))
            elites = [elites[i] for i in od[:6]]
            elite_scores = [float(elite_scores[i]) for i in od[:6]]
            if s1 < best_score:
                best_X = _norm_rows(X1).copy(); best_score = float(s1)
                # quick squeeze of new best if time allows
                if time.time() < HOP_END - 4:
                    try:
                        best_X = _lp_polish(best_X, seconds=1.0, deadline=HOP_END)
                        best_score = _max_cos(best_X)
                        elites[0] = best_X.copy(); elite_scores[0] = best_score
                    except Exception:
                        pass
    # final intensive polish
    try:
        best_score = _max_cos(best_X)
    except Exception:
        best_X = _norm_rows(elites[0])
        best_score = _max_cos(best_X)
    # candidate pool for final: top 3 elites
    order = np.argsort(np.array(elite_scores))
    final_cands = [elites[i] for i in order[:min(3, len(elites))]]
    polished = []
    for fc in final_cands:
        tleft = HARD_END - time.time() - 1.0
        if tleft < 1.0:
            polished.append((fc, _max_cos(fc)))
            continue
        try:
            # low-T tighten
            Xp, sp = _quench_lse(fc.copy(), steps=1200, T0=0.0012, T1=0.00025, lr=0.25, mu=0.9)
            # LP
            tlp = min(4.0, max(1.0, tleft / (len(final_cands) + 1)))
            Xp = _lp_polish(Xp, seconds=tlp, deadline=HARD_END - 6.0, r0=2e-3)
            # second tighten even lower T
            Xp2, sp2 = _quench_lse(Xp, steps=800, T0=0.0006, T1=0.00018, lr=0.18, mu=0.9)
            if _max_cos(Xp2) < _max_cos(Xp):
                Xp = Xp2
            Xp = _lp_polish(Xp, seconds=min(3.0, max(0.8, HARD_END - time.time() - 5.0)), deadline=HARD_END - 4.0, r0=1e-3)
            polished.append((Xp, _max_cos(Xp)))
        except Exception:
            try:
                polished.append((fc, _max_cos(fc)))
            except Exception:
                pass
    if polished:
        polished.sort(key=lambda t: t[1])
        best_X, best_score = polished[0][0], float(polished[0][1])
    # micro hill-climbing on worst points
    deadline_micro = HARD_END - 0.8
    try:
        G = best_X @ best_X.T
        np.fill_diagonal(G, -2.0)
        cur = float(G.max())
        rng2 = np.random.default_rng(999)
        nsteps = 0
        while time.time() < deadline_micro and nsteps < 8000:
            nsteps += 1
            rm = (_row_max(best_X) if nsteps % 5 == 0 else None)
            if rm is not None:
                # bias to worst points
                probs = np.clip(rm - (cur - 0.01), 0, None)
                if probs.sum() <= 0:
                    idx = int(rng2.integers(0, n))
                else:
                    probs = probs / probs.sum()
                    idx = int(rng2.choice(n, p=probs))
            else:
                # random of top-10 worst from last rm
                idx = int(rng2.integers(0, n))
            sigma = 3e-4 if (nsteps % 3 == 0) else 1e-4
            if nsteps > 4000:
                sigma *= 0.4
            trial = best_X.copy()
            trial[idx] = trial[idx] + rng2.standard_normal(d) * sigma
            trial[idx] /= (np.linalg.norm(trial[idx]) + 1e-12)
            Gt = trial @ trial.T
            np.fill_diagonal(Gt, -2.0)
            v = float(Gt.max())
            if v < cur:
                best_X = trial; cur = v; best_score = v
        # final tiny shrink not needed; ensure normalized
        best_X = _norm_rows(best_X)
        best_score = _max_cos(best_X)
    except Exception:
        pass
    best_X = _norm_rows(np.asarray(best_X, dtype=np.float64))
    return best_X

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench_lse(X, steps, T0, T1, lr, mu=0.9, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(max(T0, 1e-12)); lt1 = np.log(max(T1, 1e-12))
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max()); bX = X.copy()
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
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        mk = gn[:, 0] > 2.0
        if np.any(mk):
            Gt[mk] *= (2.0 / gn[mk])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 200 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    return bX, bv

def _quench_riesz(X, steps, s_pow, lr, mu=0.9, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        D2 = np.clip(2.0 - 2.0 * G, 1e-6, 4.0)
        W = np.power(D2, -s_pow * 0.5 - 1.0) * s_pow
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / gn
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    return X

def _refine_single(s, R, steps=150, T=0.004, lr=0.12, mu=0.8):
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
        try:
            s1 = _refine_single(s0, R, steps=150, T=0.005, lr=0.12)
            s1 = _refine_single(s1, R, steps=150, T=0.0015, lr=0.07)
        except Exception:
            s1 = s0
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    if best_s is None:
        best_s = _norm_rows(rng.standard_normal((1, d)))[0]
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _lp_polish(X0, seconds, deadline=None):
    X = _norm_rows(np.array(X0, dtype=float, copy=True))
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    bestX = X.copy()
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (X @ X.T)[iu]
        m = float(cmat.max())
        band = max(25.0 * radius, 2e-4)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1600:
            order = np.argsort(-cmat[sel])[:1600]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 1600
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return bestX

def _build_d12_pool(rng, want=2200):
    d = 12
    pool = []
    tries = 0
    batch = 20000
    while len(pool) < want and tries < 12:
        tries += 1
        V = rng.integers(-2, 3, size=(batch, d))
        s = V.sum(axis=1) % 2
        n2 = (V * V).sum(axis=1)
        mask = (s == 0) & ((n2 == 2) | (n2 == 4) | (n2 == 6) | (n2 == 8))
        idx = np.flatnonzero(mask)
        for k in idx[: max(0, want - len(pool)) * 2]:
            v = V[k].astype(float)
            pool.append(v / np.linalg.norm(v))
            if len(pool) >= want:
                break
    # add random sphere to fill
    need = want - len(pool)
    if need > 0:
        Z = _norm_rows(rng.standard_normal((need, d)))
        for k in range(need):
            pool.append(Z[k])
    if len(pool) == 0:
        pool = [_norm_rows(rng.standard_normal((1, d)))[0]]
    return np.array(pool)

def _greedy_from_pool(pool, rng, n=71):
    m = pool.shape[0]
    order = rng.permutation(m)
    sel = [int(order[0])]
    dots = pool @ pool[sel[0]]
    best = dots.copy()
    for _ in range(1, n):
        # minimize max dot to selected set
        # best[k] = max dot so far; pick smallest among unused
        cand = best.copy()
        cand[sel] = 1e100
        # randomize: pick among 5 best
        idx = np.argpartition(cand, 5)[:5]
        # choose one with weights favouring smallest
        j = int(rng.choice(idx))
        # slight bias to smallest: 60% take argmin
        if rng.random() < 0.6:
            j = int(np.argmin(cand))
        sel.append(j)
        nd = pool @ pool[j]
        best = np.maximum(best, nd)
    return pool[np.array(sel)]

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = t0 + 128.0
    FINAL_END = HARD_END - 1.0
    rng = np.random.default_rng(7)
    n, d = 71, 12
    init_cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                init_cands.append(_norm_rows(a))
        except Exception:
            pass
    if not init_cands:
        init_cands.append(_norm_rows(rng.standard_normal((n, d))))
    elites = []
    scores = []
    for a in init_cands:
        try:
            elites.append(a.copy()); scores.append(_max_cos(a))
        except Exception:
            pass
    # D12 pool greedy structured seeds
    try:
        pool = _build_d12_pool(rng, want=2200)
    except Exception:
        pool = None
    # quick population: greedy + random quenches
    n_seed = 10
    for f in range(n_seed):
        if time.time() > HOP_END - 50:
            break
        try:
            if pool is not None and (f % 2 == 0):
                X0 = _greedy_from_pool(pool, rng, n=n)
                X0 = X0 + rng.standard_normal((n, d)) * 0.02
                X0 = _norm_rows(X0)
            else:
                X0 = _norm_rows(rng.standard_normal((n, d)))
            T0 = float(10 ** rng.uniform(-2.4, -1.6))
            T1 = float(10 ** rng.uniform(-4.0, -3.1))
            lr = float(10 ** rng.uniform(-1.3, -0.7))
            mu = float(rng.uniform(0.82, 0.93))
            Xq, sq = _quench_lse(X0, steps=800, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
            elites.append(_norm_rows(Xq)); scores.append(float(sq))
        except Exception:
            continue
    if len(elites) == 0:
        elites = [_norm_rows(rng.standard_normal((n, d)))]
        scores = [_max_cos(elites[0])]
    od = np.argsort(np.array(scores))
    keep = min(6, len(od))
    elites = [elites[i] for i in od[:keep]]
    scores = [float(scores[i]) for i in od[:keep]]
    best_idx = int(np.argmin(scores))
    best_X = elites[best_idx].copy(); best_score = float(scores[best_idx])
    # hopping
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 0.8:
            break
        r = rng.random()
        if r < 0.45:
            base = best_X
        elif r < 0.75 and len(elites) > 1:
            base = elites[int(rng.integers(0, len(elites)))]
        elif pool is not None and rng.random() < 0.25:
            try:
                base = _greedy_from_pool(pool, rng, n=n)
                base = _norm_rows(base + rng.standard_normal((n, d)) * 0.02)
            except Exception:
                base = _norm_rows(rng.standard_normal((n, d)))
        else:
            base = _norm_rows(rng.standard_normal((n, d)))
            # quench directly without extra perturbation
            try:
                T0 = float(10 ** rng.uniform(-2.5, -1.6)); T1 = float(10 ** rng.uniform(-4.2, -3.1))
                lr = float(10 ** rng.uniform(-1.4, -0.7)); mu = float(rng.uniform(0.82, 0.94))
                steps = 900 if tleft > 20 else 600
                X1, s1 = _quench_lse(base, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
                if np.isfinite(s1) and s1 < 0.27:
                    elites.append(_norm_rows(X1)); scores.append(float(s1))
                    if len(elites) > 8:
                        od2 = np.argsort(np.array(scores)); elites = [elites[i] for i in od2[:8]]; scores = [scores[i] for i in od2[:8]]
                    if s1 < best_score:
                        best_X = _norm_rows(X1).copy(); best_score = float(s1)
            except Exception:
                pass
            continue
        u = rng.random()
        X0 = base.copy()
        try:
            if u < 0.28:
                sigma = float(10 ** rng.uniform(-2.6, -0.4))
                X0 = base + rng.standard_normal((n, d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.62:
                k = int(rng.integers(3, 10))
                direction = rng.standard_normal(d)
                direction /= (np.linalg.norm(direction) + 1e-12)
                if rng.random() < 0.5:
                    direction = base[int(rng.integers(0, n))].copy() + rng.standard_normal(d) * 0.3
                    direction /= (np.linalg.norm(direction) + 1e-12)
                proj = base @ direction
                idx = np.argpartition(-proj, k)[:k]
                R = np.delete(base, idx, axis=0)
                for j in idx:
                    if rng.random() < 0.35:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                    else:
                        h = _find_hole(R, rng, n_sample=3500, n_cand=2)
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                keepm = np.ones(n, dtype=bool); keepm[idx] = False
                sigma = float(10 ** rng.uniform(-2.8, -1.2))
                X0[keepm] += rng.standard_normal((np.count_nonzero(keepm), d)) * sigma
                X0 = _norm_rows(X0)
            else:
                k = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(-rm + rng.random(n) * 0.002)
                idx = order[:k]
                keepm = np.ones(n, dtype=bool); keepm[idx] = False
                R = base[keepm]
                for j in idx:
                    h = _find_hole(R, rng, n_sample=4500, n_cand=3)
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                sigma = float(10 ** rng.uniform(-2.8, -1.0))
                X0[keepm] += rng.standard_normal((n - k, d)) * sigma
                X0 = _norm_rows(X0)
        except Exception:
            continue
        if tleft < 6:
            steps = 800
        elif tleft < 25:
            steps = 1100
        else:
            steps = 1300
            if rng.random() < 0.15:
                steps = 2200
        try:
            if rng.random() < 0.30:
                s_pow = float(rng.uniform(2.0, 7.0))
                lr_r = float(10 ** rng.uniform(-1.6, -0.9))
                mu_r = float(rng.uniform(0.8, 0.92))
                Xr = _quench_riesz(X0, steps=max(250, steps // 2), s_pow=s_pow, lr=lr_r, mu=mu_r, deadline=HOP_END)
                T0 = float(10 ** rng.uniform(-2.6, -1.9)); T1 = float(10 ** rng.uniform(-4.0, -3.2))
                lr2 = float(10 ** rng.uniform(-1.3, -0.8)); mu2 = float(rng.uniform(0.82, 0.93))
                X1, s1 = _quench_lse(Xr, steps=steps, T0=T0, T1=T1, lr=lr2, mu=mu2, deadline=HOP_END)
            else:
                T0 = float(10 ** rng.uniform(-2.5, -1.6)); T1 = float(10 ** rng.uniform(-4.2, -3.1))
                lr = float(10 ** rng.uniform(-1.4, -0.7)); mu = float(rng.uniform(0.82, 0.94))
                X1, s1 = _quench_lse(X0, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HOP_END)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        if s1 < 0.27:
            elites.append(_norm_rows(X1)); scores.append(float(s1))
            if len(elites) > 8:
                od2 = np.argsort(np.array(scores)); elites = [elites[i] for i in od2[:8]]; scores = [scores[i] for i in od2[:8]]
            if s1 < best_score:
                best_score = float(s1); best_X = _norm_rows(X1).copy()
    # final intensification on top 3
    try:
        od = np.argsort(np.array(scores))
        tops = [elites[i] for i in od[:min(3, len(od))]]
    except Exception:
        tops = [best_X]
    for ti, tc in enumerate(tops):
        if time.time() > FINAL_END - 12:
            break
        try:
            for rep in range(2):
                if time.time() > FINAL_END - 10:
                    break
                T0 = 0.006 if rep == 0 else 0.003
                T1 = 0.0004 if rep == 0 else 0.0002
                lr = 0.25 if rep == 0 else 0.18
                mu = 0.9
                X1, s1 = _quench_lse(_norm_rows(tc.copy()), steps=2600, T0=T0, T1=T1, lr=lr, mu=mu, deadline=FINAL_END - 8)
                if s1 < best_score and np.all(np.isfinite(X1)):
                    best_score = float(s1); best_X = _norm_rows(X1).copy()
                    tc = best_X.copy()
        except Exception:
            continue
    # LP polish best
    try:
        tleft = FINAL_END - time.time()
        if tleft > 6:
            Xp = _lp_polish(best_X, seconds=min(tleft - 1.0, 18.0), deadline=FINAL_END)
            sp = _max_cos(Xp)
            if np.all(np.isfinite(Xp)) and sp < best_score:
                best_X, best_score = _norm_rows(Xp), float(sp)
            # also try second elite polish if time
            tleft2 = FINAL_END - time.time()
            if tleft2 > 8 and len(tops) > 1:
                try:
                    X2 = _lp_polish(tops[1], seconds=min(6.0, tleft2 - 1.0), deadline=FINAL_END)
                    s2 = _max_cos(X2)
                    if s2 < best_score and np.all(np.isfinite(X2)):
                        best_X, best_score = _norm_rows(X2), float(s2)
                except Exception:
                    pass
    except Exception:
        pass
    best_X = _norm_rows(np.asarray(best_X, dtype=np.float64))
    # safety: replace non-finite rows
    nr = np.linalg.norm(best_X, axis=1)
    bad = (~np.isfinite(nr)) | (nr < 1e-12)
    if np.any(bad):
        Zr = _norm_rows(rng.standard_normal((int(np.count_nonzero(bad)), d)))
        best_X[bad] = Zr
        best_X = _norm_rows(best_X)
    return best_X

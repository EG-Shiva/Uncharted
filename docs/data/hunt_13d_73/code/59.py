import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _rand_sphere(rng, n, d):
    X = rng.standard_normal(size=(n, d))
    return _norm(X)

def _adam_lse(X0, s0, s1, steps, lr0, lr1, t_lim=None):
    n, d = X0.shape
    X = X0.copy()
    m1 = np.zeros((n, d))
    v1 = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    iu0, iu1 = np.triu_indices(n, 1)
    l0, l1 = np.log(lr0), np.log(lr1)
    ls0, ls1 = np.log(s0), np.log(s1)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    bestv = float(G.max())
    bestX = X.copy()
    WX = np.empty((n, d))
    for k in range(steps):
        if t_lim is not None and (k & 127) == 0 and time.time() > t_lim:
            break
        f = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * f)
        lr = np.exp(l0 + (l1 - l0) * f)
        G = X @ X.T
        c = G[iu0, iu1]
        mx = float(c.max())
        if mx < bestv:
            bestv = mx
            bestX = X.copy()
        e = np.exp(s * (c - mx))
        se = float(e.sum())
        if not np.isfinite(se) or se <= 0:
            w = np.zeros_like(c)
            w[int(np.argmax(c))] = 1.0
        else:
            w = e / se
        WX.fill(0.0)
        np.add.at(WX, iu0, w[:, None] * X[iu1])
        np.add.at(WX, iu1, w[:, None] * X[iu0])
        sp = np.sum(WX * X, axis=1)
        Gt = WX - sp[:, None] * X
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        mh = m1 / (1 - b1p)
        vh = v1 / (1 - b2p)
        b1p *= b1
        b2p *= b2
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    fv = _max_cos(X)
    if fv < bestv:
        bestv = fv
        bestX = X.copy()
    return bestX, bestv

def _hole_vec(X, rng, n_try=2000):
    n, d = X.shape
    C = rng.standard_normal(size=(n_try, d))
    C = _norm(C)
    S = C @ X.T
    m = S.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(X, best, t_deadline, max_iters=14):
    X = _norm(X)
    n, d = X.shape
    radius = 6e-4
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while it < max_iters and time.time() < t_deadline and radius > 1e-14:
        it += 1
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(25.0 * radius, 2e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 1900:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:1900]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        vv = _max_cos(y)
        if vv < best - 1e-12:
            X, best = y, vv
            radius = min(radius * 1.6, 0.01)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 140.0
    MID_END = t0 + 152.0
    LP_END = t0 + 169.0
    rng = np.random.default_rng(20261102)
    n, d = 73, 13
    Xinc = None
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.asarray(np.load(fn), dtype=np.float64)
            if v.shape == (n, d):
                v = _norm(v)
                if Xinc is None or _max_cos(v) < _max_cos(Xinc):
                    Xinc = v.copy()
        except Exception:
            continue
    if Xinc is None:
        Xinc = _rand_sphere(rng, n, d)
    best_val = _max_cos(Xinc)
    Xbest = Xinc.copy()
    # Phase A: cheap diverse screening
    seeds = []
    seeds.append(Xinc.copy())
    for _ in range(3):
        seeds.append(_norm(Xinc + 0.01 * rng.standard_normal(size=(n, d))))
    for _ in range(5):
        seeds.append(_rand_sphere(rng, n, d))
    pool = [(best_val, Xbest.copy())]
    for si, S0 in enumerate(seeds):
        if time.time() > t0 + 30 or time.time() > HARD_END - 100:
            break
        try:
            if si == 0:
                Xa, va = _adam_lse(S0, 60.0, 450.0, 700, 0.015, 0.002, t_lim=SEARCH_END)
            elif si <= 3:
                Xa, va = _adam_lse(S0, 50.0, 500.0, 600, 0.018, 0.0025, t_lim=SEARCH_END)
            else:
                Xa, va = _adam_lse(S0, 40.0, 350.0, 600, 0.02, 0.003, t_lim=SEARCH_END)
            pool.append((va, Xa.copy()))
            if va < best_val:
                best_val = va
                Xbest = Xa.copy()
        except Exception:
            continue
    pool.sort(key=lambda t: t[0])
    pool = pool[:4]
    # Phase B: basin hopping with kicks + hole + rattler moves
    itcount = 0
    while time.time() < SEARCH_END and time.time() < HARD_END - 35:
        itcount += 1
        # pick base: mostly best, sometimes pool member
        if itcount % 7 == 0 and len(pool) > 1:
            base = pool[rng.integers(1, len(pool))][1]
        else:
            base = Xbest
        mode = itcount % 5
        try:
            Xp = base.copy()
            if mode == 0:
                sg = 0.008 + 0.02 * rng.random()
                Xp = _norm(base + sg * rng.standard_normal(size=(n, d)))
            elif mode == 1:
                sg = 0.025 + 0.05 * rng.random()
                Xp = _norm(base + sg * rng.standard_normal(size=(n, d)))
            elif mode == 2:
                # worst-point hole reinsertion
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                pw = G.max(axis=1)
                iw = int(np.argmax(pw))
                hv = _hole_vec(base, rng, n_try=1500)
                Xp[iw] = hv
                # small jitter others
                Xp = _norm(Xp + 0.003 * rng.standard_normal(size=(n, d)))
            elif mode == 3:
                # rattler randomization
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mcur = float(G.max())
                pw = G.max(axis=1)
                ratt = np.where(pw < mcur - 8e-4)[0]
                if len(ratt) >= 3:
                    k = min(len(ratt), 4 + int(rng.integers(0, 4)))
                    ch = rng.choice(ratt, size=k, replace=False)
                    R = _rand_sphere(rng, k, d)
                    Xp[ch] = R
                    Xp = _norm(Xp + 0.002 * rng.standard_normal(size=(n, d)))
                else:
                    Xp = _norm(base + 0.015 * rng.standard_normal(size=(n, d)))
            else:
                # two-point hole: replace worst two
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                pw = G.max(axis=1)
                ordp = np.argsort(-pw)[:2]
                Xtmp = base.copy()
                for iw in ordp:
                    hv = _hole_vec(Xtmp, rng, n_try=1200)
                    Xtmp[iw] = hv
                Xp = _norm(Xtmp + 0.002 * rng.standard_normal(size=(n, d)))
            # choose anneal at random
            r = rng.random()
            if r < 0.4:
                s0, s1, lr0, lr1, steps = 60.0, 500.0, 0.015, 0.002, 650
            elif r < 0.75:
                s0, s1, lr0, lr1, steps = 120.0, 700.0, 0.010, 0.0015, 700
            else:
                s0, s1, lr0, lr1, steps = 200.0, 900.0, 0.008, 0.001, 700
            Xa, va = _adam_lse(Xp, s0, s1, steps, lr0, lr1, t_lim=SEARCH_END)
            if va < best_val:
                best_val = va
                Xbest = Xa.copy()
                pool.append((va, Xa.copy()))
                pool.sort(key=lambda t: t[0])
                pool = pool[:4]
            elif va < pool[-1][0] and rng.random() < 0.3:
                pool.append((va, Xa.copy()))
                pool.sort(key=lambda t: t[0])
                pool = pool[:4]
        except Exception:
            continue
    # Phase C: long equalizing anneal on best
    try:
        if time.time() < MID_END:
            Xa, va = _adam_lse(Xbest, 30.0, 550.0, 1800, 0.008, 0.0012, t_lim=MID_END)
            if va < best_val:
                best_val = va
                Xbest = Xa.copy()
    except Exception:
        pass
    try:
        if time.time() < MID_END:
            Xb, vb = _adam_lse(Xbest, 150.0, 1000.0, 1200, 0.005, 0.0008, t_lim=MID_END)
            if vb < best_val:
                best_val = vb
                Xbest = Xb.copy()
    except Exception:
        pass
    # final LP squeeze
    try:
        if time.time() < LP_END - 3:
            Xbest, best_val = _lp_polish(Xbest, best_val, LP_END, max_iters=14)
    except Exception:
        pass
    Xbest = _norm(Xbest)
    # tiny shrink guard (no-op, keep unit rows)
    return Xbest

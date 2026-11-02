import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < 120:
        it += 1
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            # keep only tightest 1500 to keep LP fast
            order = np.argsort(c[sel])[-1500:]
            I, J = I[order], J[order]
            sel_idx = np.where(sel)[0][order]
            csel = c[sel_idx]
            p = len(I)
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def fast_quench(Y, steps=170, b0=80.0, b1=600.0, s0=0.04, s1=0.004):
    n, d = Y.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = b0 * (b1 / b0) ** frac
        step = s0 * (s1 / s0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # avoid overflow: beta*(G-m) <= 0
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Grad = W @ Y
        dot = (Grad * Y).sum(axis=1, keepdims=True)
        Gt = Grad - dot * Y
        mn = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(mn) or mn < 1e-14:
            continue
        Y = Y - (step / mn) * Gt
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Y = Y / nr
    return Y

def hole_vector(X, idx, rng, n_cand=1500):
    n, d = X.shape
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Rest = X[mask]
    dots = C @ Rest.T
    scores = dots.max(axis=1)
    bi = int(np.argmin(scores))
    return C[bi]

def solve():
    t0 = time.time()
    LIMIT = 165.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    # quick snap of starting point
    if time.time() - t0 < LIMIT - 30:
        try:
            bestX = lp_polish(bestX, min(3.0, max(0.5, LIMIT - 30 - (time.time() - t0))))
            bestV = max_cos(bestX)
        except Exception:
            pass
    it = 0
    # precompute triu for contact eval
    while time.time() - t0 < LIMIT - 13:
        it += 1
        time_left = LIMIT - 13 - (time.time() - t0)
        if time_left < 2.0:
            break
        # contact counts
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        cnts = (G > mcur - 1e-9).sum(axis=1)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 5
        if mode == 4:
            # global shake to new basin
            sigma = float(rng.uniform(0.10, 0.25))
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 150
        else:
            if mode == 0:
                k = 1
            elif mode in (1, 3):
                k = 2
            else:
                k = 3
            pool = order[:16]
            # bias to lowest contacts
            chosen = []
            # always include the very lowest with high prob
            perm = rng.permutation(pool)[:k]
            chosen = perm
            for idx in chosen:
                hv = hole_vector(Xtry, int(idx), rng, n_cand=1200 if time_left > 20 else 600)
                # small jitter around hole to avoid exact ties
                hv = hv + 0.002 * rng.standard_normal(d)
                hv /= np.linalg.norm(hv)
                Xtry[int(idx)] = hv
            # shake rest slightly
            Xtry = Xtry + rng.standard_normal((n, d)) * 0.006
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 170 if time_left > 20 else 110
        # fast quench
        Xq = fast_quench(Xtry, steps=steps, b0=80.0, b1=650.0)
        vq = max_cos(Xq)
        # only LP-polish if promising (saves huge time)
        if vq < bestV + 0.0015:
            budget = 1.6 if time_left > 15 else 0.7
            try:
                Xp = lp_polish(Xq, min(budget, time_left - 0.4))
            except Exception:
                Xp = Xq
            vp = max_cos(Xp)
            if vp < bestV - 1e-12:
                bestX = Xp
                bestV = vp
        # opportunistic accept of near-equal with different topology? keep best only
        if time.time() - t0 > LIMIT - 13:
            break
    # final intensive polish
    try:
        remain = LIMIT - (time.time() - t0) - 0.8
        if remain > 1.5:
            bestX = lp_polish(bestX, min(remain, 11.0))
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

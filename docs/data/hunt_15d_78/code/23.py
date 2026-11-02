import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def lp_polish(X, t_end, r0=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    # preallocate index helpers
    ar_d = np.arange(d)
    while time.time() < t_end and radius > 1e-13:
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        thr = max(20.0 * radius, 1e-8)
        sel = c > m - thr
        if not np.any(sel):
            break
        # cap constraint count for speed
        idx = np.where(sel)[0]
        if idx.size > 2500:
            # keep largest 2500
            part = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[part]
        I = iu0[0][idx]
        J = iu0[1][idx]
        p = idx.size
        b = -c[idx]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        Ioff = I[:, None] * d + ar_d[None, :]
        Joff = J[:, None] * d + ar_d[None, :]
        cols = np.concatenate([Ioff, Joff, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=np.zeros(n),
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
        # early exit if time nearly up
        if time.time() >= t_end:
            break
    return X, best


def void_best(X_rest, rng, n_cand, d):
    cands = rng.standard_normal((n_cand, d))
    cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    dots = cands @ X_rest.T
    scores = dots.max(axis=1)
    bi = int(np.argmin(scores))
    return cands[bi]


def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = t0 + 155.0
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
            if np.any(~np.isfinite(a)):
                continue
            v = max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        bestX = a
        bestV = max_cos(a)

    # initial snap
    try:
        bestX, bestV = lp_polish(bestX, min(t0 + 8.0, HARD_END - 5.0), r0=1e-3)
    except Exception:
        pass

    it = 0
    while time.time() < HOP_END and time.time() < HARD_END - 6.0:
        it += 1
        # contact counts for victim selection
        try:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            cnts = (G > mcur - 1e-7).sum(axis=1)
        except Exception:
            cnts = np.zeros(n)
        order = np.argsort(cnts, kind="stable")
        mode = it % 5
        Xtry = bestX.copy()
        try:
            if mode == 0:
                # global kick, keep topology change
                sigma = 0.14 if (it // 5) % 2 == 0 else 0.07
                Xtry = bestX + rng.standard_normal((n, d)) * sigma
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            else:
                if mode == 1:
                    k = 1
                    nc = 600
                    shake = 0.004
                elif mode == 2:
                    k = 2
                    nc = 400
                    shake = 0.005
                elif mode == 3:
                    k = 3
                    nc = 300
                    shake = 0.006
                else:
                    k = 1
                    nc = 400
                    shake = 0.010
                # victims: mostly low-contact tail, sometimes random
                pool = order[:16]
                if mode == 4 and (rng.random() < 0.4):
                    chosen = rng.choice(n, size=k, replace=False)
                else:
                    chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
                chosen = sorted([int(c) for c in chosen])
                for idx in chosen:
                    mask = np.ones(n, dtype=bool)
                    mask[idx] = False
                    Xr = Xtry[mask]
                    bv = void_best(Xr, rng, nc, d)
                    Xtry[idx] = bv
                if shake > 0:
                    Xtry = Xtry + rng.standard_normal((n, d)) * shake
                    Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        except Exception:
            continue

        budget = 2.4
        # shorter budget when time is low
        if HARD_END - time.time() < 12:
            budget = 1.2
        t_end = min(time.time() + budget, HARD_END - 4.0, HOP_END + 6.0)
        if t_end - time.time() < 0.4:
            break
        try:
            Xpol, v = lp_polish(Xtry, t_end, r0=8e-4)
        except Exception:
            continue
        if v < bestV - 1e-12:
            bestX, bestV = Xpol, v

    # final intensive polish
    try:
        remain = HARD_END - time.time() - 0.5
        if remain > 1.0:
            bestX, bestV = lp_polish(bestX, time.time() + min(remain, 16.0), r0=8e-4)
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

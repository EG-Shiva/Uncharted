import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, deadline, r_init=1e-3, r_max=0.03, cap=2500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)]*(n*d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs",
                          options={"time_limit": max(1.0, deadline - time.time())})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius*1.5, r_max)
        else:
            radius *= 0.5
    return best_x, best

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    # short initial snap
    if time.time() < HARD - 100:
        bestX, bestV = _lp_polish(bestX, min(HARD - 100, time.time()+6.0))
    it = 0
    # tail-targeted basin hopping: reinsert low-contact points into voids
    while time.time() < HARD - 12:
        it += 1
        time_left = HARD - 12 - time.time()
        if time_left < 3:
            break
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnts = (G > m - 1e-9).sum(axis=1)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        if it % 3 == 0:
            # global small shake to probe nearby basin
            sig = 0.02 if (it % 6 == 0) else 0.008
            Xtry = bestX + rng.standard_normal((n, d)) * sig
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            # pick 1-2 loose points
            k = 1 if (it % 3 == 1) else 2
            pool = order[:16]
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for idx in chosen:
                idx = int(idx)
                # sample candidates, keep farthest from rest
                nc = 400
                cands = rng.standard_normal((nc, d))
                cands /= np.linalg.norm(cands, axis=1, keepdims=True)
                dots = cands @ Xtry.T
                dots[:, idx] = -2.0
                scores = dots.max(axis=1)
                bi = int(np.argmin(scores))
                Xtry[idx] = cands[bi]
            # tiny shake of rest to break symmetry
            Xtry = Xtry + rng.standard_normal((n, d)) * 0.004
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        budget = 2.0 if time_left > 15 else 1.0
        Xpol, v = _lp_polish(Xtry, time.time()+budget)
        if v < bestV - 1e-12:
            bestX = Xpol; bestV = v
    # final intensive polish of champion
    try:
        remain = HARD - time.time() - 0.5
        if remain > 1.5:
            bestX, bestV = _lp_polish(bestX, time.time()+min(remain, 9.0))
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

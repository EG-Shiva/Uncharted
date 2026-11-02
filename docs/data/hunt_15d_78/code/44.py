import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _near_count(X, delta=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    return int((G > m - delta).sum() // 2), m

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:  # subsample worst to keep LP cheap
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            ss = np.where(sel)[0][idx]
            csel = c[ss]
            p = len(I)
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _adam_quench(X0, steps=600, lr0=0.006):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    iu0, iu1 = np.triu_indices(n, 1)
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    t = 0
    for s in range(steps):
        beta = 50.0 + (400.0 - 50.0) * s / max(1, steps - 1)
        lr = lr0 * (1.0 - 0.75 * s / steps)
        G = X @ X.T
        gv = G[iu0, iu1]
        mx = gv.max()
        e = np.exp(beta * (gv - mx))
        S = e.sum()
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Rg = Gr - dot * X
        t += 1
        M = 0.9 * M + 0.1 * Rg
        V = 0.999 * V + 0.001 * (Rg * Rg)
        Mh = M / (1.0 - 0.9 ** t)
        Vh = V / (1.0 - 0.999 ** t)
        step = lr * Mh / (np.sqrt(Vh) + 1e-8)
        sdot = (step * X).sum(axis=1, keepdims=True)
        step_t = step - sdot * X
        # clip extreme steps
        nm = np.linalg.norm(step_t, axis=1, keepdims=True)
        cap = 0.05
        scale = np.minimum(1.0, cap / np.maximum(nm, 1e-12))
        step_t = step_t * scale
        X = X - step_t
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD = 150.0
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incX = bestX.copy()
    incV = bestV
    try:
        incNC, _ = _near_count(incX, 1e-4)
    except Exception:
        incNC = 10**9

    configs = [
        (12, 0.010, 1e-3, 101), (14, 0.012, 1e-3, 102), (10, 0.008, 1e-3, 103),
        (12, 0.006, 3e-4, 104), (14, 0.008, 3e-4, 105), (10, 0.015, 1e-3, 106),
        (11, 0.010, 1e-3, 107), (13, 0.012, 5e-4, 108), (12, 0.015, 1e-3, 109),
        (10, 0.005, 1e-3, 110), (14, 0.005, 1e-3, 111), (12, 0.010, 5e-4, 112),
    ]
    for ci, (k, sigma, tol, seed) in enumerate(configs):
        if time.time() - t0 > HARD - 12:
            break
        try:
            G = incX @ incX.T
            np.fill_diagonal(G, -2.0)
            m0 = G.max()
            A = (G > m0 - tol)
            deg = A.sum(axis=1)
            # personal max for tie-break
            G2 = incX @ incX.T
            np.fill_diagonal(G2, -2.0)
            pmax = G2.max(axis=1)
            # start from highest degree (tie: highest pmax)
            order = np.lexsort((-pmax, -deg))
            start = int(order[0])
            cluster = set([start])
            # expand greedily: pick non-member with most links to cluster, tie by deg
            while len(cluster) < k:
                members = list(cluster)
                links = A[:, members].sum(axis=1)
                cand = [(links[i], deg[i], pmax[i], i) for i in range(n) if i not in cluster]
                # require connectivity if possible
                conn = [c for c in cand if c[0] > 0]
                pool_c = conn if len(conn) > 0 else cand
                pool_c.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
                # take top, but add slight seed-dependent shuffle among top-5
                top = pool_c[:5]
                pick = top[(seed + len(cluster)) % len(top)][3]
                cluster.add(int(pick))
            cluster = sorted(cluster)
            kept = [i for i in range(n) if i not in set(cluster)]
            rng = np.random.default_rng(seed * 100003 + 7)
            Xkept = incX[kept].copy()
            Xkept = Xkept + sigma * rng.standard_normal(Xkept.shape)
            Xkept /= np.linalg.norm(Xkept, axis=1, keepdims=True)
            # greedy reinsert from 8192 candidates
            pool = rng.standard_normal((8192, d))
            pool /= np.linalg.norm(pool, axis=1, keepdims=True)
            # scores vs kept
            S = pool @ Xkept.T
            smax = S.max(axis=1)
            used = np.zeros(8192, dtype=bool)
            chosen = []
            for _ in range(k):
                smax_masked = np.where(used, np.inf, smax)
                bi = int(np.argmin(smax_masked))
                used[bi] = True
                chosen.append(pool[bi])
                # update with new point
                dp = pool @ chosen[-1]
                smax = np.maximum(smax, dp)
            Xtry = np.empty((n, d), dtype=np.float64)
            Xtry[kept] = Xkept
            Xtry[cluster] = np.array(chosen)
            # cheap Adam quench ~600 steps
            Xq = _adam_quench(Xtry, steps=600, lr0=0.006)
            vq = _max_cos(Xq)
            if vq > 0.175:
                continue
            # LP polish only promising
            if vq > incV + 0.012:
                continue
            tleft = HARD - (time.time() - t0)
            if tleft < 6:
                # no time for LP; accept raw if better
                if vq < incV - 1e-12:
                    incX, incV = Xq, vq
                    incNC, _ = _near_count(incX, 1e-4)
                continue
            lp_sec = min(5.0, max(1.5, tleft - 6))
            Xp = _lp_polish(Xq, lp_sec)
            vp = _max_cos(Xp)
            nc, _ = _near_count(Xp, 1e-4)
            if vp < incV - 1e-12 or (abs(vp - incV) <= 1e-12 and nc < incNC):
                incX, incV, incNC = Xp, vp, nc
        except Exception:
            continue
    # final safety: return incumbent (already valid)
    try:
        incX = incX / np.linalg.norm(incX, axis=1, keepdims=True)
        if not np.all(np.isfinite(incX)):
            return bestX
        return incX
    except Exception:
        return bestX

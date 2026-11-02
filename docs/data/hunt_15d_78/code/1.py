import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def contact_counts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = (G > m - tol)
    # counts per point (each contact counted once per endpoint)
    return C.sum(axis=1), m, G


def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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


def smooth_refine(X0, betas, maxiter_each=250):
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Xcur = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xflat = Xcur.ravel().copy()
    for beta in betas:
        def fun(g):
            X = g.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            gv = G[iu0]
            m = gv.max()
            # stable log-sum-exp
            e = np.exp(beta * (gv - m))
            S = e.sum()
            F = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0] = w
            W = W + W.T
            GradY = W @ Y
            dot = (GradY * Y).sum(axis=1, keepdims=True)
            GradX = (GradY - dot * Y) / nr
            return float(F), GradX.ravel()
        try:
            res = minimize(fun, xflat, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-7})
            xflat = res.x
        except Exception:
            break
    Xopt = xflat.reshape(n, d)
    Xopt = Xopt / np.linalg.norm(Xopt, axis=1, keepdims=True)
    return Xopt


def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(12345)
    # load start
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        a = rng.standard_normal((78, 15))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        bestX = a
        bestV = max_cos(a)
    n, d = 78, 15
    # short initial snap (already polished, keep tiny)
    if time.time() - t0 < LIMIT - 150:
        bestX = lp_polish(bestX, min(3.0, max(0.5, LIMIT - (time.time() - t0) - 150)))
        bestV = max_cos(bestX)

    it = 0
    # main basin-hopping
    while time.time() - t0 < LIMIT - 18:
        it += 1
        time_left = LIMIT - 18 - (time.time() - t0)
        if time_left < 3:
            break
        cnts, mcur, _ = contact_counts(bestX, tol=1e-9)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 4
        if mode == 3:
            sigma = float(rng.uniform(0.12, 0.32))
            noise = rng.standard_normal((n, d))
            Xtry = bestX + sigma * noise
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            if mode == 0:
                k = 1
            elif mode == 1:
                k = 2
            else:
                k = 3
            pool = order[:14]
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for idx in chosen:
                # 24 candidates, pick farthest from rest
                cands = rng.standard_normal((24, d))
                cands /= np.linalg.norm(cands, axis=1, keepdims=True)
                # max dot vs all except idx
                Gc = cands @ Xtry.T
                Gc[:, int(idx)] = -2.0
                scores = Gc.max(axis=1)
                bi = int(np.argmin(scores))
                # small random tie-break: with prob take 2nd best to diversify
                Xtry[int(idx)] = cands[bi]
            # small shake of the rest to break symmetry
            Xtry = Xtry + rng.standard_normal((n, d)) * 0.008
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # smooth annealing: two betas
        try:
            Xsm = smooth_refine(Xtry, betas=[120.0, 450.0], maxiter_each=220)
        except Exception:
            continue
        # quick LP snap with small budget
        budget = 2.2 if time_left > 10 else 1.0
        try:
            Xpol = lp_polish(Xsm, min(budget, time_left - 0.5))
        except Exception:
            Xpol = Xsm
        v = max_cos(Xpol)
        if v < bestV - 1e-12:
            bestX = Xpol
            bestV = v
        # timeout guard
        if time.time() - t0 > LIMIT - 18:
            break

    # final intensive polish of champion
    try:
        remain = LIMIT - (time.time() - t0) - 1.0
        if remain > 2:
            bestX = lp_polish(bestX, min(remain, 14.0))
            # one more smooth+polish at high beta for tie cleanup
            remain2 = LIMIT - (time.time() - t0) - 0.5
            if remain2 > 4:
                Xsm = smooth_refine(bestX, betas=[600.0], maxiter_each=200)
                v1 = max_cos(bestX)
                # polish both and keep better
                Xpol = lp_polish(Xsm, min(4.0, remain2 - 1.0))
                if max_cos(Xpol) < v1:
                    bestX = Xpol
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # sanitize
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

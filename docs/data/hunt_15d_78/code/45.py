import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds, max_it=12):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    t0 = time.time()
    radius = 2e-3
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < seconds and radius > 1e-13 and it < max_it:
        it += 1
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu0[sel], iu1[sel]
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
            tl = max(0.3, seconds - (time.time() - t0))
            # limit per-solve time
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(tl, 2.0)})
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


def _smooth_quench(X0, betas, maxiter_each, iu0, iu1):
    n, d = X0.shape
    xflat = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()

    def make_fun(beta):
        def fun(g):
            X = g.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            gv = (Y @ Y.T)[iu0, iu1]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            S = e.sum()
            F = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ Y
            dot = (Gr * Y).sum(axis=1, keepdims=True)
            GY = (Gr - dot * Y) / nr
            return float(F), GY.ravel()
        return fun

    for beta in betas:
        try:
            r = minimize(make_fun(beta), xflat, jac=True, method="L-BFGS-B",
                         options={"maxiter": maxiter_each, "ftol": 1e-14, "gtol": 1e-6})
            xflat = r.x
        except Exception:
            break
    Xopt = xflat.reshape(n, d)
    Xopt /= np.linalg.norm(Xopt, axis=1, keepdims=True)
    return Xopt


def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    iu0, iu1 = np.triu_indices(n, 1)

    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    # quick snap of starting point
    if time.time() - t0 < LIMIT - 30:
        bestX = _lp_polish(bestX, 2.0, max_it=10)
        bestV = _max_cos(bestX)

    it = 0
    while time.time() - t0 < LIMIT - 14:
        it += 1
        time_left = LIMIT - 14 - (time.time() - t0)
        if time_left < 4:
            break
        Xtry = bestX.copy()
        mode = it % 5
        if mode == 1:
            # partial whitening / isotropizing kick
            C = (Xtry.T @ Xtry) / n
            w, V = np.linalg.eigh(C)
            w = np.maximum(w, 1e-12)
            target = 1.0 / d
            scale = np.sqrt(target / w)
            scale = np.clip(scale, 0.5, 2.0)
            A = (V * scale) @ V.T
            alpha = 0.45
            Abl = (1 - alpha) * np.eye(d) + alpha * A
            Xtry = Xtry @ Abl
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xtry += 0.025 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 2:
            # void reinsertion of low-contact points
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnts = (G > m - 1e-9).sum(axis=1)
            order = np.argsort(cnts, kind="stable")
            k = 4 if (it % 2 == 0) else 3
            victims = order[:k]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Xkeep = Xtry[keep]
            for vi in victims:
                cand = rng.standard_normal((1500, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xkeep.T
                worst = dots.max(axis=1)
                Xtry[vi] = cand[np.argmin(worst)]
                Xkeep = np.vstack([Xkeep, Xtry[vi:vi+1]])
            Xtry += 0.01 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 3:
            # hub shake: perturb high-contact points more
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnts = (G > m - 1e-9).sum(axis=1)
            order = np.argsort(cnts, kind="stable")
            hubs = order[-6:]
            Xtry[hubs] += 0.09 * rng.standard_normal((len(hubs), d))
            mask = np.ones(n, dtype=bool)
            mask[hubs] = False
            Xtry[mask] += 0.012 * rng.standard_normal((mask.sum(), d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 4:
            # moderate global shake
            Xtry += 0.05 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            # random subset strong shake
            k = int(rng.integers(6, 11))
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] += 0.16 * rng.standard_normal((k, d))
            Xtry += 0.008 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # quench with rising beta, cheap
        betas = [60.0, 160.0] if (it % 3 != 0) else [60.0, 160.0, 350.0]
        Xq = _smooth_quench(Xtry, betas, 60, iu0, iu1)
        vq = _max_cos(Xq)
        # only LP-polish if promising
        if vq < bestV + 0.004 and time.time() - t0 < LIMIT - 16:
            Xq = _lp_polish(Xq, 1.2, max_it=8)
            vq = _max_cos(Xq)
        if vq < bestV:
            bestX, bestV = Xq, vq
        # occasional sideways acceptance to diversify (very strict)
        elif vq < bestV + 0.001 and (it % 7 == 0) and time.time() - t0 < LIMIT - 40:
            # keep as new base for one step without overwriting best? use Xtry chain
            # do a second quench from here next iter by moving base slightly toward it
            bestX = bestX * 0.7 + Xq * 0.3
            # re-project rows
            bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
            # do not update bestV

        if bestV < 0.1665:
            break

    # final polish of best
    remain = LIMIT - (time.time() - t0) - 1.0
    if remain > 1:
        bestX = _lp_polish(bestX, min(remain, 9.0), max_it=20)
        bestV = _max_cos(bestX)
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

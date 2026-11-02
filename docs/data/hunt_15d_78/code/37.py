import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0.05:
        return X
    start = time.time()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[sel]; J = iu1[sel]
        # cap constraints to tightest for speed
        if len(I) > 700:
            order = np.argsort(-c[sel])[:700]
            I = I[order]; J = J[order]
            cc = c[sel][order]
        else:
            cc = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X


def _split_clusters(X):
    # PC1 split, fallback to index split 0..56 / rest
    try:
        _, _, Vt = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
        v1 = Vt[0]
        scores = X @ v1
        # orient so that majority (first 57) is negative
        if scores[:57].mean() > scores[57:].mean():
            v1 = -v1
            scores = -scores
        maskA = scores < -0.05
        if maskA.sum() >= 10 and maskA.sum() <= 68:
            idxA = np.where(maskA)[0]
            idxB = np.where(~maskA)[0]
            return idxA, idxB
    except Exception:
        pass
    idxA = np.arange(57)
    idxB = np.arange(57, 78)
    return idxA, idxB


def _build_planes(XA, XB, rng, n_prin=4, n_rand=2):
    M = XA.T @ XB  # 15x15 cross covariance
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        V = Vt.T
    except Exception:
        U = np.eye(15); V = np.eye(15)
    planes = []
    for k in range(n_prin):
        u = U[:, k].copy()
        u /= max(np.linalg.norm(u), 1e-12)
        vk = V[:, k].copy()
        w = vk - float(u @ vk) * u
        nw = np.linalg.norm(w)
        if nw < 1e-8:
            r = rng.standard_normal(15)
            r -= float(r @ u) * u
            r /= max(np.linalg.norm(r), 1e-12)
            v = r
        else:
            v = w / nw
        planes.append((u, v))
    for _ in range(n_rand):
        a = rng.standard_normal(15)
        a /= max(np.linalg.norm(a), 1e-12)
        b = rng.standard_normal(15)
        b = b - float(b @ a) * a
        nb = np.linalg.norm(b)
        if nb < 1e-8:
            b = rng.standard_normal(15)
            b = b - float(b @ a) * a
            nb = np.linalg.norm(b)
        b /= max(nb, 1e-12)
        planes.append((a, b))
    return planes


def _apply_thetas(Xbase, idx, planes, thetas):
    X = Xbase.copy()
    S = X[idx]  # view copy
    S = S.copy()
    for (u, v), th in zip(planes, thetas):
        th = float(np.clip(th, -0.35, 0.35))
        c = np.cos(th); s = np.sin(th)
        if abs(th) < 1e-15:
            continue
        au = S @ u
        av = S @ v
        cu = (c - 1.0) * au - s * av
        cv = (c - 1.0) * av + s * au
        S = S + cu[:, None] * u[None, :] + cv[:, None] * v[None, :]
    # renormalize (rotation preserves norm up to fp error)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    X[idx] = S
    return X


def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        a = rng.standard_normal((78, 15))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        bestX = a
        bestV = _max_cos(a)

    # tiny initial snap if time allows
    if time.time() - t0 < LIMIT - 120:
        bestX = _lp_polish(bestX, min(2.0, max(0.5, LIMIT - (time.time() - t0) - 120)))
        bestV = _max_cos(bestX)

    # alternate collective rotations of B and A
    nrounds = 4
    for rnd in range(nrounds):
        if time.time() - t0 > LIMIT - 20:
            break
        for side in ["B", "A"]:
            if time.time() - t0 > LIMIT - 18:
                break
            idxA, idxB = _split_clusters(bestX)
            idx = idxB if side == "B" else idxA
            XA = bestX[idxA]; XB = bestX[idxB]
            # fresh planes each move: top principal pairs + random cross pairs
            n_prin = 4 if rnd < 2 else 3
            n_rand = 2 if rnd < 2 else 3
            planes = _build_planes(XA, XB, rng, n_prin=n_prin, n_rand=n_rand)
            K = len(planes)
            curV = _max_cos(bestX)

            def obj(th):
                Xn = _apply_thetas(bestX, idx, planes, th)
                G = Xn @ Xn.T
                np.fill_diagonal(G, -2.0)
                return float(G.max())

            x0 = np.zeros(K)
            try:
                res = minimize(obj, x0, method="Nelder-Mead",
                               options={"maxiter": 90, "maxfev": 320,
                                        "xatol": 1e-3, "fatol": 1e-9,
                                        "initial_simplex": None})
                th_best = np.clip(np.asarray(res.x).ravel(), -0.35, 0.35)
            except Exception:
                th_best = x0
            # also try a few random angle draws to seed different basin
            Xn_best = _apply_thetas(bestX, idx, planes, th_best)
            v_best = _max_cos(Xn_best)
            # random shots (cheap Gram evals)
            for _ in range(40):
                if time.time() - t0 > LIMIT - 15:
                    break
                th = rng.uniform(-0.35, 0.35, size=K)
                v = obj(th)
                if v < v_best:
                    v_best = v
                    th_best = th.copy()
                    Xn_best = _apply_thetas(bestX, idx, planes, th_best)
            if v_best < curV - 1e-12:
                # short LP polish per accepted rotation
                budget = 1.6 if (LIMIT - (time.time() - t0)) > 12 else 0.8
                try:
                    Xpol = _lp_polish(Xn_best, budget)
                except Exception:
                    Xpol = Xn_best
                vpol = _max_cos(Xpol)
                if vpol < bestV:
                    bestX = Xpol
                    bestV = vpol
                elif v_best < bestV:
                    # keep unpolished if polish hurt (rare)
                    bestX = Xn_best / np.linalg.norm(Xn_best, axis=1, keepdims=True)
                    bestV = v_best
                    # re-polish lightly to snap
                    try:
                        Xpol2 = _lp_polish(bestX, 0.8)
                        v2 = _max_cos(Xpol2)
                        if v2 < bestV:
                            bestX, bestV = Xpol2, v2
                    except Exception:
                        pass
            else:
                # no direct win: try polish of best rotated anyway with tiny budget
                # only once per round to save time
                pass

    # final intensive polish
    try:
        remain = LIMIT - (time.time() - t0) - 1.0
        if remain > 2:
            bestX = _lp_polish(bestX, min(remain, 12.0))
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

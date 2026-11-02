import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def anneal_opt(X, schedule):
    n, d = X.shape
    v = np.zeros_like(X)
    mu = 0.9
    for beta, iters, lr0, lr1 in schedule:
        v.fill(0.0)
        inv_b = 12.0 / beta
        for it in range(iters):
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            curmax = float(np.max(G))
            thr = curmax - inv_b
            cmin = curmax - 0.03
            if thr < cmin:
                thr = cmin
            mask = G >= thr
            W = np.zeros_like(G)
            # exp(beta*(G-curmax)) on mask (<=0 argument, safe)
            Gm = G[mask]
            W[mask] = np.exp(beta * (Gm - curmax))
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            tg = grad - dot * X
            mn = float(np.max(np.sqrt(np.sum(tg * tg, axis=1))))
            if mn < 1e-14:
                continue
            tg /= mn
            v = mu * v + (1.0 - mu) * tg
            frac = it / max(1, iters - 1)
            lr = lr0 + (lr1 - lr0) * frac
            X = X - lr * v
            X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X


def lp_polish(x, seconds, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius0)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iu_r = iu0[0]
    iu_c = iu0[1]
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1.0:
            break
        Gfull = x @ x.T
        c = Gfull[iu_r, iu_c]
        m = float(np.max(c))
        tol = 20.0 * radius
        if tol < 1e-9:
            tol = 1e-9
        sel = c > m - tol
        I = iu_r[sel]
        J = iu_c[sel]
        p = int(I.size)
        if p == 0:
            radius *= 0.5
            continue
        # cap constraints for speed
        if p > 6000:
            # keep largest 6000
            cs = c[sel]
            kp = np.argpartition(-cs, 6000)[:6000]
            I = I[kp]
            J = J[kp]
            p = 6000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + ar, J[:, None] * d + ar,
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != int(np.sum(sel)) else -c[sel],
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x = y
            best = vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(0)
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(b)
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(r)
    except Exception:
        pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            cc = np.asarray(cc, dtype=np.float64)
            if cc.shape != (453, 5):
                continue
            nrm = np.linalg.norm(cc, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            cc = cc / nrm
            if not np.all(np.isfinite(cc)):
                continue
            vv = max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.normal(size=(453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)

    sched = [
        (30.0, 250, 0.005, 0.001),
        (80.0, 250, 0.003, 0.0008),
        (200.0, 250, 0.002, 0.0005),
        (500.0, 250, 0.001, 0.0002),
    ]

    # first anneal from incumbent
    try:
        X = anneal_opt(bestX.copy(), sched)
        vv = max_cos(X)
        if vv < bestV and np.all(np.isfinite(X)):
            bestX, bestV = X, vv
    except Exception:
        pass

    sigmas = [0.008, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0005]
    trial = 0
    iu0 = np.triu_indices(453, 1)
    iu_r_all = iu0[0]
    iu_c_all = iu0[1]
    while (time.time() - t0 < 100.0) and (trial < 12):
        sigma = sigmas[trial % len(sigmas)]
        try:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            c = G[iu_r_all, iu_c_all]
            K = 60
            part = np.argpartition(-c, K)[:K]
            I = iu_r_all[part]
            J = iu_c_all[part]
            worst = np.unique(np.concatenate([I, J]))
            Xtry = bestX.copy()
            Xtry = Xtry + rng.normal(scale=sigma * 0.3, size=Xtry.shape)
            Xtry[worst] = Xtry[worst] + rng.normal(scale=sigma, size=(worst.size, 5))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xtry = anneal_opt(Xtry, sched)
            vv = max_cos(Xtry)
            if vv < bestV and np.all(np.isfinite(Xtry)):
                bestX, bestV = Xtry, vv
        except Exception:
            pass
        trial += 1
        if time.time() - t0 > 105.0:
            break

    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 12.0:
        try:
            Xp = lp_polish(bestX.copy(), seconds=min(remain - 2.0, 65.0))
            vp = max_cos(Xp)
            if vp < bestV and np.all(np.isfinite(Xp)):
                bestX, bestV = Xp, vp
        except Exception:
            pass

    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

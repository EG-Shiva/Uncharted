import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t0=None, budget=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() - t0 > budget:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        if cnt > 2500:
            thr = np.sort(c)[-2500]
            sel = c >= thr
        I = iu[0][sel]
        J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
    return x


def _smooth_anneal(X, steps, k0, k1, lr0):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lk0, lk1 = np.log(k0), np.log(k1)
    b1t, b2t = 1.0, 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        k = float(np.exp(lk0 + (lk1 - lk0) * frac))
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s == 0.0:
            continue
        Grad = (W @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1t *= b1
        b2t *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1t)
        vhat = v1 / (1.0 - b2t)
        step = (lr * mhat / (np.sqrt(vhat) + eps))
        # clip extreme steps
        step = np.clip(step, -0.05, 0.05)
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _greedy_reinsert(X, rng, n_try=3000, n_points=6):
    n, d = X.shape
    X = X.copy()
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    cur = float(rowmax.max())
    for _round in range(2):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        improved = False
        for idx in order[:n_points]:
            # rest-rest max
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Gr = G[np.ix_(mask, mask)].max()
            C = rng.standard_normal((n_try, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = X[mask] @ C.T
            bv = S.max(axis=0)
            j = int(np.argmin(bv))
            new_global = bv[j] if bv[j] > Gr else Gr
            if new_global < cur - 1e-12:
                X[idx] = C[j]
                cur = float(new_global)
                # update G row/col
                g = X @ X[idx]
                G[idx, :] = g
                G[:, idx] = g
                G[idx, idx] = -2.0
                rowmax = G.max(axis=1)
                improved = True
        if not improved:
            break
    return X


def solve():
    t0 = time.time()
    BUDGET = 168.0
    np.random.seed(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal((121, 8))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands = [xb]
    Xbest = None
    best = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best:
            best = v
            Xbest = c.copy()
    sigmas = [0.015, 0.008, 0.025, 0.005, 0.04, 0.012]
    lrs = [0.008, 0.005, 0.012, 0.006, 0.010, 0.007]
    trial = 0
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 12.0:
            break
        sigma = sigmas[trial % len(sigmas)]
        lr0 = lrs[trial % len(lrs)]
        rng = np.random.default_rng(1000 + trial)
        # perturb, kicking worst points harder
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        noise = rng.standard_normal(Xbest.shape) * sigma
        # extra kick to top-10 worst
        extra = np.zeros_like(Xbest)
        extra[order[:10]] = rng.standard_normal((10, Xbest.shape[1])) * sigma * 2.0
        X = Xbest + noise + extra
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # annealed smooth optimisation
        k1 = 5000.0 if (trial % 2 == 0) else 2500.0
        X = _smooth_anneal(X, steps=2200, k0=40.0, k1=k1, lr0=lr0)
        v = _max_cos(X)
        # cheap greedy re-insertion
        if v < best + 0.01:
            X = _greedy_reinsert(X, rng, n_try=3000, n_points=6)
            v = _max_cos(X)
        # decide to LP-polish
        promising = (v < best + 0.004) or (trial % 3 == 0 and v < best + 0.01)
        if promising:
            rem2 = BUDGET - (time.time() - t0)
            secs = min(6.0, rem2 - 4.0)
            if secs > 1.0:
                X = _lp_polish(X, seconds=secs, t0=t0, budget=BUDGET - 2.0)
                v = _max_cos(X)
        if v < best - 1e-12:
            best = v
            Xbest = X.copy()
        trial += 1
        if trial > 60:
            break
    # final squeeze of global best
    rem = BUDGET - (time.time() - t0)
    if rem > 3.0:
        secs = min(rem - 1.0, 10.0)
        if secs > 1.0:
            Xbest = _lp_polish(Xbest, seconds=secs, t0=t0, budget=BUDGET - 0.5)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return m, cnt

def _smooth(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters + 1):
        f = it / max(iters, 1)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        s = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - s * X
        if not np.all(np.finite(Gt)):
            break
        b1p *= b1
        b2p *= b2
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m1 / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X = _norm(X)
    return X

def _lp_polish(X, seconds):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.1:
            break
        G = X @ X.T
        c = G[iu]
        mm = float(c.max())
        sel = c > mm - max(20 * radius, 1e-9)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            break
        if len(idx) > 1500:
            o = np.argsort(c[idx])[-1500:]
            idx = idx[o]
        I, J = iu[0][idx], iu[1][idx]
        csel = c[idx]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX

def _refine_hole(others, c0, steps=250, lr0=0.08):
    c = c0 / (np.linalg.norm(c0) + 1e-30)
    lr = lr0
    for _ in range(steps):
        dots = others @ c
        m = float(dots.max())
        act = others[dots > m - 2e-3]
        if len(act) == 0:
            break
        g = act.sum(axis=0)
        g = g - (g @ c) * c
        n = np.linalg.norm(g)
        if n < 1e-12:
            break
        c = c - lr * g / (n + 1e-12) * 0.1 - lr * 0.1 * g
        c = c / (np.linalg.norm(c) + 1e-30)
        lr *= 0.997
    return c / (np.linalg.norm(c) + 1e-30)

def _hole_reinsert(Y, idx_list, rng):
    Y = Y.copy()
    for r in list(idx_list):
        others = np.delete(Y, r, axis=0)
        cand = rng.standard_normal((8000, 6))
        cand = _norm(cand)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        k = int(np.argmin(worst))
        c0 = cand[k]
        c0 = _refine_hole(others, c0)
        # check refined vs best of few
        Y[r] = c0
    return Y

def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261031)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = rng.standard_normal((116, 6))
    Xb = _norm(np.asarray(Xb, dtype=float))
    if Xb.shape != (116, 6) or not np.all(np.finite(Xb)):
        r2 = np.random.default_rng(1)
        Xb = _norm(r2.standard_normal((116, 6)))
    bv = _max_cos(Xb)
    try:
        Xr = np.load("record.npy").astype(float)
        Xr = _norm(Xr)
        vr = _max_cos(Xr)
        if vr < bv:
            Xb = Xr.copy(); bv = vr
    except Exception:
        pass
    Xbest = Xb.copy(); bestV = bv
    curX = Xb.copy(); curV = bv
    n, d = 116, 6
    explore_end = t0 + 150.0
    it = 0
    T0 = 3e-4
    while time.time() < explore_end:
        it += 1
        el = (time.time() - t0) / 150.0
        Temp = T0 * (1.0 - el) + 2e-6
        mode = it % 6
        Y = curX.copy()
        if mode == 0:
            sig = 0.01 + 0.025 * rng.random()
            Y = Y + sig * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 600, 60, 450, 0.020, 0.003)
        elif mode == 1:
            _, cnt = _contacts(curX, tol=1e-7)
            order = np.argsort(-cnt)
            k = 6 + int(rng.integers(0, 5))
            idx = order[:k]
            Y[idx] = Y[idx] + 0.16 * rng.standard_normal((k, d))
            Y = Y + 0.006 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 800, 80, 700, 0.020, 0.002)
        elif mode == 2:
            _, cnt = _contacts(curX, tol=1e-7)
            ratt = np.where(cnt <= 2)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:8]
            if len(ratt) > 14:
                ratt = ratt[rng.choice(len(ratt), 14, replace=False)]
            Y = _hole_reinsert(Y, ratt, rng)
            Y = Y + 0.003 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 650, 90, 500, 0.015, 0.002)
        elif mode == 3:
            G = curX @ curX.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            pairs = np.argwhere(G > m - 1e-7)
            # pick up to 3 distinct indices from worst pairs
            uniq = np.unique(pairs[:40].ravel())
            if len(uniq) < 2:
                uniq = np.argsort(-_contacts(curX, tol=1e-7)[1])[:3]
            sel = uniq[rng.choice(len(uniq), min(3, len(uniq)), replace=False)]
            Y = _hole_reinsert(Y, sel, rng)
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 700, 100, 800, 0.018, 0.002)
        elif mode == 4:
            k = 12
            idx = rng.choice(n, k, replace=False)
            Y[idx] = rng.standard_normal((k, d))
            Y = _norm(Y)
            Y = _smooth(Y, 900, 50, 500, 0.025, 0.003)
        else:
            sig = 0.025 + 0.02 * rng.random()
            Y = Xbest + sig * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 700, 70, 600, 0.020, 0.0025)
        v = _max_cos(Y)
        # short LP if promising
        rem = explore_end - time.time()
        if v < curV + 4e-4 and v < bestV + 8e-4 and rem > 3.0:
            Y2 = _lp_polish(Y, min(1.6, max(0.6, rem * 0.08)))
            v2 = _max_cos(Y2)
            if v2 < v:
                Y, v = Y2, v2
        # acceptance: walk on cur, elitist on best
        if v < curV or rng.random() < np.exp((curV - v) / max(Temp, 1e-9)):
            curX, curV = Y.copy(), v
        if v < bestV - 1e-12:
            Xbest, bestV = Y.copy(), v
        # occasional restart walk from best if stuck
        if it % 25 == 0 and time.time() < explore_end - 5:
            if curV > bestV + 5e-4:
                curX, curV = Xbest.copy(), bestV
    # final long polish of best
    if time.time() < t0 + 177:
        Xbest = _lp_polish(Xbest, max(1.0, (t0 + 177.5) - time.time()))
    Xbest = _norm(Xbest)
    return Xbest

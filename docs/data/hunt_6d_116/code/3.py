import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_counts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return m, cnt

def _smooth(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for it in range(1, iters + 1):
        frac = it / iters
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = G.max()
        # W = exp(alpha*(G-cmax)), <=1 so no overflow
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        s = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - s * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for linprog itself
        if time.time() - start > seconds - 0.05:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        rng0 = np.random.default_rng(1)
        Xbest = rng0.standard_normal((116, 6))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    cur = _max_cos(Xbest)
    # also consider record.npy as alternate basin
    try:
        Xr = np.load("record.npy").astype(float)
        Xr = Xr / np.linalg.norm(Xr, axis=1, keepdims=True)
        vr = _max_cos(Xr)
        if vr < cur:
            Xbest = Xr
            cur = vr
    except Exception:
        pass
    n, d = Xbest.shape
    it = 0
    # search phase until ~115s
    while time.time() - t0 < 115:
        it += 1
        mode = it % 5
        _, cnt = _contact_counts(Xbest, tol=1e-7)
        Y = Xbest.copy()
        if mode == 0:
            sigma = 0.015 + 0.015 * rng.random()
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 1:
            order = np.argsort(-cnt)
            k = 8
            idx = order[:k]
            Y[idx] = Y[idx] + 0.12 * rng.standard_normal((k, d))
            Y = Y + 0.008 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 2:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:6]
            Y[ratt] = rng.standard_normal((len(ratt), d))
            Y = Y + 0.003 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 3:
            ratt = np.where(cnt <= 2)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:5]
            perm = rng.permutation(ratt)
            for r in perm:
                others = np.delete(Y, r, axis=0)
                cand = rng.standard_normal((3000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ others.T
                mx = dots.max(axis=1)
                Y[r] = cand[np.argmin(mx)]
            Y = Y + 0.005 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        else:
            sigma = 0.06 + 0.04 * rng.random()
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # smooth polish with annealed temperature
        if mode in (2, 3):
            Y = _smooth(Y, iters=1200, a0=60.0, a1=600.0, lr0=0.015, lr1=0.002)
        else:
            Y = _smooth(Y, iters=900, a0=50.0, a1=500.0, lr0=0.012, lr1=0.002)
        v = _max_cos(Y)
        # only LP-refine promising candidates
        if v < cur + 0.004:
            # short LP budget
            Y2 = _lp_polish(Y, seconds=1.0)
            v2 = _max_cos(Y2)
            if v2 < cur - 1e-12:
                Xbest = Y2
                cur = v2
        if time.time() - t0 > 114:
            break
    # fine smooth at high temperature then final LP
    try:
        remain = 170 - (time.time() - t0)
        if remain > 8:
            # brief high-alpha smoothing
            Xbest = _smooth(Xbest, iters=600, a0=500.0, a1=1200.0, lr0=0.004, lr1=0.0005)
            remain = 170 - (time.time() - t0)
        remain = max(5.0, min(40.0, remain))
        Xbest = _lp_polish(Xbest, seconds=float(remain))
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

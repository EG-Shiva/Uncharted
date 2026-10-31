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


def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def _best_hole(others, rng, n_cand=8000):
    d = others.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    mx = dots.max(axis=1)
    return cand[np.argmin(mx)]


def _hinge_adam(X, T, iters, lr0, lr1):
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
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        D = G - T
        dmax = float(D.max())
        if dmax <= 0:
            alpha = 250.0 + 550.0 * frac
            cmax = float(G.max())
            W = np.exp(alpha * (G - cmax))
            np.fill_diagonal(W, 0.0)
        else:
            W = np.where(D > 0, 2.0 * D, 0.0)
            # tiny background repulsion to keep pack spread
            # add small softmax weight for near-max pairs
            cmax = float(G.max())
            Wb = np.exp(120.0 * (G - cmax))
            np.fill_diagonal(Wb, 0.0)
            W = W + 0.02 * Wb
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
        Xbest = rng.standard_normal((116, 6))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    cur = _max_cos(Xbest)
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
    # search phase
    while time.time() - t0 < 135:
        it += 1
        if time.time() - t0 > 134:
            break
        _, cnt = _contact_counts(Xbest, tol=1e-7)
        ppm = _per_point_max(Xbest)
        Y = Xbest.copy()
        mode = int(rng.integers(0, 6))
        if mode == 0:
            sigma = 0.012 + 0.020 * float(rng.random())
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            hinge_iters = 800
        elif mode == 1:
            order = np.argsort(-cnt)
            k = int(rng.integers(6, 13))
            idx = order[:k]
            Y[idx] = Y[idx] + 0.14 * rng.standard_normal((k, d))
            Y = Y + 0.006 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            hinge_iters = 900
        elif mode == 2:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:5]
            if len(ratt) > 8:
                ratt = ratt[rng.choice(len(ratt), 8, replace=False)]
            perm = rng.permutation(ratt)
            for r in perm:
                others = np.delete(Y, r, axis=0)
                Y[r] = _best_hole(others, rng, n_cand=8000)
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            hinge_iters = 900
        elif mode == 3:
            k = int(rng.integers(3, 7))
            order = np.argsort(-ppm)
            idx = order[:k]
            perm = rng.permutation(idx)
            for r in perm:
                others = np.delete(Y, r, axis=0)
                Y[r] = _best_hole(others, rng, n_cand=10000)
            Y = Y + 0.005 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            hinge_iters = 900
        elif mode == 4:
            k = int(rng.integers(10, 21))
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = rng.standard_normal((k, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            # greedy hole pass over those indices
            perm = rng.permutation(idx)
            for r in perm:
                others = np.delete(Y, r, axis=0)
                Y[r] = _best_hole(others, rng, n_cand=5000)
            Y = Y + 0.003 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            hinge_iters = 1000
        else:
            sigma = 0.05 + 0.05 * float(rng.random())
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            hinge_iters = 800
        # target compression level: force below cur
        lexp = float(rng.uniform(-4.0, -2.3))
        delta = 10.0 ** lexp
        T = cur - delta
        # hinge compression then soft-max smoothing
        Y = _hinge_adam(Y, T, iters=hinge_iters, lr0=0.012, lr1=0.002)
        Y = _smooth(Y, iters=700, a0=70.0, a1=650.0, lr0=0.010, lr1=0.0018)
        v = _max_cos(Y)
        if v < cur - 1e-12:
            # promising enough to spend LP time
            if v < cur + 0.002:
                rem = 135 - (time.time() - t0)
                if rem > 1.0:
                    Y2 = _lp_polish(Y, seconds=0.5)
                    v2 = _max_cos(Y2)
                    if v2 < cur - 1e-12:
                        Xbest = Y2
                        cur = v2
                        continue
            # if LP did not improve but raw did, keep raw
            if v < cur - 1e-12:
                # short extra smoothing at high temp before accept
                Ys = _smooth(Y, iters=200, a0=500.0, a1=1000.0, lr0=0.003, lr1=0.0005)
                vs = _max_cos(Ys)
                if vs < v:
                    Y, v = Ys, vs
                Xbest = Y
                cur = v
        elif v < cur + 0.002:
            # near-miss: only LP if very close to save time
            if v < cur + 0.0005:
                rem = 135 - (time.time() - t0)
                if rem > 1.5:
                    Y2 = _lp_polish(Y, seconds=0.4)
                    v2 = _max_cos(Y2)
                    if v2 < cur - 1e-12:
                        Xbest = Y2
                        cur = v2
    # final polish
    try:
        remain = 170 - (time.time() - t0)
        if remain > 10:
            Xbest = _smooth(Xbest, iters=700, a0=400.0, a1=1300.0, lr0=0.004, lr1=0.0004)
            remain = 170 - (time.time() - t0)
        if remain > 6:
            budget = float(max(5.0, min(32.0, remain - 1.0)))
            Xbest = _lp_polish(Xbest, seconds=budget)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

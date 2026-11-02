import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_polish(v0, iters, beta0, beta1, lr0, mu):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 * (1.0 - 0.5 * frac)
        D = v @ v.T
        np.fill_diagonal(D, -2.0)
        m = D.max()
        # weights, stable (D-m <= 0)
        W = np.exp(beta * (D - m))
        np.fill_diagonal(W, 0.0)
        S = W @ v
        wdot = (W * D).sum(axis=1, keepdims=True)
        grad = S - wdot * v
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad /= rs
        vel = mu * vel - lr * grad
        v = v + vel
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    HARD = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(189, 12))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # also consider record
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = 189, 12
    if xb.shape != (n, d):
        xb = rng.normal(size=(n, d))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    trial = 0
    # exploration budget: leave ~45s for final LP
    while time.time() - t0 < 122.0 and time.time() - t0 < HARD - 50.0:
        trial += 1
        mode = trial % 4
        if mode == 0:
            # worst-point void reinsertion
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # pick among top-5 worst randomly for diversity
            order = np.argpartition(-rowmax, 5)[:5]
            iw = order[rng.integers(0, 5)]
            nc = 1200
            cands = rng.normal(size=(nc, d))
            cands /= np.linalg.norm(cands, axis=1, keepdims=True)
            dots = cands @ x_best.T
            dots[:, iw] = -1e9
            cmax = dots.max(axis=1)
            ib = int(np.argmin(cmax))
            y = x_best.copy()
            y[iw] = cands[ib]
            y += 0.004 * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters = 350
            b0, b1, lr0, mu = 40.0, 260.0, 0.10, 0.85
        elif mode == 1:
            sigma = 0.018
            y = x_best + sigma * rng.normal(size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters = 300
            b0, b1, lr0, mu = 30.0, 220.0, 0.12, 0.88
        elif mode == 2:
            sigma = 0.045
            y = x_best + sigma * rng.normal(size=x_best.shape)
            # reset 2 random points fully
            idx = rng.choice(n, 2, replace=False)
            y[idx] = rng.normal(size=(2, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters = 400
            b0, b1, lr0, mu = 25.0, 250.0, 0.12, 0.88
        else:
            sigma = 0.010
            y = x_best + sigma * rng.normal(size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            iters = 250
            b0, b1, lr0, mu = 50.0, 300.0, 0.08, 0.80
        yp = _smooth_polish(y, iters, b0, b1, lr0, mu)
        v = _max_cos(yp)
        if v < best_val:
            # tiny extra refine with high-beta short run
            yp2 = _smooth_polish(yp, 120, 150.0, 350.0, 0.04, 0.8)
            v2 = _max_cos(yp2)
            if v2 < v:
                yp, v = yp2, v2
            x_best = yp
            best_val = v
        if time.time() - t0 > HARD - 50.0:
            break

    remain = HARD - (time.time() - t0) - 1.0
    if remain > 4.0:
        x_best = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    Iu0, Ju0 = iu
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    # quick initial LP snap is already done in parent; skip, go to hopping
    it = 0
    # phase 1: hopping until ~140s
    while time.time() - t0 < 138.0:
        it += 1
        if time.time() - t0 > HARD - 30:
            break
        # contacts of incumbent
        G = x_best @ x_best.T
        gall = G[iu]
        mx = gall.max()
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 6
        if mode == 0:
            sigma, nrep, extra = 0.015, 0, 0.0
        elif mode == 1:
            sigma, nrep, extra = 0.03, 1, 0.25
        elif mode == 2:
            sigma, nrep, extra = 0.07, 2, 0.25
        elif mode == 3:
            sigma, nrep, extra = 0.02, 1, 0.4
        elif mode == 4:
            sigma, nrep, extra = 0.12, 2, 0.0
        else:
            sigma, nrep, extra = 0.05, 3, 0.15
        y = x_best.copy()
        y += rng.normal(0, sigma, size=y.shape)
        if len(ratt) > 0 and extra > 0:
            # big kick or full replacement for rattlers
            for idx in ratt:
                if rng.random() < 0.6:
                    y[idx] = rng.normal(size=4)
                else:
                    y[idx] += rng.normal(0, extra, size=4)
        if len(hubs) > 0:
            y[hubs] += rng.normal(0, sigma * 0.8, size=(len(hubs), d))
        if nrep > 0:
            rep = rng.choice(n, size=nrep, replace=False)
            y[rep] = rng.normal(size=(nrep, 4))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed smooth descent
        steps = 1100
        # vary beta schedule slightly by mode for diversity
        if mode in (2, 4):
            beta0, beta1 = 25.0, 350.0
        else:
            beta0, beta1 = 40.0, 280.0
        y2 = _smooth_opt(y, steps, beta0, beta1, 0.006, 0.0004, iu)
        v2 = _max_cos(y2)
        # only LP-polish promising candidates
        thresh = best_val + (0.004 if mode in (2, 4, 5) else 0.0015)
        if v2 < thresh:
            remain = HARD - (time.time() - t0) - 22.0
            if remain < 1.0:
                break
            b = min(3.0, max(1.2, remain / 8.0))
            yp, vp = _lp_polish(y2, b)
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp
            elif vp < best_val + 1e-12 and rng.random() < 0.02:
                # sideways move to escape plateau rarely
                x_best = yp
        # safety time check
        if time.time() - t0 > HARD - 25:
            break
    # final polish of incumbent with remaining time
    remain = HARD - (time.time() - t0) - 2.0
    if remain > 2.0:
        remain = min(remain, 28.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

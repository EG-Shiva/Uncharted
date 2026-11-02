import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for single linprog overrun
        Gfull = x @ x.T
        c_full = Gfull[iu]
        m = c_full.max()
        sel = c_full > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_full[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def smooth_opt(x, steps=1200, a0=30.0, a1=280.0, eta0=0.04):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        eta = eta0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # exp in stable way: max is 0
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W @ x
        wg = (W * G).sum(axis=1, keepdims=True)
        Tang = S - wg * x
        wsum = W.sum(axis=1, keepdims=True) + 1e-12
        step = Tang / wsum
        x = x - eta * step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
    return x

def solve():
    t0 = time.time()
    BUDGET = 170.0
    # load starting points
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (189, 12):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append(c)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        c = rng0.normal(size=(189, 12))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    # pick best of loads
    best_x = cands[0]
    best_v = max_cos(best_x)
    for c in cands[1:]:
        v = max_cos(c)
        if v < best_v:
            best_x, best_v = c, v
    best_x = best_x.copy()
    rng = np.random.default_rng(0)
    sigmas = [0.03, 0.06, 0.015, 0.10, 0.04, 0.02, 0.05]
    etas = [0.04, 0.06, 0.025]
    it = 0
    # reserve ~28s for final polish
    while time.time() - t0 < BUDGET - 28:
        sigma = sigmas[it % len(sigmas)]
        eta0 = etas[(it // len(sigmas)) % len(etas)]
        # analyse current best structure
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        m_best = G.max()
        per_max = G.max(axis=1)
        deg = (G > m_best - 0.01).sum(axis=1)
        rattlers = np.where(per_max < m_best - 1e-3)[0]
        overload = np.argsort(deg)[-8:]
        y = best_x.copy()
        if it % 4 == 3:
            # largest-hole re-insertion of 2 worst points
            worst = np.argsort(per_max)[-2:]
            for wi in worst:
                mask = np.arange(y.shape[0]) != wi
                Xr = y[mask]
                K = 20000
                cand = rng.normal(size=(K, y.shape[1]))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = Xr @ cand.T
                cmax = dots.max(axis=0)
                bi = int(np.argmin(cmax))
                y[wi] = cand[bi]
            y = y + rng.normal(scale=0.01, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_opt(y, steps=1100, a0=40.0, a1=300.0, eta0=eta0)
            # short LP
            remain = BUDGET - 28 - (time.time() - t0)
            if remain < 2:
                break
            y = lp_polish(y, seconds=min(3.0, remain))
        else:
            noise = rng.normal(scale=sigma, size=y.shape)
            if len(rattlers) > 0:
                noise[rattlers] *= 2.5
            if len(overload) > 0:
                noise[overload] *= 2.0
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # anneal params vary slightly per iter
            a1 = 250.0 + (it % 3) * 50.0
            y = smooth_opt(y, steps=1300, a0=30.0, a1=a1, eta0=eta0)
            remain = BUDGET - 28 - (time.time() - t0)
            if remain < 2:
                break
            y = lp_polish(y, seconds=min(3.0, remain))
        v = max_cos(y)
        if v < best_v - 1e-12 and np.all(np.isfinite(y)):
            best_x, best_v = y, v
        it += 1
        if it > 200:
            break
    # final long polish of best
    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        best_x = lp_polish(best_x, seconds=min(remain - 1.0, 28.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

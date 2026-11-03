import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    if t_end is None:
        t_end = start + seconds
    else:
        t_end = min(t_end, start + seconds)
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, steps=1800, k0=25.0, k1=500.0, lr=0.004):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        kk = k0 * ((k1 / k0) ** (t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights, avoid overflow
        W = np.exp(kk * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        # keep tangent
        sn = np.sum(step * x, axis=1, keepdims=True)
        step = step - sn * x
        # clip large steps
        snorm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.03 / snorm)
        step = step * clip
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def best_void_replace(X, idx, rng, n_cand=8000, refine=350):
    n, d = X.shape
    others = np.delete(X, idx, axis=0)
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = cand @ others.T
    mxs = scores.max(axis=1)
    c = cand[int(np.argmin(mxs))].copy()
    # subgradient refine: minimize max dot
    for it in range(refine):
        dots = others @ c
        j = int(np.argmax(dots))
        g = others[j] - float(others[j] @ c) * c
        st = 0.02 / (1.0 + it / 60.0)
        c = c - st * g
        nc = np.linalg.norm(c)
        if nc < 1e-12:
            c = rng.standard_normal(size=d); nc = np.linalg.norm(c)
            c = c / nc
        else:
            c = c / nc
    Xn = X.copy()
    Xn[idx] = c
    return Xn

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.standard_normal(size=(95, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
        if vr < best:
            best, xb = vr, xr
    except Exception:
        pass
    n, d = xb.shape
    trial = 0
    # reserve ~28s for final polish
    while time.time() < deadline - 28.0:
        trial += 1
        mode = trial % 7
        x_try = xb.copy()
        # contact stats of current best for guided moves
        try:
            G0 = xb @ xb.T
            np.fill_diagonal(G0, -2.0)
            rowmax = G0.max(axis=1)
            contacts = (G0 > best - 1e-7).sum(axis=1)
        except Exception:
            rowmax = None
            contacts = None
        if mode == 0:
            noise = rng.standard_normal(size=x_try.shape)
            noise -= np.sum(noise * x_try, axis=1, keepdims=True) * x_try
            x_try = x_try + 0.015 * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 1200
        elif mode == 1:
            noise = rng.standard_normal(size=x_try.shape)
            noise -= np.sum(noise * x_try, axis=1, keepdims=True) * x_try
            x_try = x_try + 0.05 * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 1800
        elif mode == 2:
            center = int(rng.integers(n))
            dots = xb[center] @ xb.T
            order = np.argsort(-dots)[:13]
            noise = rng.standard_normal(size=x_try.shape)
            noise -= np.sum(noise * x_try, axis=1, keepdims=True) * x_try
            sig = np.full(n, 0.008)
            sig[order] = 0.09
            x_try = x_try + sig[:, None] * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 1800
        elif mode == 3:
            worst = int(np.argmax(rowmax))
            x_try = best_void_replace(x_try, worst, rng, n_cand=8000, refine=350)
            # small global jitter
            noise = rng.standard_normal(size=x_try.shape)
            noise -= np.sum(noise * x_try, axis=1, keepdims=True) * x_try
            x_try = x_try + 0.008 * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 1800
        elif mode == 4:
            order = np.argsort(-rowmax)[:3]
            for idx in order:
                x_try = best_void_replace(x_try, int(idx), rng, n_cand=5000, refine=250)
            noise = rng.standard_normal(size=x_try.shape)
            noise -= np.sum(noise * x_try, axis=1, keepdims=True) * x_try
            x_try = x_try + 0.01 * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 2000
        elif mode == 5:
            order = np.argsort(contacts)[:6]
            noise = rng.standard_normal(size=x_try.shape)
            noise -= np.sum(noise * x_try, axis=1, keepdims=True) * x_try
            sig = np.full(n, 0.01)
            sig[order] = 0.22
            x_try = x_try + sig[:, None] * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 1800
        else:
            noise = rng.standard_normal(size=x_try.shape)
            noise -= np.sum(noise * x_try, axis=1, keepdims=True) * x_try
            x_try = x_try + 0.12 * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 2200
        # smooth annealed repulsion
        x_try = smooth_relax(x_try, steps=steps, k0=25.0, k1=500.0, lr=0.004)
        remain = (deadline - 28.0) - time.time()
        if remain < 0.8:
            # evaluate without LP if no time
            v = max_cos(x_try)
            if v < best - 1e-12:
                best, xb = v, x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
            break
        x_try = lp_polish(x_try, seconds=min(3.0, remain))
        v = max_cos(x_try)
        if v < best - 1e-12:
            best = v
            xb = x_try
        if time.time() >= deadline - 28.0:
            break
    remain = deadline - time.time() - 0.8
    if remain > 2.0:
        xb = lp_polish(xb, seconds=remain)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb

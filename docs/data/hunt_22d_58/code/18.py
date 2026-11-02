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
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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

def adam_quench(x0, steps, T0, T1, lr0, lr1):
    x = x0.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    # precompute log schedules
    logT = np.log(T1 / T0) / max(steps - 1, 1)
    logL = np.log(lr1 / lr0) / max(steps - 1, 1)
    for s in range(steps):
        T = T0 * np.exp(logT * s)
        lr = lr0 * np.exp(logL * s)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gm = G.max()
        # softmax weights
        E = np.exp((G - gm) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tang = Grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * Tang
        vv = b2 * vv + (1.0 - b2) * (Tang * Tang)
        mhat = m1 / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        b1p *= b1
        b2p *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # keep tangential
        rad = (step * x).sum(axis=1, keepdims=True)
        step = step - rad * x
        # clip max move per point
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-15
        cap = 0.05
        sc = np.minimum(1.0, cap / nm)
        step = step * sc
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    LIMIT = 165.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            x_best = np.load("record.npy").astype(float)
        except Exception:
            x_best = rng.standard_normal((58, 22))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best = max_cos(x_best)
    # quick record fallback
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
        if vr < best:
            best = vr
            x_best = xr.copy()
    except Exception:
        pass
    x0base = x_best.copy()
    trial = 0
    # budget: leave ~45s for final LP
    while time.time() - t0 < 110.0 and trial < 35:
        trial += 1
        if trial == 1:
            x_init = x0base.copy()
            T0, T1, lr0, lr1, steps = 0.008, 0.0007, 0.003, 0.0003, 1800
        else:
            # diversify sigmas and params
            if trial % 5 == 0:
                sigma = 0.09
            elif trial % 3 == 0:
                sigma = 0.04
            else:
                sigma = 0.015 + 0.005 * (trial % 4)
            x_init = x0base + sigma * rng.standard_normal(x0base.shape)
            if trial % 4 == 0:
                # reinsert worst points
                G = x0base @ x0base.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                worst = np.argsort(rowmax)[-3:]
                x_init[worst] = rng.standard_normal((len(worst), x_init.shape[1]))
            elif trial % 4 == 1:
                G = x0base @ x0base.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                worst = np.argsort(rowmax)[-1:]
                x_init[worst] = rng.standard_normal((len(worst), x_init.shape[1]))
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            # cycle quench params
            cyc = trial % 6
            if cyc == 0:
                T0, T1, lr0, lr1, steps = 0.012, 0.0006, 0.004, 0.00025, 1800
            elif cyc == 1:
                T0, T1, lr0, lr1, steps = 0.008, 0.0008, 0.003, 0.0003, 1500
            elif cyc == 2:
                T0, T1, lr0, lr1, steps = 0.015, 0.0005, 0.005, 0.0002, 2000
            elif cyc == 3:
                T0, T1, lr0, lr1, steps = 0.010, 0.0006, 0.0035, 0.00025, 1800
            elif cyc == 4:
                T0, T1, lr0, lr1, steps = 0.006, 0.0004, 0.0025, 0.0002, 1500
            else:
                T0, T1, lr0, lr1, steps = 0.011, 0.0009, 0.004, 0.00035, 1600
        # time guard: shrink steps if little time
        if time.time() - t0 > 90:
            steps = min(steps, 1200)
        try:
            xq = adam_quench(x_init, steps, T0, T1, lr0, lr1)
        except Exception:
            continue
        v = max_cos(xq)
        if v < best:
            best = v
            x_best = xq.copy()
            x0base = xq.copy()
        if time.time() - t0 > LIMIT - 50:
            break
    remain = LIMIT - (time.time() - t0) - 3.0
    remain = float(np.clip(remain, 15.0, 50.0))
    try:
        x_best = lp_polish(x_best, seconds=remain)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

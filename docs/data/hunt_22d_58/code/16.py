import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def quench_softmax(x, steps, step0, step1, t0, t1, hard_tail=200, hard_width=0.005):
    n, d = x.shape
    # annealed softmax phase
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        temp = t0 * (t1 / t0) ** frac if t0 != t1 else t0
        step = step0 * (step1 / step0) ** frac if step0 != step1 else step0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights, stable
        W = np.exp((G - gmax) / temp)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.einsum('ij,ij->i', grad, x)[:, None]
        grad -= dot * x
        nr = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        x = x - step * (grad / nr)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    # hard-max tail: only near-max pairs repel
    if hard_tail > 0:
        for s in range(hard_tail):
            frac = s / max(hard_tail - 1, 1)
            step = step1 * (0.5 ** frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            M = (G > m - hard_width).astype(float)
            np.fill_diagonal(M, 0.0)
            # if too few, widen
            if M.sum() < 10:
                M = (G > m - 5 * hard_width).astype(float)
                np.fill_diagonal(M, 0.0)
            grad = M @ x
            dot = np.einsum('ij,ij->i', grad, x)[:, None]
            grad -= dot * x
            nr = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
            x = x - step * (grad / nr)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and time.time() < t_end - 0.1 and radius > 1e-14:
        # allow early exit if seconds exceeded - use t_end
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= t_end:
            break
    return x


def solve():
    t0 = time.time()
    T_HOP = t0 + 115.0
    T_LP = t0 + 170.0
    rng = np.random.default_rng(12345)
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        x_best = rng.normal(size=(58, 22))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    else:
        # pick best of loaded
        bestv = 1e9
        x_best = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bestv:
                bestv = v
                x_best = c.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best = max_cos(x_best)

    n, d = 58, 22
    trial = 0
    # quick initial quench of record (tiny shake) to squeeze noise
    # hopping loop
    while time.time() < T_HOP:
        trial += 1
        mode = trial % 5
        x = x_best.copy()
        if mode == 0:
            sigma = 0.008 + 0.004 * rng.random()
            x = x + sigma * rng.normal(size=x.shape)
        elif mode == 1:
            sigma = 0.02 + 0.02 * rng.random()
            x = x + sigma * rng.normal(size=x.shape)
        elif mode == 2:
            k = int(rng.integers(3, 9))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.08 + 0.08 * rng.random()
            x[idx] += sigma * rng.normal(size=(k, d))
        elif mode == 3:
            k = int(rng.integers(1, 3))
            idx = rng.choice(n, size=k, replace=False)
            for ii in idx:
                v = rng.normal(size=d)
                v /= np.linalg.norm(v)
                # mix random with old for partial reinsert
                a = rng.random() * 0.7 + 0.3
                x[ii] = (1 - a) * x[ii] + a * v * 3.0
        else:
            # single full reinsert of worst point occasionally
            G = x @ x.T
            np.fill_diagonal(G, -2)
            # point with largest row-max
            rm = G.max(axis=1)
            wi = int(np.argmax(rm))
            v = rng.normal(size=d)
            v /= np.linalg.norm(v)
            x[wi] = v
            # also shake neighbours slightly
            x += 0.005 * rng.normal(size=x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        # choose quench params depending on mode
        if mode in (0, 1):
            steps = 450
            s0, s1 = 0.015, 0.0015
            tt0, tt1 = 0.015, 0.004
            hw = 0.008
        elif mode == 2:
            steps = 600
            s0, s1 = 0.02, 0.001
            tt0, tt1 = 0.02, 0.003
            hw = 0.006
        else:
            steps = 700
            s0, s1 = 0.025, 0.001
            tt0, tt1 = 0.02, 0.003
            hw = 0.006
        # slight randomization
        s0 *= (0.8 + 0.4 * rng.random())
        tt0 *= (0.8 + 0.4 * rng.random())

        x = quench_softmax(x, steps, s0, s1, tt0, tt1, hard_tail=150, hard_width=hw)
        v = max_cos(x)
        if v < best:
            best = v
            x_best = x
            # intensify around new best: a few tiny shakes immediately
            # continue (keep time check)
        # early time check inside quench not possible, check outer
        if time.time() >= T_HOP:
            break
        # safety: if trial huge, break
        if trial > 5000:
            break

    # final LP polish on best
    try:
        x_best = lp_polish(x_best, 60, T_LP)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g.max()

def adam_refine(y, schedule=((60.0, 350, 0.015), (150.0, 350, 0.010), (320.0, 350, 0.008))):
    n, d = y.shape
    m = np.zeros_like(y)
    vv = np.zeros_like(y)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    eye = np.eye(n)
    for alpha, iters, lr in schedule:
        for _ in range(iters):
            G = y @ y.T
            # off-diagonal max for shift (diag is 1)
            gmax = float((G - 2.0 * eye).max())
            # shifted weights, stable and scale invariant for Adam
            W = np.exp(alpha * (G - gmax))
            np.fill_diagonal(W, 0.0)
            Grad = alpha * (W @ y)
            s = np.sum(Grad * y, axis=1, keepdims=True)
            Gt = Grad - s * y
            b1p *= beta1
            b2p *= beta2
            m *= beta1
            m += (1.0 - beta1) * Gt
            vv *= beta2
            vv += (1.0 - beta2) * (Gt * Gt)
            mhat = m / (1.0 - b1p)
            vhat = vv / (1.0 - b2p)
            step = mhat / (np.sqrt(vhat) + eps)
            np.clip(step, -3.0, 3.0, out=step)
            y -= lr * step
            y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def lp_polish(x, seconds):
    x = _norm(x)
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

def solve():
    t0 = time.time()
    TIME_BUDGET = 175.0
    EXPLORE_BUDGET = 138.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (82, 14):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((82, 14))
        cands.append(_norm(a))
    best = None
    bestv = np.inf
    for c in cands:
        v = _max_cos(c)
        if v < bestv:
            bestv = v
            best = c.copy()
    cur = best.copy()
    curv = bestv
    Temp = 0.002
    it = 0
    n, d = best.shape
    while time.time() - t0 < EXPLORE_BUDGET:
        it += 1
        # time guard
        if TIME_BUDGET - (time.time() - t0) < 20:
            break
        u = rng.random()
        if u < 0.40:
            sigma = 10.0 ** rng.uniform(-2.3, -0.8)
            cand = cur + sigma * rng.standard_normal((n, d))
        elif u < 0.70:
            k = int(rng.integers(1, 6))
            idx = rng.choice(n, k, replace=False)
            noise = 0.01 * rng.standard_normal((n, d))
            noise[idx] = (0.30 + 0.40 * rng.random()) * rng.standard_normal((k, d))
            cand = cur + noise
        else:
            k = int(rng.integers(1, 4))
            # find rattlers (smallest rowmax)
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(rowmax)
            if rng.random() < 0.75:
                pool = order[:15]
                idx = rng.choice(pool, k, replace=False)
            else:
                idx = rng.choice(n, k, replace=False)
            cand = cur.copy() + 0.005 * rng.standard_normal((n, d))
            rnd = rng.standard_normal((k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            cand[idx] = rnd
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # smooth annealed refinement
        try:
            cand = adam_refine(cand)
        except Exception:
            continue
        v = _max_cos(cand)
        if not np.isfinite(v):
            continue
        if v < bestv:
            bestv = v
            best = cand.copy()
            cur = cand.copy()
            curv = v
        else:
            if v < curv or rng.random() < np.exp((curv - v) / Temp):
                cur = cand.copy()
                curv = v
        # slow cooling
        Temp *= 0.9995
        if Temp < 0.0005:
            Temp = 0.0005
    remain = TIME_BUDGET - (time.time() - t0) - 1.0
    if remain > 5:
        try:
            best = lp_polish(best, seconds=min(remain, 35.0))
        except Exception:
            pass
    best = _norm(best)
    return best

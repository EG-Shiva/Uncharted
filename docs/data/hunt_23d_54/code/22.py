import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        cols_b = np.empty((p, 2 * d + 1), dtype=int)
        cols_b[:, :d] = I[:, None] * d + np.arange(d)
        cols_b[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols_b[:, 2 * d] = n * d
        vals_b = np.empty((p, 2 * d + 1))
        vals_b[:, :d] = x[J]
        vals_b[:, d:2 * d] = x[I]
        vals_b[:, 2 * d] = -1.0
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals_b.ravel(), (rows, cols_b.ravel())), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_desc(y, iters, k0, k1, lr, rng):
    n, d = y.shape
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for it in range(iters):
        frac = it / max(1, iters - 1)
        k = k0 + (k1 - k0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = np.exp(k * (G - gmax))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            break
        W = E / (2.0 * s)
        grad = W @ y + W.T @ y
        grad *= 0.5
        dot = np.sum(grad * y, axis=1, keepdims=True)
        g = grad - dot * y
        t += 1
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        y = y - step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    HARD = 168.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((54, 23))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (54, 23):
        tmp = rng.standard_normal((54, 23))
        r = min(xb.size, tmp.size)
        tmp.ravel()[:r] = xb.ravel()[:r]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    x_best = xb.copy()
    n, d = 54, 23
    # precompute nothing else
    trial = 0
    # exploration budget: leave 22s for final LP
    while time.time() - t0 < HARD - 22.0:
        trial += 1
        y = x_best.copy()
        u = rng.random()
        if u < 0.45:
            sigma = float(rng.uniform(0.02, 0.22))
            y = y + sigma * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif u < 0.70:
            ksub = int(rng.integers(1, 4))
            idx = rng.choice(n, size=ksub, replace=False)
            R = rng.standard_normal((ksub, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[idx] = R
        else:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            ksub = int(rng.integers(1, 4))
            worst = np.argsort(-per)[:ksub]
            keep = np.ones(n, dtype=bool)
            keep[worst] = False
            fixed = y[keep]
            # hole re-insertion: sample candidates, pick best hole
            for w in worst:
                M = 2000
                C = rng.standard_normal((M, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                vals = (C @ fixed.T).max(axis=1)
                y[w] = C[int(np.argmin(vals))]
                fixed = np.vstack([fixed, y[w][None, :]])
            # small jitter of neighbours to open topology
            y += 0.01 * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        iters = int(rng.integers(180, 420))
        k0 = float(rng.uniform(30, 100))
        k1 = float(rng.uniform(200, 600))
        lr = float(rng.uniform(0.004, 0.016))
        y2 = smooth_desc(y, iters, k0, k1, lr, rng)
        v2 = max_cos(y2)
        if v2 < best - 1e-12:
            best = v2
            x_best = y2
        if time.time() - t0 > HARD - 22.0:
            break
    rem = HARD - (time.time() - t0) - 1.0
    if rem > 3:
        rem = min(rem, 21.0)
        x_best = lp_polish(x_best, rem)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

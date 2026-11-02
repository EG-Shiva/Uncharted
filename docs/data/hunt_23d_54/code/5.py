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
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        c = c_all[sel]
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        # interleave per row: x[J] block, x[I] block, -1
        cols_blocks = np.empty((p, 2 * d + 1), dtype=int)
        cols_blocks[:, :d] = I[:, None] * d + np.arange(d)
        cols_blocks[:, d:2*d] = J[:, None] * d + np.arange(d)
        cols_blocks[:, 2*d] = n * d
        cols = cols_blocks.ravel()
        vals_blocks = np.empty((p, 2 * d + 1))
        vals_blocks[:, :d] = x[J]
        vals_blocks[:, d:2*d] = x[I]
        vals_blocks[:, 2*d] = -1.0
        vals = vals_blocks.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def adam_smooth(y, iters, k0, k1, lr, rng=None):
    n, d = y.shape
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for it in range(iters):
        k = k0 + (k1 - k0) * (it / max(1, iters - 1))
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights exp(k*(G-gmax))
        E = np.exp(k * (G - gmax))
        np.fill_diagonal(E, 0.0)
        s = E.sum() * 0.5
        if s <= 0 or not np.isfinite(s):
            break
        W = E / s
        grad = W @ y
        # tangent projection
        dot = (grad * y).sum(axis=1, keepdims=True)
        g = grad - dot * y
        t += 1
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        # decay slightly
        y = y - step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def best_hole_candidates(fixed, m, rng):
    # fixed: (nf,d) kept points, find candidate minimizing max inner
    M = 3000
    C = rng.standard_normal((M, fixed.shape[1]))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ fixed.T
    vals = S.max(axis=1)
    return C[int(np.argmin(vals))]

def solve():
    np.random.seed(0)
    rng = np.random.default_rng(12345)
    t_start = time.time()
    HARD = 178.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((54, 23))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (54, 23):
        # try to fix / fallback
        tmp = rng.standard_normal((54, 23))
        r = min(xb.size, tmp.size)
        tmp.ravel()[:r] = xb.ravel()[:r]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    x_best = xb.copy()
    n, d = 54, 23
    trial = 0
    # reserve ~25s for final polish
    while time.time() - t_start < HARD - 28:
        trial += 1
        rtype = rng.random()
        y = x_best.copy()
        if rtype < 0.40:
            sigma = float(rng.uniform(0.02, 0.35))
            y = x_best + sigma * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif rtype < 0.70:
            ksub = int(rng.integers(1, 6))
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
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            fixed = y[keep_mask]
            for w in worst:
                # fixed excludes all worst to allow joint search; use current fixed
                cand = best_hole_candidates(fixed, 1, rng)
                y[w] = cand
                # add it to fixed for next insertion in same trial
                fixed = np.vstack([fixed, cand[None, :]])
        # smooth params randomized
        iters = int(rng.integers(500, 1300))
        k0 = float(rng.uniform(40, 120))
        k1 = float(rng.uniform(250, 800))
        lr = float(rng.uniform(0.004, 0.018))
        y2 = adam_smooth(y, iters, k0, k1, lr)
        v2 = max_cos(y2)
        # only LP-polish promising ones to save time
        if v2 > best + 0.015:
            continue
        # quick LP
        remain = HARD - 28 - (time.time() - t_start)
        if remain < 2.5:
            break
        sec = 2.0 if (v2 > best) else 3.0
        sec = min(sec, remain)
        y3 = lp_polish(y2, sec)
        v3 = max_cos(y3)
        if v3 < best - 1e-12:
            best = v3
            x_best = y3
            # intensify immediately with a bit more polish if time allows
            if HARD - (time.time() - t_start) > 35:
                y4 = lp_polish(x_best, 2.0)
                v4 = max_cos(y4)
                if v4 < best:
                    best = v4
                    x_best = y4
        # time guard
        if time.time() - t_start > HARD - 28:
            break
    # final long polish of best
    rem = HARD - (time.time() - t_start) - 1.0
    if rem > 2:
        rem = min(rem, 26.0)
        x_best = lp_polish(x_best, rem)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

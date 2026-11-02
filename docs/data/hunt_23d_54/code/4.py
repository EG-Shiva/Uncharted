import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    if t_deadline is None:
        t_deadline = start + seconds
    else:
        t_deadline = min(start + seconds, t_deadline)
    iu = np.triu_indices(n, 1)
    while time.time() < t_deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq,
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
    return x


def smooth_anneal(x, iters=2500, T0=0.03, T1=0.003, lr0=0.05, lr1=0.005):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Gu = G[iu]
        gmax = Gu.max()
        E = np.exp((Gu - gmax) / T)
        s = E.sum()
        if not np.isfinite(s) or s == 0:
            break
        Wu = E / s
        W = np.zeros((n, n))
        W[iu] = Wu
        W = W + W.T
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def refine_hole(h, Y, iters=300, T0=0.01, lr0=0.1):
    h = h / np.linalg.norm(h)
    T, lr = T0, lr0
    for _ in range(iters):
        s = Y @ h
        smax = s.max()
        w = np.exp((s - smax) / max(T, 1e-12))
        ws = w.sum()
        if not np.isfinite(ws) or ws == 0:
            break
        w /= ws
        grad = w @ Y
        gt = grad - float(grad @ h) * h
        h = h - lr * gt
        n = np.linalg.norm(h)
        if n == 0 or not np.isfinite(n):
            break
        h /= n
        T *= 0.997
        lr *= 0.997
    return h / np.linalg.norm(h)


def find_hole(Y, rng, n_sample=30000):
    n, d = Y.shape
    S = rng.standard_normal((n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ Y.T
    vals = dots.max(axis=1)
    idx = int(np.argmin(vals))
    h = S[idx].copy()
    h = refine_hole(h, Y)
    return h


def contact_counts(x, tol=0.005):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    return (g > m - tol).sum(axis=1), m


def solve():
    t0 = time.time()
    TIME_HARD = 170.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    rng = np.random.default_rng(20261102)
    cands = []
    if xb is not None:
        xb = np.asarray(xb, dtype=float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = np.asarray(xr, dtype=float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        X = rng.standard_normal((54, 23))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        best = X
    else:
        vals = [max_cos(c) for c in cands]
        best = cands[int(np.argmin(vals))].copy()
    best_val = max_cos(best)
    n, d = best.shape

    sigmas = [0.03, 0.015, 0.06, 0.008, 0.04, 0.01]
    trial = 0
    # exploration phase, reserve time for final polish
    while time.time() - t0 < 135.0 and trial < 60:
        trial += 1
        remain = 135.0 - (time.time() - t0)
        if remain < 5:
            break
        counts, _ = contact_counts(best, tol=0.005)
        order = np.argsort(counts)
        if trial % 6 == 0:
            # double-point hole reinsertion
            k_low = rng.choice(order[:8])
            cand_h = order[-8:]
            k_high = rng.choice(cand_h)
            if k_high == k_low:
                k_high = rng.integers(0, n)
                while k_high == k_low:
                    k_high = rng.integers(0, n)
            victims = sorted([int(k_low), int(k_high)])
            Y = np.delete(best, victims, axis=0)
            h1 = find_hole(Y, rng, n_sample=25000)
            Y2 = np.vstack([Y, h1])
            h2 = find_hole(Y2, rng, n_sample=25000)
            Xtry = np.vstack([Y2, h2])
            Xtry = Xtry + rng.standard_normal(Xtry.shape) * 0.003
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            iters = 2500
        elif trial % 3 == 0:
            # single-point hole reinsertion, alternate low/high victim
            if (trial // 3) % 2 == 0:
                k = int(rng.choice(order[:8]))
            else:
                k = int(rng.choice(order[-8:]))
            Y = np.delete(best, k, axis=0)
            h = find_hole(Y, rng, n_sample=30000)
            Xtry = np.vstack([Y, h])
            Xtry = Xtry + rng.standard_normal(Xtry.shape) * 0.003
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            iters = 2500
        else:
            sigma = float(sigmas[trial % len(sigmas)])
            Xtry = best + rng.standard_normal(best.shape) * sigma
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            iters = 3000
        # smooth annealing with slightly varying schedule
        T0 = 0.025 + 0.01 * ((trial % 3) / 2.0)
        Xtry = smooth_anneal(Xtry, iters=iters, T0=T0, T1=0.0025,
                             lr0=0.05, lr1=0.004)
        # short LP polish
        budget = min(3.0, 135.0 - (time.time() - t0))
        if budget < 0.5:
            break
        Xtry = lp_polish(Xtry, seconds=budget)
        v = max_cos(Xtry)
        if v < best_val - 1e-12:
            best, best_val = Xtry, v
        if time.time() - t0 > TIME_HARD - 30:
            break

    # final long polish of the best
    left = TIME_HARD - (time.time() - t0)
    if left > 2:
        best = lp_polish(best, seconds=min(left, 30.0))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

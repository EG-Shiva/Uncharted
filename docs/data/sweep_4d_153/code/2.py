import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_descent(x, steps=550, T0=0.025, T1=0.003, s0=0.015, s1=0.0015):
    x = x.copy()
    n = x.shape[0]
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        s = s0 + (s1 - s0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = G.max()
        # weights, avoid overflow
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        norms = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        step = Gt / norms
        x = x - s * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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

def hole_candidate(others, rng, M=4000):
    cand = rng.normal(size=(M, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    return cand[np.argmin(worst)]

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 148.0
    rng = np.random.default_rng(0)
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
    for a in (xb, xr):
        if a is not None:
            a = np.asarray(a, dtype=float)
            if a.shape == (153, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append((max_cos(a), a))
    if len(cands) == 0:
        x0 = rng.normal(size=(153, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        cands.sort(key=lambda t: t[0])
        x0 = cands[0][1].copy()
    best = x0.copy()
    best_val = max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n = 153
    Tb = 0.0025
    it = 0
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 25:
        it += 1
        u = rng.random()
        # generate kick from cur
        if u < 0.30:
            sig = 0.008 + 0.008 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 500
            T0, T1 = 0.02, 0.0025
        elif u < 0.52:
            k = int(rng.integers(1, 6))
            idx = rng.choice(n, size=k, replace=False)
            y = cur.copy()
            y[idx] += rng.normal(size=(k, 4)) * 0.28
            y += rng.normal(size=y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 650
            T0, T1 = 0.025, 0.003
        elif u < 0.72:
            k = int(rng.integers(5, 18))
            idx = rng.choice(n, size=k, replace=False)
            y = cur.copy()
            y[idx] += rng.normal(size=(k, 4)) * 0.12
            y += rng.normal(size=y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 650
            T0, T1 = 0.025, 0.003
        elif u < 0.88:
            # hole reinsertion: replace 1-2 points by largest holes
            y = cur.copy()
            nr = 1 if rng.random() < 0.7 else 2
            ridx = rng.choice(n, size=nr, replace=False)
            for r in ridx:
                others = np.delete(y, r, axis=0)
                y[r] = hole_candidate(others, rng, M=4000)
            y += rng.normal(size=y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 650
            T0, T1 = 0.022, 0.0025
        else:
            sig = 0.04 + 0.03 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 550
            T0, T1 = 0.03, 0.004
        ys = smooth_descent(y, steps=steps, T0=T0, T1=T1, s0=0.015, s1=0.0012)
        v = max_cos(ys)
        # basin acceptance
        accept = False
        if v < cur_val:
            accept = True
        else:
            if rng.random() < np.exp((cur_val - v) / Tb):
                accept = True
        if accept:
            cur, cur_val = ys, v
        # promising? quick LP refine
        if v < best_val + 0.0025 and accept:
            # only polish if time allows
            if time.time() < EXPLORE_END - 2:
                z = lp_polish(ys, budget=1.2)
                vz = max_cos(z)
                if vz < best_val - 1e-12:
                    best, best_val = z, vz
                    cur, cur_val = z.copy(), vz
                elif vz < cur_val:
                    cur, cur_val = z, vz
            else:
                if v < best_val - 1e-12:
                    best, best_val = ys, v
        # reset wandering cur
        if cur_val > best_val + 0.012:
            cur = best.copy() + rng.normal(size=best.shape) * 0.008
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_val = max_cos(cur)
            # anneal basin temp slightly
            Tb = max(0.0012, Tb * 0.98)
    # final polish of best with remaining time
    remaining = DEADLINE - time.time() - 2.0
    if remaining > 3:
        best = lp_polish(best, budget=min(remaining, 22.0))
        best_val = max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

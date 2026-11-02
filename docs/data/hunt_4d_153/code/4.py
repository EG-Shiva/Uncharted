import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
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

def anneal_softmax(x, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # stable softmax weights in [0,1]
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-300)
        A = (W @ x) / rs
        dot = np.sum(A * x, axis=1, keepdims=True)
        T = A - dot * x
        x = x - lr * T
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def largest_void(x, exclude, n_sample, rng):
    n, d = x.shape
    Z = rng.standard_normal(size=(n_sample, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    mask = np.ones(n, dtype=bool)
    mask[exclude] = False
    Y = x[mask]
    D = Z @ Y.T
    vals = D.max(axis=1)
    return Z[int(np.argmin(vals))]

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(0)
    # load starting point
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            y = np.load(fn)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if y.shape == (153, 4) and np.all(np.isfinite(y)):
                v = max_cos(y)
                if v < best_val:
                    best_val = v
                    best = y.copy()
        except Exception:
            pass
    if best is None:
        z = rng.standard_normal(size=(153, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = max_cos(best)
    n, d = best.shape
    sigmas = [0.02, 0.04, 0.015, 0.06, 0.03, 0.09]
    it = 0
    MAX_IT = 80
    while it < MAX_IT and (time.time() - t0) < 150.0:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        y = best + sigma * rng.standard_normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # targeted surgery every few iterations: replace overloaded points by voids
        if it % 3 == 0:
            try:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                cnt = np.sum(G > m - 1e-3, axis=1).astype(float)
                # number to replace 1 or 2
                krep = 1 + (it % 2)
                done = set()
                for _ in range(krep):
                    idx = int(np.argmax(cnt))
                    if idx in done:
                        break
                    done.add(idx)
                    cnt[idx] = -1e9
                    v = largest_void(y, idx, 15000, rng)
                    y[idx] = v
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                pass
        # annealed repulsion, vary schedule slightly
        if it % 2 == 0:
            y2 = anneal_softmax(y, steps=500, a0=50.0, a1=700.0, lr0=0.03, lr1=0.003)
        else:
            y2 = anneal_softmax(y, steps=650, a0=30.0, a1=900.0, lr0=0.025, lr1=0.002)
        v2 = max_cos(y2)
        # screen: only LP-polish if promising or periodic
        if v2 > best_val + 0.004 and (it % 8 != 0):
            # keep diversity: if much worse skip
            continue
        # short LP polish
        remain = DEADLINE - (time.time() - t0)
        if remain < 4:
            break
        y3 = lp_polish(y2, seconds=1.6)
        v3 = max_cos(y3)
        if v3 < best_val - 1e-12 and np.all(np.isfinite(y3)):
            best = y3
            best_val = v3
    # final long polish of champion
    remain = DEADLINE - (time.time() - t0)
    if remain > 4:
        best = lp_polish(best, seconds=min(remain - 1.0, 14.0))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # strict finite check
    best = np.where(np.isfinite(best), best, 0.0)
    norms = np.linalg.norm(best, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best = best / norms
    return best

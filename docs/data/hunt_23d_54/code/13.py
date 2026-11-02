import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _anneal(x_init, steps, k0, k1, lr0, lr1, rng=None):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 else k1
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 else lr1
        G = x @ x.T
        # current max for shift stability
        m = float(G[iu].max())
        # exp weights, ignore diagonal
        # clip argument to avoid underflow overhead
        A = k * (G - m)
        # threshold very negative -> 0
        A[A < -12] = -12
        E = np.exp(A)
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        step = Gt / deg
        # normalize step scale
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _best_hole_vector(others, rng, n_cand=2000):
    # others: (m,d) unit vectors, find unit v minimizing max dot
    d = others.shape[1]
    cand = rng.randn(n_cand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    j = int(np.argmin(worst))
    return cand[j]

def _lp_polish(x, seconds, t0=None, budget=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
        if vr < best:
            best, x_best = vr, xr.copy()
    except Exception:
        pass
    x_cur = x_best.copy()
    cur = best
    Temp = 0.003
    it = 0
    # reserve 12s for final LP
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        remain = BUDGET - 12.0 - (time.time() - t0)
        if remain < 2.0:
            break
        mode = it % 8
        x = x_cur.copy()
        # choose kick
        if mode == 0:
            x += rng.randn(n, d) * 0.008
            steps, k0, k1, lr0, lr1 = 450, 150.0, 1800.0, 0.02, 0.002
        elif mode == 1:
            x += rng.randn(n, d) * 0.025
            steps, k0, k1, lr0, lr1 = 550, 120.0, 2200.0, 0.02, 0.002
        elif mode == 2:
            # replace worst 1 with hole + small noise
            _, G = _per_point_max(x)
            worst = int(np.argmax(G.max(axis=1)))
            others = np.delete(x, worst, axis=0)
            x[worst] = _best_hole_vector(others, rng, n_cand=2000)
            x += rng.randn(n, d) * 0.006
            steps, k0, k1, lr0, lr1 = 550, 150.0, 2000.0, 0.02, 0.002
        elif mode == 3:
            _, G = _per_point_max(x)
            rm = G.max(axis=1)
            idx = np.argsort(-rm)[:2]
            for w in idx:
                others = np.delete(x, w, axis=0)
                x[w] = _best_hole_vector(others, rng, n_cand=1500)
            x += rng.randn(n, d) * 0.006
            steps, k0, k1, lr0, lr1 = 600, 120.0, 2200.0, 0.02, 0.002
        elif mode == 4:
            x += rng.randn(n, d) * 0.09
            steps, k0, k1, lr0, lr1 = 700, 80.0, 1500.0, 0.025, 0.002
        elif mode == 5:
            # randomize 3 random points
            idx = rng.choice(n, 3, replace=False)
            for w in idx:
                v = rng.randn(d); v /= np.linalg.norm(v)
                x[w] = v
            steps, k0, k1, lr0, lr1 = 700, 80.0, 1800.0, 0.025, 0.002
        elif mode == 6:
            # jammed-region kick: points with highest contact count
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cnt = np.sum(G > cur - 0.02, axis=1)
            idx = np.argsort(-cnt)[:4]
            x[idx] += rng.randn(len(idx), d) * 0.10
            x += rng.randn(n, d) * 0.005
            steps, k0, k1, lr0, lr1 = 600, 120.0, 2000.0, 0.02, 0.002
        else:
            # loose-region kick
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cnt = np.sum(G > cur - 0.02, axis=1)
            idx = np.argsort(cnt)[:5]
            x[idx] += rng.randn(len(idx), d) * 0.08
            x += rng.randn(n, d) * 0.004
            steps, k0, k1, lr0, lr1 = 550, 150.0, 2000.0, 0.02, 0.002
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # adapt steps to remaining time
        if remain < 10:
            steps = min(steps, 250)
        x = _anneal(x, steps, k0, k1, lr0, lr1)
        v = _max_cos(x)
        if v < best:
            best = v
            x_best = x.copy()
            cur = v
            x_cur = x.copy()
        else:
            # simulated-annealing acceptance for wandering
            if v < cur or rng.rand() < np.exp(-(v - cur) / Temp):
                cur = v
                x_cur = x.copy()
            # occasionally restart from best
            if it % 12 == 0:
                x_cur = x_best.copy()
                cur = best
                Temp *= 0.92
    # final squeeze
    try:
        x_best = _lp_polish(x_best, seconds=11.0, t0=t0, budget=BUDGET - 0.5)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

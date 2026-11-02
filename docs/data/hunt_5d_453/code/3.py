import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_refine(x, steps, T0, T1, lr0, lr1, deadline, max_step=0.03):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    for s in range(steps):
        if time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        # geometric annealing
        T = T0 * (T1 / T0) ** frac if T0 != T1 else T0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        mv = G.max()
        D = (G - mv) / T
        E = np.zeros_like(G)
        msk = D > -12.0
        E[msk] = np.exp(D[msk])
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-30
        Gr = (E @ x) / row
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mhat = m / bc1
        vhat = v / bc2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step_t = step - sdot * x
        nn = np.linalg.norm(step_t, axis=1, keepdims=True) + 1e-30
        scale = np.minimum(1.0, max_step / nn)
        step_t *= scale
        x = x - step_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
    return x

def farthest_void(X, exclude_idx, rng, M=20000):
    n, d = X.shape
    cand = rng.normal(size=(M, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    mask = np.ones(n, dtype=bool)
    mask[exclude_idx] = False
    # max cosine to remaining points
    G = cand @ X[mask].T
    worst = G.max(axis=1)
    return cand[np.argmin(worst)]

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPLORE_END = t0 + 122.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    if xb.shape != (453, 5):
        # try transpose or truncate
        xb = np.array(xb, dtype=float).reshape(453, 5)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # also consider record
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if max_cos(xr) < max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = xb.shape
    best = max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur_val = best

    it = 0
    # initial quick smooth to get gradient-improved baseline
    x_best = adam_refine(x_best, 600, 0.004, 0.0006, 0.015, 0.003, min(HARD, EXPLORE_END))
    best = max_cos(x_best)
    x_cur = x_best.copy()
    cur_val = best

    while time.time() < EXPLORE_END:
        it += 1
        if time.time() > HARD - 5:
            break
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        worst_order = np.argsort(-row_max)
        cycle = it % 7
        if cycle == 0:
            K, sigma, steps = 25, 0.08, 1100
            T0, T1 = 0.005, 0.0005
            lr0, lr1 = 0.02, 0.002
            use_void = False
        elif cycle == 1:
            K, sigma, steps = 10, 0.04, 900
            T0, T1 = 0.004, 0.0004
            lr0, lr1 = 0.015, 0.002
            use_void = False
        elif cycle == 2:
            K, sigma, steps = 5, 0.02, 800
            T0, T1 = 0.003, 0.0003
            lr0, lr1 = 0.012, 0.002
            use_void = False
        elif cycle == 3:
            K, sigma, steps = 3, 0.01, 800
            T0, T1 = 0.002, 0.00025
            lr0, lr1 = 0.01, 0.0015
            use_void = False
        elif cycle == 4:
            # void reinsertion of single worst
            use_void = True
            steps = 1000
            T0, T1 = 0.004, 0.00035
            lr0, lr1 = 0.015, 0.002
            K, sigma = 1, 0.0
        elif cycle == 5:
            K, sigma, steps = 15, 0.05, 1000
            T0, T1 = 0.0045, 0.0004
            lr0, lr1 = 0.018, 0.002
            use_void = False
        else:
            K, sigma, steps = 8, 0.03, 900
            T0, T1 = 0.0035, 0.00035
            lr0, lr1 = 0.015, 0.002
            use_void = False

        if use_void:
            idx_w = int(worst_order[0])
            # occasionally replace 2 worst
            x_try = x_best.copy()
            nv = farthest_void(x_best, [idx_w], rng, M=20000)
            x_try[idx_w] = nv
            if it % 14 == 4:
                idx_w2 = int(worst_order[1])
                nv2 = farthest_void(x_try, [idx_w2], rng, M=15000)
                x_try[idx_w2] = nv2
            # tiny global jitter
            x_try += rng.normal(scale=0.0015, size=x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            idx = worst_order[:K].copy()
            if it % 3 == 0:
                extra = rng.integers(0, n, size=2)
                idx = np.unique(np.concatenate([idx, extra]))
            x_try = x_best.copy()
            # with small prob perturb current instead of best to diversify
            if it % 9 == 8 and cur_val < best + 0.003:
                x_try = x_cur.copy()
                # recompute worst w.r.t x_cur?
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                rc = Gc.max(axis=1)
                idx = np.argsort(-rc)[:K]
                x_try = x_cur.copy()
            noise = rng.normal(scale=sigma, size=(len(idx), d))
            x_try[idx] += noise
            x_try += rng.normal(scale=0.0015, size=x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        dl = min(HARD - 2, time.time() + 8.0, EXPLORE_END + 2.0)
        # don't exceed explore except last iter allowed to finish
        x_try = adam_refine(x_try, steps, T0, T1, lr0, lr1, min(dl, HARD - 2))
        v = max_cos(x_try)
        if v < best:
            best = v
            x_best = x_try
            x_cur = x_try.copy()
            cur_val = v
        else:
            if v < cur_val:
                x_cur = x_try.copy()
                cur_val = v
            # stochastic acceptance of slightly worse to change basin
            elif v < cur_val + 0.001 and rng.random() < 0.15:
                x_cur = x_try.copy()
                cur_val = v

    # final fine tune with small T then LP polish
    if time.time() < HARD - 10:
        x_best = adam_refine(x_best, 800, 0.0015, 0.0002, 0.008, 0.001, HARD - 8, max_step=0.015)
        best = max_cos(x_best)
    if time.time() < HARD - 2:
        x_best = lp_polish(x_best, HARD - 1.0)
        best = max_cos(x_best)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # guard against nan
    x_best = np.nan_to_num(x_best, nan=1.0 / np.sqrt(5.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

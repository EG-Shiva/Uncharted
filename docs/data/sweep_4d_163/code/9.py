import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > (m - window)
        # cap constraints for speed
        p = int(np.count_nonzero(sel))
        if p == 0:
            break
        if p > 2500:
            # keep top 2500
            idx = np.argpartition(c_all, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = 2500
        I = iu[0][sel]
        J = iu[1][sel]
        c_sel = c_all[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        ct = np.full(p, n * d)
        cols = np.concatenate([ci, cj, ct]) if False else None
        # build cols in per-row order: [I*d .., J*d .., t]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth_refine(x, steps, k0, k1, lr0, lr1, mu, noise0, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # weights, shift for stability
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x  # (n,d)
        # tangent projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        gt = gt / sc
        v = mu * v - lr * gt
        if noise0 > 0:
            temp = noise0 * (1.0 - frac)
            z = rng.randn(n, d)
            z -= np.sum(z * x, axis=1, keepdims=True) * x
            v += temp * z
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_candidates(x_keep, m, rng):
    cand = rng.randn(m, 4)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_keep.T
    worst = dots.max(axis=1)
    return cand, worst

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng0 = np.random.RandomState(0)
    # load start
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng0.randn(163, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (163, 4):
        # fix shape if needed
        tmp = rng0.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    best_x = xb.copy()
    best_v = _max_cos(best_x)
    n, d = best_x.shape
    # quick initial LP (short)
    best_x, best_v = _lp_polish(best_x, 4.0, T_END - 5.0)
    trial = 0
    while time.time() < T_END - 8.0:
        trial += 1
        rng = np.random.RandomState(1000 + trial * 7919)
        mode = trial % 3
        x = best_x.copy()
        # per-point worst value
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order_worst = np.argsort(-rowmax)
        order_best = np.argsort(rowmax)  # small rowmax = rattler-like
        if mode == 0:
            sigma = 0.015 if (trial % 6 != 0) else 0.035
            x = x + sigma * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1, mu, nz = 500, 25.0, 280.0, 0.012, 0.002, 0.85, 0.0015
        elif mode == 1:
            # shake worst cluster hard
            kbad = 12
            bad = order_worst[:kbad]
            shake = np.zeros_like(x)
            shake += 0.006 * rng.randn(n, d)
            shake[bad] += 0.09 * rng.randn(kbad, d)
            # project not needed, just add
            x = x + shake
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1, mu, nz = 600, 30.0, 320.0, 0.010, 0.0015, 0.88, 0.001
        else:
            # transplant 1-3 worst into voids
            nrep = 2 if (trial % 2 == 0) else 1
            # include one rattler sometimes to recycle
            reps = list(order_worst[:nrep])
            if trial % 6 == 2:
                reps[-1] = int(order_best[0])
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[reps] = False
            # iterative void fill
            for r in reps:
                x_keep = x[keep_mask]
                cand, worst = _void_candidates(x_keep, 3000, rng)
                bi = int(np.argmin(worst))
                x[r] = cand[bi]
                keep_mask[r] = True  # for next rep, include it
                # small jitter others
            x = x + 0.004 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1, mu, nz = 650, 20.0, 250.0, 0.012, 0.002, 0.85, 0.0008
        # smooth anneal
        x = _smooth_refine(x, steps, k0, k1, lr0, lr1, mu, nz, rng)
        v_smooth = _max_cos(x)
        # only LP-polish if promising to save time
        if v_smooth < best_v + 0.015 and time.time() < T_END - 6.0:
            # budget for LP depends on promise
            if v_smooth < best_v:
                lp_sec = 5.0
            else:
                lp_sec = 2.5
            # cap by remaining
            lp_sec = min(lp_sec, T_END - 6.0 - time.time())
            if lp_sec > 0.5:
                x, v_smooth = _lp_polish(x, lp_sec, T_END - 4.0)
        if v_smooth < best_v:
            best_v = float(v_smooth)
            best_x = x.copy()
            # early exit if big win? keep going to improve further
        # timeout guard
        if time.time() >= T_END - 6.0:
            break
    # final short polish on best if time left
    if time.time() < T_END - 2.0:
        sec = max(0.5, min(8.0, T_END - 2.0 - time.time()))
        best_x, best_v = _lp_polish(best_x, sec, T_END - 1.0)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

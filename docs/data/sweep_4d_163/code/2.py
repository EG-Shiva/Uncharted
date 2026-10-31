import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def row_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g


def lp_polish(x, seconds, verbose=False):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    # precompute for speed
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        # cap number of constraints for speed
        if np.count_nonzero(sel) > 2500:
            thr = np.sort(c_all[sel])[-2500] if np.count_nonzero(sel) > 2500 else m
            # keep only top 2500
            sel = c_all >= thr
            # ensure at least tight ones
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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


def smooth_polish(x, iters=600, T0=0.02, T1=0.002, lr0=0.008, lr1=0.001):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights, <=1
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        # sparsify tiny weights to reduce noise
        # (keep all, n small)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        # clip per-point step norm to 1 for stability
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    rng = np.random.default_rng(0)
    t_start = time.time()
    # load starting point
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
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    # pick best candidate
    best = None
    best_val = 1e9
    for c in cands:
        c = np.asarray(c, dtype=float)
        if c.shape != (163, 4):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    n, d = 163, 4
    sigmas = [0.012, 0.03, 0.06, 0.02, 0.045, 0.008, 0.025, 0.05]
    trial = 0
    # main basin-hopping loop, reserve time for final polish
    # aim to use ~135s for hopping
    while time.time() - t_start < 135 and trial < 60:
        sigma = sigmas[trial % len(sigmas)]
        move = trial % 4
        y = best.copy()
        if move == 0:
            # global shake
            y = y + rng.normal(scale=sigma, size=y.shape)
        elif move == 1:
            # cap-localized shake
            u = rng.normal(size=4)
            u /= np.linalg.norm(u)
            dots = y @ u
            mask = dots > 0.70  # cap
            if np.count_nonzero(mask) < 3:
                # pick nearest 12 to u
                idx = np.argsort(dots)[-12:]
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True
            noise = np.zeros_like(y)
            nmask = np.count_nonzero(mask)
            noise[mask] = rng.normal(scale=sigma * 2.0, size=(nmask, d))
            noise[~mask] = rng.normal(scale=sigma * 0.25, size=(n - nmask, d))
            y = y + noise
        elif move == 2:
            # worst-point shake
            rm, _ = row_maxs(y)
            k = 12
            idx = np.argsort(rm)[-k:]
            noise = rng.normal(scale=sigma * 0.3, size=y.shape)
            noise[idx] = rng.normal(scale=sigma * 2.5, size=(k, d))
            y = y + noise
        else:
            # rattler resample + small shake
            rm, _ = row_maxs(y)
            m = rm.max()
            ratt = np.where(rm < m - 5e-4)[0]
            y = y + rng.normal(scale=sigma * 0.5, size=y.shape)
            if len(ratt) > 0:
                for r in ratt:
                    v = rng.normal(size=4)
                    v /= np.linalg.norm(v)
                    # try to place in a void: pick best of 8 random candidates (farthest from others)
                    cand = rng.normal(size=(8, 4))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    # score by max dot to current set (excluding r)
                    tmp = y.copy()
                    scores = []
                    for cc in cand:
                        s = float(np.max(tmp[np.arange(n) != r] @ cc))
                        scores.append(s)
                    y[r] = cand[int(np.argmin(scores))]
            else:
                # no clear rattlers: relocate the single worst point randomly
                rm2, _ = row_maxs(y)
                w = int(np.argmax(rm2))
                # random jump for worst point
                if trial % 8 == 3:
                    v = rng.normal(size=4)
                    v /= np.linalg.norm(v)
                    y[w] = v
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth descent (fast)
        it = 500 if sigma > 0.03 else 650
        y = smooth_polish(y, iters=it)
        # short LP polish
        remain = 135 - (time.time() - t_start)
        sec = min(2.8, max(0.8, remain / (8 if trial < 20 else 4)))
        if sec > 0.4:
            y = lp_polish(y, seconds=sec)
        v = max_cos(y)
        if v < best_val - 1e-12:
            best = y
            best_val = v
        trial += 1
        # safety: if time running out break
        if time.time() - t_start > 134:
            break
    # final long polish of best
    elapsed = time.time() - t_start
    # leave ~10s margin (limit 180)
    budget = max(5.0, min(30.0, 170.0 - elapsed))
    best = lp_polish(best, seconds=budget)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # final safety: remove NaN
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

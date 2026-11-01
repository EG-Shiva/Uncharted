import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = iu_c > thresh
        # cap size for speed
        if int(sel.sum()) > 3000:
            # keep tightest 3000
            idx = np.where(sel)[0]
            order = np.argsort(iu_c[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I-block, J-block, t
        base_I = (I[:, None] * d + np.arange(d)).ravel()
        base_J = (J[:, None] * d + np.arange(d)).ravel()
        cols_IJ = np.concatenate([base_I, base_J])
        # build vals/cols interleaved correctly per row:
        # each row has x[J] at I cols, x[I] at J cols, -1 at t
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        # fill row by row via vectorized blocks
        # layout per row: [I_d, J_d, t]
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(7)
    # load start
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (173, 5):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(173, 5))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = 173, 5

    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        mask = G >= m - 1e-9
        cnt = mask.sum(axis=1)
        return m, cnt, G

    trial = 0
    # initial quick polish to ensure local opt (short)
    # main basin-hopping loop
    while True:
        rem = deadline - time.time()
        if rem < 7.0:
            break
        pt = 1.4
        if rem < 20:
            pt = max(0.6, rem - 6.0)
        pt = min(pt, rem - 6.0)
        if pt < 0.5:
            break
        _, cnt, _ = contact_info(best)
        order_desc = np.argsort(-cnt)
        # choose mode
        mode = trial % 6
        y = best.copy()
        if mode == 0:
            # kick jammed + random, medium
            k_rand = rng.choice(n, 5, replace=False)
            subset = np.unique(np.concatenate([order_desc[:3], k_rand]))
            sigma = 0.07
            y[subset] += sigma * rng.normal(size=(len(subset), d))
        elif mode == 1:
            # small global shake
            sigma = 0.015
            y += sigma * rng.normal(size=(n, d))
        elif mode == 2:
            # reinit worst point
            victim = int(order_desc[0])
            y[victim] = rng.normal(size=(d,))
            # also shake its contact neighbours a bit
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            neigh = np.where(G[victim] > m - 0.02)[0]
            if len(neigh) > 0:
                y[neigh] += 0.05 * rng.normal(size=(len(neigh), d))
        elif mode == 3:
            # kick rattlers + random medium set
            ratt = np.where(cnt == 0)[0]
            k_rand = rng.choice(n, 8, replace=False)
            subset = np.unique(np.concatenate([ratt, k_rand])) if len(ratt) else k_rand
            sigma = 0.10
            y[subset] += sigma * rng.normal(size=(len(subset), d))
        elif mode == 4:
            # broad moderate shake of 20 points
            subset = rng.choice(n, 20, replace=False)
            sigma = 0.04
            y[subset] += sigma * rng.normal(size=(len(subset), d))
        else:
            # tiny shake of high-contact neighbourhood
            subset = order_desc[:8]
            sigma = 0.025
            y[subset] += sigma * rng.normal(size=(len(subset), d))
        # renormalize
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        y = _lp_polish(y, pt)
        v = _max_cos(y)
        if v < best_val:
            best = y
            best_val = v
        trial += 1

    # final long polish
    rem = deadline - time.time()
    if rem > 1.2:
        best = _lp_polish(best, rem - 0.6)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

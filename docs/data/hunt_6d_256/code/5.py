import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, deadline, r_init=0.01):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(r_init)
    iu0 = np.triu_indices(n, 1)
    # pre-allocate not needed
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[0][sel]
        J = iu0[1][sel]
        cc = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            # keep worst 2500
            k = np.argpartition(-cc, 2500)[:2500]
            I = I[k]; J = J[k]; cc = cc[k]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        cT = np.full(p, n * d)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2*d+1] = 0  # placeholder, will overwrite fully below
        # build cols vectorized
        base = np.empty((p, 2 * d + 1), dtype=np.int64)
        base[:, :d] = I[:, None] * d + np.arange(d)
        base[:, d:2*d] = J[:, None] * d + np.arange(d)
        base[:, 2*d] = n * d
        cols = base.ravel()
        vals = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals[:, :d] = x[J]
        vals[:, d:2*d] = x[I]
        vals[:, 2*d] = -1.0
        A_ub = coo_matrix((vals.ravel(), (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            if best < m - 1e-12:
                pass
        else:
            radius *= 0.5
    return x

def contact_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    tight = G >= m - 1e-9
    # counts per point (each contact counted once per endpoint)
    cnt = np.sum(tight, axis=1)
    # per-point max
    np.fill_diagonal(G, -2.0)
    pmax = G.max(axis=1)
    return m, cnt, pmax, G

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    LOOP_END = t0 + 150.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((256, 6))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (256, 6):
        # try to fix / fallback
        xb = rng.standard_normal((256, 6))
    n, d = 256, 6
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    # initial contact info
    trial = 0
    sigmas_cycle = [0.08, 0.05, 0.015, 0.15, 0.03, 0.10]
    while time.time() < LOOP_END and time.time() < HARD_END - 25:
        trial += 1
        mode = trial % 6
        # refresh contact info from current best every few trials
        if trial % 2 == 1 or mode == 0 or mode == 4:
            try:
                m0, cnt0, pmax0, _ = contact_info(xb)
            except Exception:
                m0, cnt0 = best, np.zeros(n)
                pmax0 = np.zeros(n)
        y = xb.copy()
        r_init = 0.012
        if mode == 0:
            # shake overloaded points
            order = np.argsort(-cnt0)
            top = order[:24]
            # pick 8 among top 24 deterministically via rng
            sel_idx = rng.choice(top, size=min(8, len(top)), replace=False)
            sigma = 0.08
            y[sel_idx] += rng.standard_normal((len(sel_idx), d)) * sigma
            r_init = 0.015
        elif mode == 1:
            k = 15
            sel_idx = rng.choice(n, size=k, replace=False)
            sigma = 0.05
            y[sel_idx] += rng.standard_normal((k, d)) * sigma
            r_init = 0.012
        elif mode == 2:
            sigma = 0.015
            y += rng.standard_normal((n, d)) * sigma
            r_init = 0.01
        elif mode == 3:
            k = 5
            sel_idx = rng.choice(n, size=k, replace=False)
            sigma = 0.15
            y[sel_idx] += rng.standard_normal((k, d)) * sigma
            r_init = 0.02
        elif mode == 4:
            # void reinsertion: replace worst point by void centre
            # find worst point (largest pmax, break ties by cnt)
            worst_candidates = np.argsort(-(pmax0 + 1e-6 * cnt0))[:8]
            worst = worst_candidates[rng.integers(len(worst_candidates))]
            # find void direction
            R = rng.standard_normal((6000, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            D = R @ xb.T
            rowmax = D.max(axis=1)
            void_dir = R[np.argmin(rowmax)]
            y[worst] = void_dir + rng.standard_normal(d) * 0.01
            r_init = 0.02
        else:  # mode 5
            # shake rattlers + random neighbours
            ratt = np.where(cnt0 <= 0.5)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt0)[:5]
            sel2 = rng.choice(n, size=8, replace=False)
            sel_idx = np.unique(np.concatenate([ratt[:5], sel2]))[:12]
            sigma = 0.10
            y[sel_idx] += rng.standard_normal((len(sel_idx), d)) * sigma
            r_init = 0.015
        # renormalize shaken points
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        # short polish
        deadline = min(time.time() + 2.5, HARD_END - 22)
        if deadline - time.time() < 0.3:
            break
        y2 = lp_polish(y, deadline, r_init=r_init)
        v2 = max_cos(y2)
        if v2 < best - 1e-12:
            xb = y2
            best = v2
            # update contact info immediately
            try:
                m0, cnt0, pmax0, _ = contact_info(xb)
            except Exception:
                pass
            continue
        # promising? extend
        if v2 < best + 0.002:
            deadline2 = min(time.time() + 3.0, HARD_END - 22)
            if deadline2 - time.time() > 0.5:
                y3 = lp_polish(y2, deadline2, r_init=0.005)
                v3 = max_cos(y3)
                if v3 < best - 1e-12:
                    xb = y3
                    best = v3
                    continue
        # else discard
    # final long polish of best
    try:
        xb = lp_polish(xb, HARD_END - 1.5, r_init=0.005)
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # safety: no NaN, no zero rows
    xb = np.where(np.isfinite(xb), xb, 0.0)
    nn = np.linalg.norm(xb, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    xb = xb / nn
    return xb

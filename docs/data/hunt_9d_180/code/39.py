import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def ordinary_polish(x, time_left, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and time_left > 0:
        tstart = time.time()
        G = x @ x.T
        cfull = G[iu0, iu1]
        m = float(np.max(cfull))
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[sel]; J = iu1[sel]
        cs = cfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        vals = np.empty((p, 2 * d + 1))
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
        if time.time() >= deadline:
            break
        # avoid infinite loop on tiny time
        if time.time() - tstart < 1e-6:
            break
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    FEAS_END = t0 + 140.0
    np.random.seed(0)
    rng = np.random.RandomState(0)
    n, d = 180, 9
    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x0 = rng.randn(n, d)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        best_val = max_cos(x0)
        x_best = x0.copy()
    else:
        cands.sort(key=lambda t: t[0])
        best_val, x_best = cands[0][0], cands[0][1].copy()
    x = x_best.copy()
    cur_val = best_val
    iu0, iu1 = np.triu_indices(n, 1)
    targets = [0.4620, 0.4615, 0.4610]
    radius = 0.02
    excl = {}  # key -> expiry iter
    it = 0
    last_duals = None
    last_top_keys = None
    last_top_IJ = None
    # feasibility loop
    while time.time() < FEAS_END and time.time() < HARD_END - 30:
        it += 1
        t = targets[(it - 1) % len(targets)]
        # purge expired
        for k in list(excl.keys()):
            if excl[k] < it:
                del excl[k]
        G = x @ x.T
        vals_all = G[iu0, iu1]
        # pairs above t-0.01
        thr = t - 0.01
        midx = np.where(vals_all > thr)[0]
        if len(midx) > 800:
            # 800 tightest (largest)
            part = np.argpartition(-vals_all[midx], 800)[:800]
            sidx = midx[part]
        elif len(midx) > 0:
            sidx = midx
        else:
            sidx = np.array([], dtype=int)
        if len(sidx) > 0:
            I0 = iu0[sidx]; J0 = ju0 if False else iu1[sidx]
            c0 = vals_all[sidx]
            # sort descending for stress ranking
            order = np.argsort(-c0)
            I0 = I0[order]; J0 = J0[order]; c0 = c0[order]
            keys0 = I0.astype(np.int64) * n + J0.astype(np.int64)
            # refresh excluded subset every 5 iters
            if it % 5 == 1:
                # candidate pool: top 250 tightest
                pool = min(250, len(keys0))
                pool_keys = keys0[:pool]
                # choose 70 at random (rotating)
                kex = min(70, pool)
                if kex > 0:
                    pick = rng.choice(pool, kex, replace=False)
                    for pi in pick:
                        excl[int(pool_keys[pi])] = it + 5
                # purge again view
                for k in list(excl.keys()):
                    if excl[k] < it:
                        del excl[k]
            if len(excl) > 0:
                # filter excluded
                # vectorized isin with active keys
                akt = np.fromiter(excl.keys(), dtype=np.int64, count=len(excl))
                mask_keep = ~np.isin(keys0, akt)
                I0 = I0[mask_keep]; J0 = J0[mask_keep]; c0 = c0[mask_keep]
                keys0 = keys0[mask_keep]
            I, J, c = I0, J0, c0
        else:
            I = np.array([], dtype=int); J = np.array([], dtype=int); c = np.array([])
        p = len(I)
        if p == 0:
            v = max_cos(x)
            if v < best_val:
                best_val = v
                x_best = x.copy()
            # small tangent random walk to keep exploring
            step = rng.randn(n, d) * 0.005
            step -= (np.sum(step * x, axis=1, keepdims=True)) * x
            x = x + step
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            continue
        # cap (already capped) but if still >800 truncate
        if p > 800:
            I, J, c = I[:800], J[:800], c[:800]
            p = 800
        nv = n * d + p
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = (n * d + np.arange(p))[:, None] if False else n * d + np.arange(p)[:, None]
        # fix cols last column assignment
        cols[:, 2 * d] = n * d + np.arange(p)
        vals = np.empty((p, 2 * d + 1))
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        b_ub = t - c
        try:
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            continue
        obj = np.zeros(nv)
        obj[n * d:] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(0.0, None)] * p
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            continue
        if res.status != 0:
            # shrink radius briefly and kick tangentially
            radius = max(0.01, radius * 0.9)
            step = rng.randn(n, d) * 0.003
            step -= (np.sum(step * x, axis=1, keepdims=True)) * x
            # clip
            nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
            step = step * np.minimum(1.0, 0.01 / nm)
            x = x + step
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            continue
        try:
            dx = res.x[:n * d].reshape(n, d)
        except Exception:
            continue
        # duals for stress identification (best effort)
        try:
            md = np.asarray(res.ineqlin.marginals).ravel()
            if md.shape[0] == p:
                last_duals = np.abs(md)
        except Exception:
            last_duals = None
        # apply move + renormalise
        y = x + dx
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        x = y
        v = max_cos(x)
        cur_val = v
        if v < best_val:
            best_val = v
            x_best = x.copy()
        # guard against blow-up: reset to best with perturbation
        if v > 0.60:
            step = rng.randn(n, d) * 0.01
            step -= (np.sum(step * x_best, axis=1, keepdims=True)) * x_best
            x = x_best + step
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # keep radius in 0.01..0.03, gently vary
        if it % 10 == 0:
            radius = float(rng.choice([0.01, 0.015, 0.02, 0.03]))
        if time.time() >= FEAS_END:
            break
    # final ordinary polish on best found
    time_left = HARD_END - time.time() - 2.0
    if time_left > 5:
        try:
            x_best = ordinary_polish(x_best, time_left, HARD_END - 1.0)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # strict finiteness guard
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best

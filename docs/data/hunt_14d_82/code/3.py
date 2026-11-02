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
    radius = 3e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20 * radius, 1e-9)
        sel = c_all > thr
        if np.count_nonzero(sel) == 0:
            break
        # cap size for speed
        if np.count_nonzero(sel) > 1500:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            k = 1500
            part = np.argpartition(-vals, k - 1)[:k]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            # early exit if clearly worse than global? no, keep polishing
        else:
            radius *= 0.5
    return x

def _coarse_relax(x, steps=250, step0=0.03, decay=0.996):
    n, d = x.shape
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # adaptive window
        delta = min(0.15, max(0.05, m - 0.15))
        thr = m - delta
        W = G - thr
        W[W < 0] = 0.0
        # zero diag already
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        nrm = float(np.sqrt((Gt ** 2).mean()) + 1e-12)
        if nrm < 1e-14:
            break
        step = step0 * (decay ** s)
        x = x - (step / nrm) * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_vector(X_keep, rng, n_cand=3000):
    n_keep, d = X_keep.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_keep.T
    worst = dots.max(axis=1)
    return cand[int(np.argmin(worst))]

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    BUDGET = 168.0
    rng = _np.random.default_rng(0)
    # load start
    try:
        xb = _np.load("best.npy")
    except Exception:
        try:
            xb = _np.load("record.npy")
        except Exception:
            xb = rng.standard_normal(size=(82, 14))
    xb = _np.asarray(xb, dtype=float)
    if xb.shape != (82, 14):
        # try to fix / random
        tmp = rng.standard_normal(size=(82, 14))
        m = min(xb.size, 82*14)
        tmp.ravel()[:m] = xb.ravel()[:m]
        xb = tmp
    xb /= _np.linalg.norm(xb, axis=1, keepdims=True)
    # quick baseline polish if time allows (already polished)
    best_val = _max_cos(xb)
    global_best = xb.copy()
    current = xb.copy()
    cur_val = best_val
    it = 0
    sigmas = [0.08, 0.03, 0.15, 0.01, 0.05, 0.20]
    while _time.time() - t0 < BUDGET - 4.0:
        it += 1
        remain = BUDGET - 4.0 - (_time.time() - t0)
        if remain < 1.0:
            break
        r = rng.random()
        # choose base
        if it % 7 == 0:
            base = global_best
        else:
            base = current if rng.random() < 0.3 else global_best
        n, d = base.shape
        trial = base.copy()
        if r < 0.35:
            sigma = sigmas[it % len(sigmas)]
            trial = base + sigma * rng.standard_normal(size=base.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif r < 0.70:
            # subset shake biased to high degree / worst rows
            G = base @ base.T
            _np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # degree with window 0.01
            deg = (G > m - 0.01).sum(axis=1)
            order = _np.argsort(-deg)
            k = int(rng.integers(3, 12))
            n_top = min(n, k // 2 + 1)
            idx = list(order[:n_top])
            while len(idx) < k:
                j = int(rng.integers(0, n))
                if j not in idx:
                    idx.append(j)
            idx = _np.array(idx, dtype=int)
            sigma = sigmas[(it + 2) % len(sigmas)] * 1.5
            trial[idx] = base[idx] + sigma * rng.standard_normal(size=(len(idx), d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # remove worst k and reinsert into holes
            G = base @ base.T
            _np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = int(rng.integers(1, 4))
            worst_idx = _np.argsort(-rowmax)[:k]
            keep_mask = _np.ones(n, dtype=bool)
            keep_mask[worst_idx] = False
            X_keep = base[keep_mask]
            for j in worst_idx:
                h = _hole_vector(X_keep, rng, n_cand=3000)
                # small random jitter around hole
                h = h + 0.01 * rng.standard_normal(size=d)
                h /= _np.linalg.norm(h)
                trial[j] = h
                X_keep = _np.vstack([X_keep, h[None, :]])
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        # coarse relaxation (fast)
        trial = _coarse_relax(trial, steps=220, step0=0.03, decay=0.996)
        # short LP polish
        budget_lp = min(1.4, remain - 0.2)
        if budget_lp < 0.3:
            budget_lp = 0.3
        trial = _lp_polish(trial, budget_lp)
        v = _max_cos(trial)
        if v < best_val - 1e-13:
            global_best = trial.copy()
            best_val = v
            current = trial.copy()
            cur_val = v
            # extra polish on new record if time permits
            if _time.time() - t0 < BUDGET - 6.0:
                global_best = _lp_polish(global_best, min(2.0, BUDGET - 4.0 - (_time.time() - t0)))
                best_val = _max_cos(global_best)
                current = global_best.copy()
                cur_val = best_val
        elif v < cur_val:
            current = trial.copy()
            cur_val = v
        else:
            Temp = 0.002
            if rng.random() < _np.exp((cur_val - v) / Temp):
                current = trial.copy()
                cur_val = v
        if it % 15 == 0:
            current = global_best.copy()
            cur_val = best_val
    global_best /= _np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best

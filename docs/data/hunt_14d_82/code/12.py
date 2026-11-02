import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1), float(np.max(g)), g

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 3e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1400:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            k = 1400
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
        else:
            radius *= 0.5
    return x

def _relax(x, steps=260, p_pow=2.0, step0=0.04, decay=0.997, wdelta=0.08):
    n, d = x.shape
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - wdelta
        W = G - thr
        W[W < 0] = 0.0
        if p_pow != 1.0:
            # emphasise top: W^p
            W = W ** p_pow
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        nrm = float(np.sqrt((Gt ** 2).mean()) + 1e-12)
        if nrm < 1e-15:
            break
        step = step0 * (decay ** s)
        x = x - (step / nrm) * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _refined_hole(X_keep, rng, n_cand=4000, refine=80):
    n_keep, d = X_keep.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_keep.T
    worst = dots.max(axis=1)
    h = cand[int(np.argmin(worst))].copy()
    # local ascent: minimise max dot (softmax)
    lr = 0.05
    for s in range(refine):
        dv = X_keep @ h  # (n_keep,)
        mv = float(np.max(dv))
        T = 0.005 + 0.02 * (1.0 - s / max(refine - 1, 1))
        E = np.exp((dv - mv) / T)
        E /= (np.sum(E) + 1e-30)
        g = E @ X_keep  # (d,)
        g = g - (g @ h) * h
        nrm = float(np.linalg.norm(g) + 1e-12)
        if nrm < 1e-14:
            break
        h = h - lr * g / nrm
        h /= np.linalg.norm(h) + 1e-30
        lr *= 0.99
    return h

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = _np.random.default_rng(42)
    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (82, 14):
                a /= _np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        xb = rng.standard_normal(size=(82, 14))
        xb /= _np.linalg.norm(xb, axis=1, keepdims=True)
    else:
        vals = [(_max_cos(c), c) for c in cands]
        vals.sort(key=lambda t: t[0])
        xb = vals[0][1].copy()
    best_val = _max_cos(xb)
    global_best = xb.copy()
    current = xb.copy()
    cur_val = best_val
    n, d = 82, 14
    sigmas = [0.02, 0.05, 0.10, 0.20, 0.01, 0.07, 0.15]
    it = 0
    # pre-generate random planes for shear
    while _time.time() < deadline - 3.0:
        it += 1
        remain = deadline - 3.0 - (_time.time() - t0)
        if remain < 0.8:
            break
        # choose base
        if it % 9 == 0:
            base = global_best
        elif rng.random() < 0.35:
            base = current
        else:
            base = global_best
        r = rng.random()
        trial = base.copy()
        if r < 0.28:
            sigma = sigmas[it % len(sigmas)]
            trial = base + sigma * rng.standard_normal(size=base.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif r < 0.52:
            # hub-targeted shake
            rowm, m, G = _rowmax(base)
            deg = (G > m - 0.008).sum(axis=1)
            order = _np.argsort(-deg)
            k = int(rng.integers(4, 11))
            idx = list(order[:max(2, k // 2)])
            while len(idx) < k:
                j = int(rng.integers(0, n))
                if j not in idx:
                    idx.append(j)
            idx = _np.array(idx, dtype=int)
            sigma = sigmas[(it + 3) % len(sigmas)] * 1.6 + 0.02
            trial[idx] = base[idx] + sigma * rng.standard_normal(size=(len(idx), d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif r < 0.74:
            # kill worst k and reinsert into refined holes
            rowm, m, G = _rowmax(base)
            k = int(rng.integers(1, 6))
            if rng.random() < 0.3:
                k = int(rng.integers(6, 9))
            worst_idx = _np.argsort(-rowm)[:k]
            keep_mask = _np.ones(n, dtype=bool)
            keep_mask[worst_idx] = False
            X_keep = base[keep_mask]
            # sequentially insert
            new_pts = []
            for j in worst_idx:
                h = _refined_hole(X_keep, rng, n_cand=4000, refine=60)
                h = h + 0.005 * rng.standard_normal(size=d)
                h /= _np.linalg.norm(h)
                new_pts.append(h)
                X_keep = _np.vstack([X_keep, h[None, :]])
            for j, h in zip(worst_idx, new_pts):
                trial[j] = h
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # coherent hemisphere rotation / shear
            u = rng.standard_normal(size=d); u /= _np.linalg.norm(u)
            proj = base @ u
            med = _np.median(proj)
            # random skew
            A = rng.standard_normal(size=(d, d)) * 0.5
            A = A - A.T
            eps = float(rng.uniform(0.05, 0.25))
            mask = proj > med if rng.random() < 0.5 else proj > 0
            if int(mask.sum()) < 5 or int(mask.sum()) > n - 5:
                mask = proj > med
            disp = (A @ base.T).T  # (n,d)
            # project to tangent
            disp = disp - (disp * base).sum(axis=1, keepdims=True) * base
            trial = base.copy()
            trial[mask] = base[mask] + eps * disp[mask]
            # small noise everywhere
            trial = trial + 0.01 * rng.standard_normal(size=base.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        # fast relaxation: two stages
        trial = _relax(trial, steps=140, p_pow=1.0, step0=0.05, decay=0.996, wdelta=0.12)
        trial = _relax(trial, steps=160, p_pow=2.0, step0=0.03, decay=0.997, wdelta=0.06)
        vc = _max_cos(trial)
        # filter: only LP if promising
        if vc > best_val + 0.012 and rng.random() > 0.05:
            # accept as current with small prob, skip LP
            if vc < cur_val:
                current = trial.copy(); cur_val = vc
            continue
        budget_lp = 0.9
        if vc < best_val + 0.002:
            budget_lp = 1.3
        # cap by remain
        budget_lp = min(budget_lp, remain - 0.2)
        if budget_lp < 0.25:
            budget_lp = 0.25
        trial = _lp_polish(trial, budget_lp, deadline=deadline - 2.5)
        v = _max_cos(trial)
        if v < best_val - 1e-13:
            global_best = trial.copy()
            best_val = v
            current = trial.copy(); cur_val = v
            # bonus polish on record
            if _time.time() < deadline - 5.0:
                extra = min(1.8, deadline - 3.0 - _time.time())
                if extra > 0.4:
                    global_best = _lp_polish(global_best, extra, deadline=deadline - 2.5)
                    best_val = _max_cos(global_best)
                    current = global_best.copy(); cur_val = best_val
        elif v < cur_val:
            current = trial.copy(); cur_val = v
        else:
            Temp = 0.0025 * max(1.0, (v - best_val) * 50.0 + 1.0)
            # cooler as time goes
            if rng.random() < _np.exp((cur_val - v) / 0.0025):
                current = trial.copy(); cur_val = v
        if it % 20 == 0:
            current = global_best.copy(); cur_val = best_val
    global_best /= _np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best

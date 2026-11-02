import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 3e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    # limit total LP iterations to bound time
    it = 0
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t_end is not None and now >= t_end:
            break
        if radius < 1e-13:
            break
        if it > 40:
            break
        it += 1
        G = x @ x.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1300:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            k = 1300
            part = np.argpartition(-vals, k - 1)[:k]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": max(0.2, min(2.0, seconds))})
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


def _coarse_relax(x, steps=200, step0=0.03, decay=0.996):
    n, d = x.shape
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        delta = m - 0.15
        if delta < 0.05:
            delta = 0.05
        elif delta > 0.15:
            delta = 0.15
        thr = m - delta
        W = G - thr
        W[W < 0.0] = 0.0
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        nrm = float(np.sqrt((Gt * Gt).mean()) + 1e-12)
        if nrm < 1e-14 or not np.isfinite(nrm):
            break
        step = step0 * (decay ** s)
        x = x - (step / nrm) * Gt
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _hole_vector(X_keep, rng, n_cand=4000):
    d = X_keep.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_keep.T
    worst = dots.max(axis=1)
    h = cand[int(np.argmin(worst))].copy()
    # local refinement: push away from nearest neighbours
    for _ in range(25):
        g = X_keep @ h
        # focus on close ones
        m = float(np.max(g))
        w = np.exp((g - m) / 0.02)
        grad = w @ X_keep
        grad = grad - (grad @ h) * h
        nrm = float(np.linalg.norm(grad))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        h = h - (0.05 / (nrm + 1e-12)) * grad
        nrm2 = float(np.linalg.norm(h))
        if nrm2 == 0:
            break
        h /= nrm2
    return h


def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (82, 14):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        xb = rng.standard_normal(size=(82, 14))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    else:
        cands.sort(key=lambda c: _max_cos(c))
        xb = cands[0].copy()
    n, d = 82, 14
    best_val = _max_cos(xb)
    global_best = xb.copy()
    current = xb.copy()
    cur_val = best_val
    # quick initial tight polish if we have time (2s)
    if t_end - time.time() > 10.0:
        global_best = _lp_polish(global_best, min(2.5, t_end - time.time() - 8.0))
        best_val = _max_cos(global_best)
        current = global_best.copy()
        cur_val = best_val
    sigmas = [0.08, 0.03, 0.15, 0.01, 0.05, 0.20, 0.10]
    it = 0
    while time.time() < t_end - 5.0:
        it += 1
        remain = t_end - 5.0 - (time.time() - t0 + 0.0) * 0.0
        # remain until exploration deadline
        explor_dead = t_end - 5.0
        if time.time() >= explor_dead:
            break
        # choose base
        if it % 9 == 0:
            base = global_best
        else:
            base = current if rng.random() < 0.35 else global_best
        trial = base.copy()
        r = rng.random()
        if r < 0.32:
            sigma = sigmas[it % len(sigmas)]
            trial = base + sigma * rng.standard_normal(size=base.shape)
            nr = np.linalg.norm(trial, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            trial /= nr
        elif r < 0.65:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = (G > m - 0.012).sum(axis=1)
            order = np.argsort(-deg)
            k = int(rng.integers(3, 11))
            n_top = min(n, k // 2 + 2)
            idx = list(order[:n_top])
            while len(idx) < k:
                j = int(rng.integers(0, n))
                if j not in idx:
                    idx.append(j)
            idx = np.array(idx, dtype=int)
            # include contact neighbours of most overloaded point
            if rng.random() < 0.5:
                hub = int(order[0])
                neigh = np.where(G[hub] > m - 0.03)[0]
                for j in neigh:
                    if len(idx) >= k + 4:
                        break
                    if int(j) not in idx:
                        idx = np.append(idx, int(j))
            sigma = sigmas[(it + 3) % len(sigmas)] * 1.6
            trial[idx] = base[idx] + sigma * rng.standard_normal(size=(len(idx), d))
            nr = np.linalg.norm(trial, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            trial /= nr
        else:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            m = float(np.max(G))
            # worst k, bias to high rowmax
            k = int(rng.integers(1, 4))
            worst_idx = np.argsort(-rowmax)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst_idx] = False
            X_keep = base[keep_mask]
            for j in worst_idx:
                h = _hole_vector(X_keep, rng, n_cand=3500)
                h = h + 0.008 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                trial[j] = h
                X_keep = np.vstack([X_keep, h[None, :]])
            nr = np.linalg.norm(trial, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            trial /= nr
        # coarse relaxation
        trial = _coarse_relax(trial, steps=200, step0=0.03, decay=0.996)
        v_coarse = _max_cos(trial)
        # only LP-polish if promising (avoid wasting time)
        if v_coarse > best_val + 0.025 and v_coarse > cur_val + 0.015:
            v = v_coarse
            # accept/reject without LP
            if v < cur_val:
                current = trial.copy()
                cur_val = v
            continue
        budget_lp = 0.9
        # leave time for final polish
        if time.time() + budget_lp + 5.0 > t_end:
            budget_lp = max(0.25, t_end - 5.0 - time.time())
        if budget_lp < 0.2:
            break
        trial = _lp_polish(trial, budget_lp, t_end=t_end - 5.0)
        v = _max_cos(trial)
        if v < best_val - 1e-13:
            global_best = trial.copy()
            best_val = v
            current = trial.copy()
            cur_val = v
            if time.time() < t_end - 8.0:
                # extra squeeze on new record
                global_best = _lp_polish(global_best, min(1.2, t_end - 5.0 - time.time()), t_end=t_end - 5.0)
                best_val = _max_cos(global_best)
                current = global_best.copy()
                cur_val = best_val
        elif v < cur_val:
            current = trial.copy()
            cur_val = v
        else:
            Temp = 0.0015
            if rng.random() < np.exp((cur_val - v) / Temp):
                current = trial.copy()
                cur_val = v
        if it % 18 == 0:
            current = global_best.copy()
            cur_val = best_val
        # safety: if very little time left, break
        if time.time() >= t_end - 5.0:
            break
    # final polish
    if time.time() < t_end - 0.5:
        global_best = _lp_polish(global_best, max(0.5, t_end - time.time() - 0.3), t_end=t_end - 0.2)
    global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best

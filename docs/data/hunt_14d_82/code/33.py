import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _quench_soft(X, steps, lr0, lr1, b0, b1):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    p1, p2 = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    e0, e1 = np.log(b0), np.log(b1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        p1 *= b_1; p2 *= b_2
        m1 = b_1 * m1 + (1.0 - b_1) * Gt
        m2 = b_2 * m2 + (1.0 - b_2) * (Gt * Gt)
        mh = m1 / (1.0 - p1)
        vh = m2 / (1.0 - p2)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _quench_target(X, steps, lr0, lr1, T0, T1, pwr):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    p1, p2 = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        T = float(T0 + (T1 - T0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        V = G - T
        V[V < 0] = 0.0
        if pwr != 2.0:
            # avoid 0**0 issues
            W = np.power(V + 1e-16, pwr - 1.0)
            W[G - T < 0] = 0.0
        else:
            W = V
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        nrm = float(np.sqrt(np.mean(Gt * Gt)) + 1e-18)
        if nrm < 1e-18 or not np.isfinite(nrm):
            continue
        # scale to keep Adam stable
        Gt = Gt / (nrm + 1e-12)
        p1 *= b_1; p2 *= b_2
        m1 = b_1 * m1 + (1.0 - b_1) * Gt
        m2 = b_2 * m2 + (1.0 - b_2) * (Gt * Gt)
        mh = m1 / (1.0 - p1)
        vh = m2 / (1.0 - p2)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_replace(base, kill, rng, n_try=3000):
    n, d = base.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[kill] = False
    cur_keep = base[keep_mask]
    out = base.copy()
    for j in kill:
        Z = rng.standard_normal((n_try, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ cur_keep.T
        m = S.max(axis=1)
        h = Z[int(np.argmin(m))]
        h = h + 0.005 * rng.standard_normal(d)
        h /= np.linalg.norm(h)
        out[j] = h
        cur_keep = np.vstack([cur_keep, h[None, :]])
    return out

def _lp_polish(x, deadline, radius_init=2e-3, cap=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    while radius > 1e-13:
        if time.time() >= deadline:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        # fix concat typo guard
        # vals built above incorrectly if shape issue; rebuild correctly
        try:
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        except Exception:
            radius *= 0.5
            continue
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = float(deadline - time.time())
        if tl <= 0.02:
            break
        tl = min(1.0, tl)
        try:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": tl})
            except Exception:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
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
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        xb = rng.standard_normal((n, d))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    else:
        cands.sort(key=lambda c: _max_cos(c))
        xb = cands[0].copy()
    best_val = _max_cos(xb)
    global_best = xb.copy()
    current = xb.copy()
    cur_val = best_val

    it = 0
    # main hopping loop, reserve ~10s for final polish
    while time.time() < t_end - 10.0:
        it += 1
        if time.time() >= t_end - 10.0:
            break
        # choose base
        if it % 6 == 0:
            base = global_best
        elif rng.random() < 0.65:
            base = global_best
        else:
            base = current
        mode = it % 8
        trial = base.copy()
        if mode in (0, 1, 2):
            sig = [0.012, 0.05, 0.13][mode]
            trial = base + sig * rng.standard_normal(size=base.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 3:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = np.sum(G > m - 0.01, axis=1)
            order = np.argsort(-deg)
            k = int(rng.integers(4, 11))
            idx = list(order[:max(1, k // 2)])
            while len(idx) < k:
                j = int(rng.integers(0, n))
                if j not in idx:
                    idx.append(j)
            idx = np.array(idx, dtype=int)
            sig = float(rng.uniform(0.06, 0.16))
            trial[idx] = base[idx] + sig * rng.standard_normal(size=(len(idx), d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode in (4, 5):
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = int(rng.integers(1, 3)) if mode == 4 else int(rng.integers(3, 7))
            worst = np.argsort(-rowmax)[:k]
            trial = _hole_replace(base, worst, rng, n_try=3000)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 6:
            k = int(rng.integers(8, 14))
            idx = rng.choice(n, size=k, replace=False)
            trial[idx] = rng.standard_normal(size=(k, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # fresh random restart with long quench potential
            trial = rng.standard_normal(size=(n, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)

        # choose quench
        steps = int(rng.integers(250, 650))
        if mode == 7:
            steps = int(rng.integers(500, 900))
        r = rng.random()
        if r < 0.55:
            lr0 = float(rng.uniform(0.008, 0.03))
            lr1 = float(rng.uniform(0.0005, 0.003))
            b_start = float(rng.uniform(8.0, 30.0))
            b_end = float(rng.uniform(120.0, 400.0))
            trial = _quench_soft(trial, steps, lr0, lr1, b_start, b_end)
        else:
            lr0 = float(rng.uniform(0.008, 0.025))
            lr1 = float(rng.uniform(0.0008, 0.004))
            delta = float(rng.uniform(0.001, 0.009))
            Tt = best_val - delta
            # anneal target upward slightly
            T0 = Tt - float(rng.uniform(0.0, 0.004))
            T1 = Tt
            pwr = float(rng.choice([2.0, 3.0, 4.0]))
            trial = _quench_target(trial, steps, lr0, lr1, T0, T1, pwr)

        v0 = _max_cos(trial)
        # selective LP polish only if promising
        v = v0
        if v0 < best_val + 0.015:
            remain = (t_end - 10.0) - time.time()
            if remain > 0.8:
                dl = min(time.time() + 0.9, t_end - 10.0)
                # smaller radius if already very good
                rinit = 1.5e-3 if v0 > best_val - 0.001 else 8e-4
                trial = _lp_polish(trial, dl, radius_init=rinit)
                v = _max_cos(trial)
        # acceptance
        if v < best_val - 1e-13:
            global_best = trial.copy()
            best_val = v
            current = trial.copy()
            cur_val = v
        elif v < cur_val:
            current = trial.copy()
            cur_val = v
        else:
            Temp = 0.0015
            if rng.random() < np.exp((cur_val - v) / Temp):
                current = trial.copy()
                cur_val = v
        if it % 20 == 0:
            current = global_best.copy()
            cur_val = best_val

    # final staged polish of incumbent
    try:
        if time.time() < t_end - 6.0:
            global_best = _lp_polish(global_best, min(t_end - 2.0, time.time() + 3.0), radius_init=1.2e-3)
        if time.time() < t_end - 2.0:
            global_best = _lp_polish(global_best, t_end - 0.5, radius_init=2e-4)
    except Exception:
        pass
    global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best

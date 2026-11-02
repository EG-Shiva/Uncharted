import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _quench(X, steps, lr0, lr1, b0, b1, rng=None):
    # annealed LogSumExp Adam quench, X assumed normalized
    X = X.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    p1, p2 = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    e0, e1 = np.log(b0), np.log(b1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        # weights focused near max
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        p1 *= b_1; p2 *= b_2
        m1 = b_1 * m1 + (1.0 - b_1) * Gt
        m2 = b_2 * m2 + (1.0 - b_2) * (Gt * Gt)
        mh = m1 / (1.0 - p1); vh = m2 / (1.0 - p2)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        X = _norm(X)
    return X

def _hole_replace(fixed, rng, n_try=4000):
    d = fixed.shape[1]
    Z = rng.standard_normal((n_try, d))
    Z = _norm(Z)
    S = Z @ fixed.T
    m = S.max(axis=1)
    return Z[int(np.argmin(m))]

def _lp_polish(X, deadline, radius_init=1.2e-3, cap=2200, max_it=12):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_it and radius > 1e-14:
        it += 1
        if time.time() >= deadline:
            break
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 2e-7)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline_final = t0 + BUDGET - 12.0
    rng = np.random.default_rng(7)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        xb = _norm(rng.standard_normal((n, d)))
    else:
        cands.sort(key=lambda c: _max_cos(c))
        xb = cands[0].copy()
    best_val = _max_cos(xb)
    X_best = xb.copy()
    # initial joint quench to start jamming the plateau
    if time.time() < t0 + BUDGET - 30:
        try:
            Xq = _quench(X_best, 600, 0.02, 0.002, 800.0, 12000.0)
            vq = _max_cos(Xq)
            if vq < best_val:
                X_best, best_val = Xq, vq
        except Exception:
            pass
    cur = X_best.copy()
    cur_val = best_val
    # temperature for basin hopping acceptance
    Temp = 8e-5
    it = 0
    while time.time() < deadline_final:
        it += 1
        time_left = deadline_final - time.time()
        if time_left < 1.0:
            break
        mode = it % 5
        prop = cur.copy()
        rm, _ = _row_max(prop)
        order = np.argsort(-rm)  # worst first
        if mode == 0:
            sig = 0.012
            prop = prop + sig * rng.standard_normal((n, d))
            prop = _norm(prop)
            steps = 350
            b0, b1 = 1500.0, 10000.0
            lr0, lr1 = 0.015, 0.0015
        elif mode == 1:
            k = int(rng.integers(3, 6))
            victims = order[:k]
            for v in victims:
                prop[v] = prop[v] + 0.09 * rng.standard_normal(d)
            prop = _norm(prop)
            steps = 400
            b0, b1 = 800.0, 12000.0
            lr0, lr1 = 0.018, 0.0012
        elif mode == 2:
            k = 2 if (it % 2 == 0) else 3
            victims = order[:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            fixed = prop[keep_mask]
            for v in victims:
                prop[v] = _hole_replace(fixed, rng, n_try=4000)
                # update fixed for next victim
                fixed = np.vstack([fixed, prop[v][None, :]])
            steps = 380
            b0, b1 = 1000.0, 14000.0
            lr0, lr1 = 0.016, 0.0012
        elif mode == 3:
            sig = 0.045
            prop = prop + sig * rng.standard_normal((n, d))
            prop = _norm(prop)
            steps = 450
            b0, b1 = 500.0, 8000.0
            lr0, lr1 = 0.02, 0.002
        else:
            k = int(rng.integers(1, 3))
            victims = rng.choice(n, size=k, replace=False)
            for v in victims:
                prop[v] = _norm(rng.standard_normal((1, d)))[0]
            steps = 400
            b0, b1 = 700.0, 10000.0
            lr0, lr1 = 0.018, 0.0015
        # adapt steps to remaining time
        if time_left < 8:
            steps = min(steps, 180)
        try:
            prop = _quench(prop, steps, lr0, lr1, b0, b1)
        except Exception:
            continue
        v = _max_cos(prop)
        # update global best
        if v < best_val - 1e-12:
            X_best, best_val = prop.copy(), v
            cur, cur_val = prop.copy(), v
            Temp = max(Temp * 0.97, 1e-5)
        else:
            # Metropolis move on cur to change basin
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-9)):
                cur, cur_val = prop.copy(), v
            # slow cooling, reheat occasionally
            Temp *= 0.998
            if it % 40 == 0:
                Temp = min(Temp * 3.0, 1.2e-4)
                # restart cur from best occasionally to not drift too far
                if rng.random() < 0.5:
                    cur, cur_val = X_best.copy(), best_val
        # early exit if time tight
        if time.time() >= deadline_final:
            break
    # final squeeze: quench + LP
    try:
        if time.time() < t0 + BUDGET - 11:
            X_best = _quench(X_best, 500, 0.008, 0.0008, 3000.0, 25000.0)
    except Exception:
        pass
    try:
        X_best = _lp_polish(X_best, deadline=t0 + BUDGET - 1.0, radius_init=8e-4, cap=2200, max_it=12)
    except Exception:
        pass
    X_best = _norm(X_best)
    return X_best

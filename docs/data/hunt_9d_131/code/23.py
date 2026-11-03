import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _degrees(x, m, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    C = g >= (m - tol)
    deg = C.sum(axis=1)
    per_max = g.max(axis=1)
    return deg, per_max, g

def _smooth_refine(x, steps, T0, T1, lr0, lr1, mom=0.82):
    n, d = x.shape
    v = np.zeros_like(x)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights for near-max pairs
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W @ x
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gr = S - dot * x
        s = float(W.sum()) / n + 1e-12
        Gr = Gr / s
        v = mom * v + (1.0 - mom) * Gr
        x = x - lr * v
        x = _norm(x)
    return x

def _find_hole(Xk, rng, n_samples=15000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=150, T=0.008, lr=0.025):
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W = W / (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
    return z

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep_n = np.repeat(np.arange(n), d)
    rng_n = np.arange(n * d)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (rep_n, rng_n)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.3:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 174.0
    HOP_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    n, d = 131, 9
    best_x = None
    best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_s:
                    best_s = s
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_s = _max_cos(best_x)
    x_best = best_x.copy()
    s_best = best_s
    # initial squeeze of starting point briefly to ensure LP optimum
    try:
        x_best = _lp_polish(x_best, t0 + 8.0)
        s_best = _max_cos(x_best)
    except Exception:
        pass

    trial = 0
    # precompute probs refresh
    while time.time() < HOP_END:
        trial += 1
        # contact stats of current best (recompute every 5 trials to save time)
        if trial % 5 == 1:
            deg, per_max, _ = _degrees(x_best, s_best, tol=5e-7)
            # weights favour hubs
            w = (deg.astype(float) + 1.0) ** 1.5
            probs = w / w.sum()
            rattlers = np.where(deg == 0)[0]
            hubs = np.argsort(-deg)
        mode = trial % 10
        Y = x_best.copy()
        try:
            if mode == 0:
                Y = Y + 0.006 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth_refine(Y, 280, 0.018, 0.004, 0.016, 0.003)
            elif mode == 1:
                Y = Y + 0.016 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth_refine(Y, 300, 0.015, 0.003, 0.014, 0.0025)
            elif mode == 2:
                k = 1
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += 0.12 * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth_refine(Y, 320, 0.012, 0.0025, 0.013, 0.002)
            elif mode == 3:
                k = 2
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += 0.14 * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth_refine(Y, 320, 0.012, 0.002, 0.012, 0.002)
            elif mode == 4:
                # hole surgery: delete worst hub, insert optimized hole
                victim = int(hubs[0])
                Xk = np.delete(Y, victim, axis=0)
                z, _ = _find_hole(Xk, rng, n_samples=12000)
                z = _refine_hole(z, Xk, steps=120, T=0.008, lr=0.025)
                Y[victim] = z
                Y = _norm(Y)
                Y = _smooth_refine(Y, 340, 0.014, 0.002, 0.012, 0.002)
            elif mode == 5:
                # rattler reshuffle + hub nudge
                if len(rattlers) > 0:
                    k = min(len(rattlers), 3)
                    idx = rng.choice(rattlers, k, replace=False)
                    # reposition rattlers to random holes w.r.t. rest
                    for vi in idx:
                        Xk = np.delete(Y, vi, axis=0)
                        z, _ = _find_hole(Xk, rng, n_samples=8000)
                        z = _refine_hole(z, Xk, steps=80, T=0.010, lr=0.03)
                        Y[vi] = z
                # also nudge top hub slightly
                victim = int(hubs[rng.integers(0, min(5, n))])
                Y[victim] += 0.06 * rng.standard_normal(d)
                Y = _norm(Y)
                Y = _smooth_refine(Y, 300, 0.012, 0.002, 0.012, 0.002)
            elif mode == 6:
                # double hole surgery
                victims = hubs[:2].copy()
                Xk = np.delete(Y, victims, axis=0)
                for vi in victims:
                    z, _ = _find_hole(Xk, rng, n_samples=10000)
                    z = _refine_hole(z, Xk, steps=100, T=0.008, lr=0.025)
                    Xk = np.vstack([Xk, z[None, :]])
                # Xk is 131 again but order changed; map back: keep non-victims + new
                mask = np.ones(n, dtype=bool)
                mask[victims] = False
                Y[mask] = Xk[:n - 2]
                Y[victims[0]] = Xk[n - 2]
                Y[victims[1]] = Xk[n - 1]
                Y = _norm(Y)
                Y = _smooth_refine(Y, 340, 0.014, 0.002, 0.012, 0.002)
            elif mode == 7:
                Y = Y + 0.030 * rng.standard_normal(Y.shape)
                # one strong hub kick
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.20 * rng.standard_normal((1, d))
                Y = _norm(Y)
                Y = _smooth_refine(Y, 300, 0.018, 0.003, 0.015, 0.0025)
                Y = _smooth_refine(Y, 200, 0.005, 0.0015, 0.005, 0.0012)
            elif mode == 8:
                k = 3
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += 0.18 * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth_refine(Y, 320, 0.012, 0.002, 0.012, 0.002)
            else:
                Y = Y + 0.010 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth_refine(Y, 240, 0.010, 0.002, 0.010, 0.002)
        except Exception:
            continue
        try:
            s_sm = _max_cos(Y)
        except Exception:
            continue
        # promising? allow small slack for LP gain
        if s_sm < s_best + 0.0009:
            # quick LP squeeze
            dl = min(time.time() + 0.7, HOP_END + 2.0)
            if dl - time.time() < 0.2:
                continue
            try:
                Yp = _lp_polish(Y, dl)
                sp = _max_cos(Yp)
            except Exception:
                continue
            if sp < s_best - 1e-12:
                x_best = Yp
                s_best = sp
                deg, per_max, _ = _degrees(x_best, s_best, tol=5e-7)
                w = (deg.astype(float) + 1.0) ** 1.5
                probs = w / w.sum()
                rattlers = np.where(deg == 0)[0]
                hubs = np.argsort(-deg)
        # timeout guard
        if time.time() > HOP_END:
            break
    # final thorough polish with remaining time
    try:
        fin_dl = t0 + BUDGET - 1.0
        x_best = _lp_polish(x_best, fin_dl)
        s_best = _max_cos(x_best)
        # extra fine smooth + lp alternation if time remains
        if fin_dl - time.time() > 6.0:
            x_best = _smooth_refine(x_best, 400, 0.004, 0.001, 0.004, 0.001)
            x_best = _lp_polish(x_best, fin_dl)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

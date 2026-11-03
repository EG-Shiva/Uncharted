import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _smooth_refine(x, steps, T0, T1, lr0, lr1, momentum=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # stable weights
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(np.sum(W)) / n + 1e-12
        grad = grad / s
        v = momentum * v + (1.0 - momentum) * grad
        x = x - lr * v
        x = _norm(x)
    return x


def _find_hole(Xk, rng, n_samples=12000):
    # largest hole w.r.t fixed set Xk (Nk,d): minimize max dot
    d = Xk.shape[1]
    Z = rng.randn(n_samples, d)
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])


def _refine_hole(z, Xk, steps=150, T=0.01, lr=0.02):
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


def _lp_polish(x, seconds, t0, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < deadline and (time.time() - t0) < seconds:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        # cap size for speed
        if int(sel.sum()) > 4000:
            # keep tightest 4000
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
    return x


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(0)
    n_target, d_target = 131, 9
    # load start
    best_x = None
    best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n_target, d_target):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_s:
                    best_s = s
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.randn(n_target, d_target))
        best_s = _max_cos(best_x)
    x_best = best_x.copy()
    score_best = best_s

    trial = 0
    # phase 1: basin hopping until ~125s
    while time.time() - t0 < 122.0:
        trial += 1
        mode = trial % 5
        try:
            if mode == 0 or mode == 1:
                sigma = 0.015 if mode == 0 else 0.06
                if trial % 23 == 0:
                    sigma = 0.25
                elif trial % 11 == 0:
                    sigma = 0.12
                x_try = x_best + sigma * rng.randn(*x_best.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, steps=250, T0=0.02, T1=0.007, lr0=0.015, lr1=0.003)
                x_try = _smooth_refine(x_try, steps=250, T0=0.007, T1=0.002, lr0=0.005, lr1=0.0012)
            elif mode == 2 or mode == 3:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                cand = order[:6]
                idx = int(cand[rng.randint(len(cand))])
                mask = np.ones(len(x_best), dtype=bool)
                mask[idx] = False
                Xk = x_best[mask]
                z, _ = _find_hole(Xk, rng, n_samples=12000)
                z = _refine_hole(z, Xk, steps=150, T=0.01, lr=0.02)
                x_try = x_best.copy()
                x_try[idx] = z
                # tiny jitter of others to break symmetry
                x_try = x_try + 0.004 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, steps=300, T0=0.015, T1=0.005, lr0=0.012, lr1=0.002)
                x_try = _smooth_refine(x_try, steps=250, T0=0.005, T1=0.002, lr0=0.004, lr1=0.001)
            else:
                # shake worst 20
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                worst = order[:20]
                x_try = x_best.copy()
                x_try[worst] = x_try[worst] + 0.09 * rng.randn(len(worst), x_try.shape[1])
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, steps=250, T0=0.018, T1=0.006, lr0=0.013, lr1=0.0025)
                x_try = _smooth_refine(x_try, steps=250, T0=0.006, T1=0.002, lr0=0.004, lr1=0.001)
            s = _max_cos(x_try)
            if s < score_best:
                score_best = s
                x_best = x_try.copy()
        except Exception:
            continue
        if time.time() - t0 > BUDGET - 5:
            break

    # phase 2: LP polish with remaining time
    remain = BUDGET - (time.time() - t0)
    if remain > 8.0:
        try:
            # deadline relative to t0
            x_best = _lp_polish(x_best, seconds=remain - 1.0, t0=time.time(), deadline=remain - 1.0)
        except Exception:
            pass
    x_best = _norm(x_best)
    # sanity: no nan
    if not np.all(np.isfinite(x_best)):
        x_best = _norm(best_x)
    return x_best

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

def _find_hole(Xk, rng, n_samples=15000):
    d = Xk.shape[1]
    Z = rng.randn(n_samples, d)
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=120, T=0.008, lr=0.02):
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
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
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
        if deadline - time.time() < 0.5:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(7)
    n_target, d_target = 131, 9
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

    def quick_lp(x, secs):
        dl = time.time() + secs
        try:
            return _lp_polish(x, dl)
        except Exception:
            return x

    trial = 0
    HOP_END = t0 + 128.0
    while time.time() < HOP_END:
        trial += 1
        mode = trial % 7
        try:
            if mode == 0:
                sigma = 0.012 if (trial % 3) else 0.03
                if trial % 29 == 0:
                    sigma = 0.22
                x_try = x_best + sigma * rng.randn(*x_best.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, 220, 0.02, 0.006, 0.014, 0.0025)
                x_try = _smooth_refine(x_try, 220, 0.006, 0.0018, 0.004, 0.001)
            elif mode == 1 or mode == 2:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                cand = order[:8]
                idx = int(cand[rng.randint(len(cand))])
                mask = np.ones(len(x_best), dtype=bool)
                mask[idx] = False
                Xk = x_best[mask]
                z, _ = _find_hole(Xk, rng, n_samples=15000)
                z = _refine_hole(z, Xk, steps=130, T=0.008, lr=0.02)
                x_try = x_best.copy()
                x_try[idx] = z
                x_try = x_try + 0.004 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, 260, 0.014, 0.004, 0.011, 0.0018)
                x_try = _smooth_refine(x_try, 220, 0.004, 0.0015, 0.0035, 0.0009)
            elif mode == 3:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                k2 = 2 if (trial % 2 == 0) else 3
                if trial % 17 == 0:
                    k2 = 4
                idxs = order[:12]
                sel_idx = rng.choice(idxs, size=k2, replace=False)
                mask = np.ones(len(x_best), dtype=bool)
                mask[sel_idx] = False
                Xk = x_best[mask]
                new_pts = []
                cur = Xk
                for _ in range(k2):
                    z, _ = _find_hole(cur, rng, n_samples=12000)
                    z = _refine_hole(z, cur, steps=110, T=0.009, lr=0.022)
                    new_pts.append(z)
                    cur = np.vstack([cur, z[None, :]])
                x_try = x_best.copy()
                for ii, zz in zip(sel_idx, new_pts):
                    x_try[int(ii)] = zz
                # diversify order
                rng.shuffle(sel_idx)
                x_try = x_try + 0.005 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, 280, 0.016, 0.005, 0.012, 0.002)
                x_try = _smooth_refine(x_try, 240, 0.005, 0.0015, 0.004, 0.001)
            elif mode == 4:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                pm = G.max(axis=1)
                ratt = np.where(pm < m - 5e-4)[0]
                x_try = x_best.copy()
                if len(ratt) >= 2:
                    nr = min(3, len(ratt))
                    pick = rng.choice(ratt, size=nr, replace=False)
                    for ii in pick:
                        zr = _norm(rng.randn(d_target))
                        x_try[int(ii)] = zr
                # also shake worst 12
                order = np.argsort(-pm)
                worst = order[:12]
                x_try[worst] = x_try[worst] + 0.07 * rng.randn(len(worst), d_target)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, 260, 0.018, 0.005, 0.012, 0.002)
                x_try = _smooth_refine(x_try, 220, 0.005, 0.0018, 0.004, 0.001)
            elif mode == 5:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                worst = order[:20]
                x_try = x_best.copy()
                x_try[worst] = x_try[worst] + 0.09 * rng.randn(len(worst), d_target)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, 240, 0.018, 0.006, 0.013, 0.0022)
                x_try = _smooth_refine(x_try, 220, 0.006, 0.0018, 0.004, 0.001)
            else:
                # large random-subset reset: replace 5 random points
                x_try = x_best.copy()
                rep = rng.choice(len(x_try), size=5, replace=False)
                for ii in rep:
                    x_try[int(ii)] = _norm(rng.randn(1, d_target))[0]
                x_try = x_try + 0.01 * rng.randn(*x_try.shape)
                x_try = _norm(x_try)
                x_try = _smooth_refine(x_try, 300, 0.022, 0.006, 0.014, 0.0022)
                x_try = _smooth_refine(x_try, 240, 0.006, 0.0018, 0.004, 0.001)
            s = _max_cos(x_try)
            if s < score_best:
                # squeeze immediately with short LP to get true basin value
                if time.time() < t0 + BUDGET - 25.0:
                    x_try2 = quick_lp(x_try, 1.2)
                    s2 = _max_cos(x_try2)
                    if s2 < s:
                        x_try, s = x_try2, s2
                if s < score_best:
                    score_best = s
                    x_best = x_try.copy()
        except Exception:
            continue
        if time.time() > t0 + BUDGET - 6.0:
            break

    remain_deadline = t0 + BUDGET - 1.0
    if time.time() < remain_deadline - 5.0:
        try:
            x_best = _lp_polish(x_best, remain_deadline - 4.0)
            # tiny low-T smooth to clean linearization error then re-polish
            x_best = _smooth_refine(x_best, 150, 0.0025, 0.001, 0.002, 0.0006, momentum=0.7)
            x_best = _lp_polish(x_best, remain_deadline)
        except Exception:
            pass
    x_best = _norm(x_best)
    if not np.all(np.isfinite(x_best)):
        x_best = _norm(best_x)
    return x_best

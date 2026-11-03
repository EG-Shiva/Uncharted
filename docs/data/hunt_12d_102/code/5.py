import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = float(c.max())
        thr = m - max(20 * radius, 1e-9)
        sel = c > thr
        if int(sel.sum()) == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns: I-block, J-block, t-column
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        ct = np.full(p, n * d)
        # interleave per row: need order matching rows repeat
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = ct if False else 0  # placeholder, rebuilt below
        # build correctly row-wise
        cols = np.concatenate(
            [np.concatenate([cI[i * d:(i + 1) * d], cJ[i * d:(i + 1) * d], [n * d]]) for i in range(p)]
        ) if False else None
        # simpler: build with tiling
        base = np.concatenate([np.arange(d), np.arange(d), [-1]])  # not used
        # explicit construction
        cols_arr = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_arr[:, :d] = I[:, None] * d + np.arange(d)
        cols_arr[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols_arr[:, 2 * d] = n * d
        cols = cols_arr.ravel()
        vals_arr = np.empty((p, 2 * d + 1))
        vals_arr[:, :d] = x[J]
        vals_arr[:, d:2 * d] = x[I]
        vals_arr[:, 2 * d] = -1.0
        vals = vals_arr.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (np.tile(x.ravel(), 1), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv),
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            break
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


def _smooth_adam(x, steps, T0, T1, lr0, lr1, deadline):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if (s & 63) == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = float(G.max())
        # softmax weights
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0:
            break
        W = E / S
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * (grad * grad)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((102, 12))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (102, 12):
        # try to reshape / fallback
        r = rng.standard_normal((102, 12))
        m = min(xb.size, r.size)
        r.ravel()[:m] = xb.ravel()[:m]
        xb = r
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=np.float64)
        if xr.shape == (102, 12):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            rv = _max_cos(xr)
            if rv < best_val:
                xb, best_val = xr, rv
    except Exception:
        pass

    # initial short polish of incumbent to lock local optimum
    xb = _lp_polish(xb, min(t0 + 8.0, HARD_END - 150.0))
    best_val = _max_cos(xb)
    x_best = xb.copy()

    cur = x_best.copy()
    cur_val = best_val
    n, d = 102, 12

    sigmas = [0.02, 0.05, 0.10, 0.03, 0.15, 0.008, 0.06]
    kreps = [0, 1, 2, 1, 3, 0, 1]

    cycle = 0
    # main basin-hopping loop, reserve ~14s for final polish
    while time.time() < HARD_END - 14.0:
        cycle += 1
        sigma = sigmas[cycle % len(sigmas)]
        krep = kreps[(cycle // 2) % len(kreps)]
        # kick current (not always best) to explore
        y = cur + sigma * rng.standard_normal((n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if krep > 0:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mv = float(G.max())
            deg = np.sum(G > mv - 0.002, axis=1)
            worst = np.argsort(-deg, kind="stable")[:krep]
            for wi in worst:
                # largest-hole candidate w.r.t. others
                nc = 3000
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                mask = np.ones(n, dtype=bool)
                mask[int(wi)] = False
                Xoth = y[mask]
                dots = cand @ Xoth.T
                holl = dots.max(axis=1)
                bi = int(np.argmin(holl))
                y[int(wi)] = cand[bi]
        # annealed smooth repulsion
        steps = 900 if (cycle % 3) else 1400
        T0, T1 = 0.02, 0.002
        lr0, lr1 = 0.015, 0.003
        # vary slightly per cycle for diversity (deterministic)
        if cycle % 4 == 1:
            T0, T1 = 0.03, 0.003
        if cycle % 4 == 3:
            lr0 = 0.025
        y2 = _smooth_adam(y, steps, T0, T1, lr0, lr1, HARD_END - 14.0)
        v2 = _max_cos(y2)
        y3, v3 = y2, v2
        # polish only promising candidates to save time
        if v2 < best_val + 0.004 and v2 < cur_val + 0.008:
            y3 = _lp_polish(y2, min(time.time() + 2.5, HARD_END - 14.0))
            v3 = _max_cos(y3)
        if v3 < best_val - 1e-12:
            x_best, best_val = y3.copy(), v3
            cur, cur_val = y3.copy(), v3
        else:
            # Metropolis walk on cur to change basin
            if v3 < cur_val or rng.random() < float(np.exp((cur_val - v3) / 0.002)):
                cur, cur_val = y3.copy(), v3
            # occasional restart from best
            if cycle % 11 == 0:
                cur, cur_val = x_best.copy(), best_val

    # final thorough polish of best
    x_best = _lp_polish(x_best, HARD_END - 1.0)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

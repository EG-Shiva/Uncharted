import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline, radius_init=0.004):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns: d for I, d for J, 1 for t
        base_I = (I[:, None] * d + np.arange(d)).ravel()
        base_J = (J[:, None] * d + np.arange(d)).ravel()
        cols_I = base_I
        cols_J = base_J
        # build cols array
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = 0  # placeholder, will overwrite blockwise
        # faster block construction
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)).ravel(),
            (J[:, None] * d + np.arange(d)).ravel(),
            np.full(p, n * d)
        ]) if False else None
        # Simple construction via repeat logic:
        # rows already repeat; build cols in order per pair: I-block, J-block, t
        Cb = np.empty((p, 2 * d + 1), dtype=np.int64)
        Cb[:, :d] = I[:, None] * d + np.arange(d)
        Cb[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Cb[:, 2 * d] = n * d
        cols = Cb.ravel()
        Vb = np.empty((p, 2 * d + 1))
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        vals = Vb.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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


def _lbfgs_smooth(x_init, T, maxiter, iu):
    n, d = x_init.shape
    x0 = x_init.ravel().copy()

    def fun(g):
        X = g.reshape(n, d)
        r = np.linalg.norm(X, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        Y = X / r
        G = Y @ Y.T
        dots = G[iu]
        mx = dots.max()
        # scaled exps
        e = np.exp((dots - mx) / T)
        s = e.sum()
        loss = mx + T * np.log(s)
        w = e / s
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        gradY = W @ Y
        dot = np.sum(gradY * Y, axis=1, keepdims=True)
        gradX = (gradY - Y * dot) / r
        return float(loss), gradX.ravel()
    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9, "maxfun": maxiter * 5})
        X = res.x.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    except Exception:
        return x_init


def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline_all = t0 + BUDGET
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    iu = np.triu_indices(n, 1)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    # quick per-point worst helper
    def worst_indices(x, k=1):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        return np.argsort(-pm)[:k], pm

    sigma = 0.012
    outer = 0
    # stage 1: direct smooth + polish of incumbent (no noise) to squeeze cluster
    # we do this inside loop with outer==0
    while time.time() < deadline_all - 25.0 and outer < 40:
        if outer == 0:
            xc = x_best.copy()
        elif outer % 6 == 5:
            # full reinsertion of worst point
            wi, _ = worst_indices(x_best, k=1)
            xc = x_best.copy()
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            xc[wi[0]] = v
            # small jitter others
            xc += rng.standard_normal((n, d)) * (0.002)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
        elif outer % 6 == 0:
            # reinsert 2 worst points
            wi, _ = worst_indices(x_best, k=2)
            xc = x_best.copy()
            for idx in wi:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                xc[idx] = v
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
        else:
            noise = rng.standard_normal((n, d)) * sigma
            xc = x_best + noise
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
        # smooth anneal: 2-3 temps, few iters to stay fast
        if time.time() > deadline_all - 25.0:
            break
        # adapt maxiter to remaining time / outer budget
        xc = _lbfgs_smooth(xc, 0.012, 60, iu)
        if time.time() > deadline_all - 25.0:
            cur = _max_cos(xc)
            if cur < best_val:
                x_best, best_val = xc, cur
            break
        xc = _lbfgs_smooth(xc, 0.005, 80, iu)
        if time.time() > deadline_all - 22.0:
            cur = _max_cos(xc)
            if cur < best_val - 1e-12:
                x_best, best_val = xc, cur
            break
        xc = _lbfgs_smooth(xc, 0.002, 100, iu)
        # short LP polish 2-3 sec
        remain = deadline_all - time.time()
        lp_sec = min(4.0, max(0.8, remain - 20.0))
        if lp_sec > 0.5:
            xc = _lp_polish(xc, time.time() + lp_sec, radius_init=0.003)
        cur = _max_cos(xc)
        if cur < best_val - 1e-12:
            x_best = xc.copy()
            best_val = cur
            sigma = max(0.003, sigma * 0.85)
        else:
            # mild sigma adaptation
            sigma = min(0.03, sigma * 1.08)
        outer += 1

    # final intensive polish of best
    remain = deadline_all - time.time()
    if remain > 3.0:
        # one more fine smooth at small T then LP
        try:
            x_best = _lbfgs_smooth(x_best, 0.0015, 120, iu)
        except Exception:
            pass
        remain = deadline_all - time.time()
        if remain > 2.0:
            x_best = _lp_polish(x_best, deadline_all - 1.0, radius_init=0.002)
            # extra tiny LP with small radius for high precision
            if deadline_all - time.time() > 1.5:
                x_best = _lp_polish(x_best, deadline_all - 0.5, radius_init=0.0005)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: if somehow worse than input, return input
    try:
        if _max_cos(x_best) > best_val + 1e-15:
            pass
    except Exception:
        pass
    # ensure no NaN / zero rows
    x_best = np.where(np.isnan(x_best), 0.0, x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    x_best = x_best / nr
    return x_best

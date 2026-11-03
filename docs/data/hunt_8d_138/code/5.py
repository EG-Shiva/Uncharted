import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, r_init=0.003):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time left check
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        delta = max(20.0 * radius, 1e-9)
        sel = c > m - delta
        # cap size for speed
        cnt_sel = int(sel.sum())
        if cnt_sel > 2500:
            # keep 2500 largest
            idx = np.argpartition(c, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            cnt_sel = 2500
        if cnt_sel == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        p = cnt_sel
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def _fast_polish(X, steps=1800, lr0=0.03, lr1=0.001,
                 d0=0.03, d1=0.003, mu=0.8):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    vel = np.zeros_like(X)
    for k in range(steps):
        f = k / max(steps - 1, 1)
        lr = lr0 * (1.0 - f) + lr1 * f
        delta = d0 * (1.0 - f) + d1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - delta
        M = (G > thr).astype(np.float64)
        # counts
        cnt = M.sum(axis=1)
        # gradient of active uniform repulsion (tangential)
        # grad_i = sum_j M_ij*(X_j - G_ij X_i)
        MX = M @ X
        wsum = (M * G).sum(axis=1)[:, None]
        grad = MX - X * wsum
        cntc = np.maximum(cnt[:, None], 1.0)
        grad /= cntc
        # momentum (EMA)
        vel = mu * vel + (1.0 - mu) * grad
        X = X - lr * vel
        # renormalize
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        # guard
        nn = np.maximum(nn, 1e-12)
        X /= nn
    return X


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 138, 8
    # load start
    best_init = None
    best_val_init = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val_init:
                    best_val_init = v
                    best_init = a.copy()
        except Exception:
            pass
    if best_init is None:
        best_init = rng.standard_normal(size=(n, d))
        best_init /= np.linalg.norm(best_init, axis=1, keepdims=True)
        best_val_init = _max_cos(best_init)
    global_best = best_init.copy()
    best_val = float(best_val_init)

    # basin hopping, reserve time for final LP
    # phase 1: explore until 135s
    TIME_EXPLORE = 135.0
    TIME_HARD = 175.0
    trial = 0
    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_EXPLORE:
            break
        if trial > 2000:
            break
        trial += 1
        # choose trial type
        if trial % 25 == 0:
            # fresh random start
            X_try = rng.standard_normal(size=(n, d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
            steps = 3000
            lr0, lr1 = 0.05, 0.001
            dd0, dd1 = 0.05, 0.004
        else:
            r = rng.random()
            if r < 0.5:
                sigma = 0.02
            elif r < 0.8:
                sigma = 0.05
            else:
                sigma = 0.12
            if rng.random() < 0.35:
                k = int(rng.integers(5, 30))
                idx = rng.choice(n, size=k, replace=False)
                X_try = global_best.copy()
                noise = rng.standard_normal(size=(k, d)) * (sigma * 2.0)
                X_try[idx] += noise
                nn = np.linalg.norm(X_try[idx], axis=1, keepdims=True)
                nn = np.maximum(nn, 1e-12)
                X_try[idx] /= nn
                steps = 1800
                lr0, lr1 = 0.03, 0.0008
                dd0, dd1 = 0.03, 0.003
            else:
                X_try = global_best + sigma * rng.standard_normal(size=(n, d))
                X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
                steps = 1800
                lr0, lr1 = 0.03, 0.0008
                dd0, dd1 = 0.03, 0.003
        X_fast = _fast_polish(X_try, steps=steps, lr0=lr0, lr1=lr1,
                              d0=dd0, d1=dd1, mu=0.8)
        val_fast = _max_cos(X_fast)
        elapsed2 = time.time() - t0
        if elapsed2 >= TIME_EXPLORE:
            # check if immediate win without LP
            if val_fast < best_val:
                global_best, best_val = X_fast, val_fast
            break
        if val_fast < best_val - 1e-12:
            # refine win
            sec = min(3.0, TIME_EXPLORE - (time.time() - t0))
            if sec > 0.6:
                X_ref = _lp_polish(X_fast, seconds=sec, r_init=0.003)
                vref = _max_cos(X_ref)
                if vref < best_val:
                    global_best, best_val = X_ref, vref
                else:
                    global_best, best_val = X_fast, val_fast
            else:
                global_best, best_val = X_fast, val_fast
        elif val_fast < best_val + 0.008:
            # promising -> short LP occasionally
            do_ref = (trial % 2 == 0) or (val_fast < best_val + 0.002)
            if do_ref:
                remain = TIME_EXPLORE - (time.time() - t0)
                if remain > 1.6:
                    X_ref = _lp_polish(X_fast, seconds=1.2, r_init=0.002)
                    vref = _max_cos(X_ref)
                    if vref < best_val - 1e-12:
                        global_best, best_val = X_ref, vref
        # else discard
    # final squeeze
    time_left = TIME_HARD - (time.time() - t0) - 1.0
    if time_left > 3.0:
        # cap to avoid overrun
        time_left = min(time_left, 40.0)
        global_best = _lp_polish(global_best, seconds=time_left, r_init=0.003)
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best

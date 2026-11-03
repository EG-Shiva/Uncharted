import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-4
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() > t_end - 1:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 5e-7)
        if np.count_nonzero(sel) > 6000:
            # keep only tightest 6000 to keep LP fast
            idx = np.argsort(c)[-6000:]
            msel = np.zeros_like(sel)
            msel[idx] = True
            sel = msel
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(1.0, t_end - time.time() - 0.5)})
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    np.random.seed(0)

    # load starting point
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (249, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = rng0.normal(size=(249, 6))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)

    n, d = best_x.shape
    iu0, iu1 = np.triu_indices(n, 1)

    cur = best_x.copy()
    cur_v = best_v

    def make_fun(alpha):
        def fun(yflat):
            Y = yflat.reshape(n, d)
            nrm = np.sqrt(np.sum(Y * Y, axis=1))
            nrm = np.maximum(nrm, 1e-12)
            X = Y / nrm[:, None]
            G = X @ X.T
            g = G[iu0, iu1]
            m = g.max()
            e = np.exp(alpha * (g - m))
            S = e.sum()
            f = m + np.log(S) / alpha
            w = e / S
            # W @ X without full matrix using accumulation
            # Build W dense (249x249) - cheap
            W = np.empty((n, n), dtype=float)
            W.fill(0.0)
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            Gx = W @ X
            s = np.sum(Gx * X, axis=1)
            Gy = (Gx - X * s[:, None]) / nrm[:, None]
            # tiny norm anchor to remove radial null space
            # penalty 1e-4 * mean((nrm-1)^2)
            diff = nrm - 1.0
            f = f + 1e-4 * np.mean(diff * diff)
            # grad of penalty: 1e-4*2*diff/n * X
            Gy = Gy + (X * (2e-4 * diff / n)[:, None])
            return float(f), Gy.ravel()
        return fun

    alphas = [60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 3500.0]
    maxiters = [200, 250, 300, 350, 400, 450, 500]

    for ai, alpha in enumerate(alphas):
        if time.time() > t_end - 25:
            break
        fun = make_fun(alpha)
        # budget per stage
        mi = maxiters[ai]
        try:
            res = minimize(fun, cur.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": mi, "maxfun": mi * 3, "ftol": 1e-14, "gtol": 1e-7, "maxcor": 10})
            y = res.x.reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            # keep best
            if v < best_v:
                best_v = v
                best_x = y.copy()
            # continue from best of cur/y to stay downhill
            if v < cur_v:
                cur = y
                cur_v = v
            else:
                # if stuck, continue from best anyway with small shake
                cur = best_x.copy()
                cur_v = best_v
        except Exception:
            continue

    # basin-hop kicks on top of best
    rng = np.random.default_rng(1)
    sigmas = [0.015, 0.008, 0.004, 0.008, 0.003]
    for k, sig in enumerate(sigmas):
        if time.time() > t_end - 22:
            break
        alpha = 1500.0 if k % 2 == 0 else 2500.0
        trial = best_x + sig * rng.standard_normal(size=(n, d))
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        fun = make_fun(alpha)
        try:
            res = minimize(fun, trial.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": 350, "maxfun": 1100, "ftol": 1e-14, "gtol": 1e-7, "maxcor": 10})
            y = res.x.reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best_v:
                best_v = v
                best_x = y.copy()
                cur = y.copy()
                cur_v = v
        except Exception:
            continue

    # final intensive smoothing at high alpha from best
    if time.time() < t_end - 15:
        try:
            fun = make_fun(4000.0)
            res = minimize(fun, best_x.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": 400, "maxfun": 1200, "ftol": 1e-15, "gtol": 1e-8, "maxcor": 10})
            y = res.x.reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best_v:
                best_v = v
                best_x = y.copy()
        except Exception:
            pass

    # final LP jam if time remains
    if time.time() < t_end - 4:
        try:
            y = _lp_polish(best_x, 30, t_end - 1.0)
            v = _max_cos(y)
            if v < best_v:
                best_v = v
                best_x = y
        except Exception:
            pass

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

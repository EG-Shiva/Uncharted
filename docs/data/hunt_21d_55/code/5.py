import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t_start=None, hard_limit=None):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_start is not None and hard_limit is not None:
            if time.time() - t_start >= hard_limit:
                break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD = 168.0
    FINAL_POLISH = 25.0
    rng = np.random.default_rng(0)

    n, d = 55, 21
    best_X = None
    best_val = 1e9

    def try_load(path):
        try:
            a = np.load(path)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1)
                if np.all(nr > 1e-8):
                    return _norm_rows(a)
        except Exception:
            pass
        return None

    xb = try_load("best.npy")
    xr = try_load("record.npy")
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        best_X = _norm_rows(rng.standard_normal((n, d)))
        best_val = _max_cos(best_X)
    else:
        for c in cands:
            v = _max_cos(c)
            if v < best_val:
                best_val = v
                best_X = c.copy()

    iu0 = np.triu_indices(n, 1)

    def smooth_refine(X_init, Ts, maxiter_each):
        Z0 = X_init.ravel().copy()

        def make_fun(T):
            def fun(Zflat):
                Z = Zflat.reshape(n, d)
                norms = np.linalg.norm(Z, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X = Z / norms
                G = X @ X.T
                # max off-diag for shift (treated as constant)
                Gmax = float(np.max(G - 2.0 * np.eye(n)))
                Emat = np.exp((G - Gmax) / T)
                np.fill_diagonal(Emat, 0.0)
                S = float(Emat.sum() * 0.5)
                if S <= 0 or not np.isfinite(S):
                    return 1e6, np.zeros_like(Zflat)
                E = Gmax + T * np.log(S)
                W = Emat / S
                Grad_X = W @ X
                dot = np.sum(Grad_X * X, axis=1, keepdims=True)
                Grad_Z = (Grad_X - dot * X) / norms
                return float(E), Grad_Z.ravel()
            return fun

        Z = Z0
        for T in Ts:
            fun = make_fun(float(T))
            try:
                res = minimize(fun, Z, method="L-BFGS-B", jac=True,
                               options={"maxiter": int(maxiter_each),
                                        "ftol": 1e-12, "gtol": 1e-7,
                                        "maxcor": 10, "maxfun": int(maxiter_each * 8)})
                Z = res.x
            except Exception:
                break
            if t0 is not None and (time.time() - t0) >= HARD - 2:
                break
        X = _norm_rows(Z.reshape(n, d))
        return X

    def row_maxima(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    it = 0
    # time for exploration
    while (time.time() - t0) < (HARD - FINAL_POLISH) and it < 300:
        mode = it % 6
        base = best_X
        rmax = row_maxima(base)
        order = np.argsort(-rmax)
        if mode == 0:
            sigma = 0.008
            Xp = _norm_rows(base + sigma * rng.standard_normal((n, d)))
            Ts = [0.008, 0.003]
            mi = 200
        elif mode == 1:
            sigma = 0.03
            Xp = _norm_rows(base + sigma * rng.standard_normal((n, d)))
            Ts = [0.012, 0.004]
            mi = 250
        elif mode == 2:
            sigma = 0.10
            Xp = _norm_rows(base + sigma * rng.standard_normal((n, d)))
            Ts = [0.015, 0.005]
            mi = 250
        elif mode == 3:
            # reinit single worst point
            Xp = base + 0.015 * rng.standard_normal((n, d))
            idx = int(order[0]) if it % 2 == 0 else int(rng.integers(n))
            Xp[idx] = rng.standard_normal(d)
            Xp = _norm_rows(Xp)
            Ts = [0.012, 0.004]
            mi = 250
        elif mode == 4:
            # reinit two points: one worst + one random
            Xp = base + 0.02 * rng.standard_normal((n, d))
            idx1 = int(order[int(rng.integers(min(10, n)))])
            idx2 = int(rng.integers(n))
            Xp[idx1] = rng.standard_normal(d)
            Xp[idx2] = rng.standard_normal(d)
            Xp = _norm_rows(Xp)
            Ts = [0.015, 0.005]
            mi = 300
        else:
            sigma = 0.18
            Xp = _norm_rows(base + sigma * rng.standard_normal((n, d)))
            Ts = [0.02, 0.006]
            mi = 300

        try:
            Xr = smooth_refine(Xp, Ts, mi)
        except Exception:
            it += 1
            continue
        v = _max_cos(Xr)
        if v < best_val:
            best_val = v
            best_X = Xr.copy()
        it += 1
        # quick opportunistic tiny LP squeeze if very promising? skip to save time

    # final trust-region LP polish with remaining time
    remain = HARD - (time.time() - t0)
    if remain > 3:
        sec = min(FINAL_POLISH, remain - 1.0)
        try:
            best_X = _lp_polish(best_X, sec, t_start=t0, hard_limit=HARD)
        except Exception:
            pass
    best_X = _norm_rows(best_X)
    return best_X

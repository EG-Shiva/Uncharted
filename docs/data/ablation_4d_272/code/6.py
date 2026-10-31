import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep_n = np.repeat(np.arange(n), d)
    col_n = np.arange(n * d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (rep_n, col_n)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn = np.maximum(yn, 1e-12)
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def lse_opt(x0, beta, maxiter):
    n, d = x0.shape
    iu = np.triu_indices(n, 1)
    Y0 = x0.ravel().copy()

    def fun(Yflat):
        Y = Yflat.reshape(n, d)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = Y / norms
        G = X @ X.T
        vals = G[iu]
        M = float(np.max(vals))
        E = np.exp(beta * (vals - M))
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            return float(M + 1.0), np.zeros_like(Yflat)
        L = M + np.log(S) / beta
        W = np.zeros((n, n), dtype=np.float64)
        W[iu] = E
        W = W + W.T
        GX = (W @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GXt = GX - dot * X
        GY = GXt / norms
        return float(L), GY.ravel()

    try:
        res = minimize(fun, Y0, method="L-BFGS-B", jac=True,
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 10000, "maxcor": 10})
        Yf = res.x.reshape(n, d)
    except Exception:
        return x0
    norms = np.linalg.norm(Yf, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xf = Yf / norms
    if not np.all(np.isfinite(Xf)):
        return x0
    return Xf


def solve():
    np.random.seed(0)
    t0 = time.time()
    HARD = 172.0
    LOOP_END = 150.0

    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = xb
    xb = np.asarray(xb, dtype=np.float64)
    xr = np.asarray(xr, dtype=np.float64)
    xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    xr = xr / np.maximum(np.linalg.norm(xr, axis=1, keepdims=True), 1e-12)
    vb = max_cos(xb)
    vr = max_cos(xr)
    if vr < vb:
        best = xr.copy()
        best_val = vr
    else:
        best = xb.copy()
        best_val = vb
    n, d = best.shape

    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12]
    betas = [30, 50, 80, 120]
    trial = 0
    # quick initial squeeze in case record not fully polished
    # (cheap, improves baseline before hopping)
    if time.time() - t0 < LOOP_END:
        pass

    while time.time() - t0 < LOOP_END:
        trial += 1
        sigma = float(np.random.choice(sigmas))
        beta = int(np.random.choice(betas))
        # perturb best
        noise = np.random.randn(n, d)
        x_try = best + sigma * noise
        # rattler-targeted replacement
        if np.random.rand() < 0.40:
            try:
                G0 = best @ best.T
                np.fill_diagonal(G0, -2.0)
                rowmax = G0.max(axis=1)
                ratt = np.where(rowmax < best_val - 0.01)[0]
                if len(ratt) > 0:
                    k = int(min(len(ratt), np.random.randint(1, 4)))
                    idx = np.random.choice(ratt, k, replace=False)
                else:
                    k = int(np.random.randint(1, 4))
                    idx = np.random.choice(n, k, replace=False)
                rnd = np.random.randn(k, d)
                rnd = rnd / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
                x_try[idx] = rnd + 0.2 * np.random.randn(k, d)
            except Exception:
                pass
        x_try = x_try / np.maximum(np.linalg.norm(x_try, axis=1, keepdims=True), 1e-12)

        use_lbfgs = (np.random.rand() < 0.75)
        if use_lbfgs:
            maxiter = 50 if sigma < 0.03 else 80
            # guard against overtime: skip if too close to end
            if time.time() - t0 > LOOP_END - 2.0:
                break
            try:
                x_opt = lse_opt(x_try, beta, maxiter)
            except Exception:
                continue
            x_opt = x_opt / np.maximum(np.linalg.norm(x_opt, axis=1, keepdims=True), 1e-12)
            v0 = max_cos(x_opt)
            if v0 > best_val + 0.015:
                continue
            if time.time() - t0 >= LOOP_END:
                break
            lp_time = 1.2 if v0 < best_val else 0.6
            # cap by remaining loop time
            rem = LOOP_END - (time.time() - t0)
            lp_time = min(lp_time, max(0.2, rem - 0.1))
            try:
                x_pol = lp_polish(x_opt, lp_time)
            except Exception:
                continue
            v = max_cos(x_pol)
            if v < best_val - 1e-12 and np.isfinite(v):
                best = x_pol
                best_val = v
        else:
            # direct LP hop (no smoothing) for fast basin change
            try:
                v_try = max_cos(x_try)
            except Exception:
                continue
            if v_try > best_val + 0.03:
                continue
            rem = LOOP_END - (time.time() - t0)
            lp_time = min(0.8, max(0.2, rem - 0.1))
            try:
                x_pol = lp_polish(x_try, lp_time)
            except Exception:
                continue
            v = max_cos(x_pol)
            if v < best_val - 1e-12 and np.isfinite(v):
                best = x_pol
                best_val = v
        if time.time() - t0 >= HARD - 5:
            break

    remain = HARD - (time.time() - t0)
    if remain > 2.5:
        try:
            best = lp_polish(best, min(remain - 1.0, 20.0))
        except Exception:
            pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    # final safety: replace any non-finite row
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rnd = np.random.randn(int(np.sum(bad)), d)
        rnd = rnd / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
        best[bad] = rnd
    return best

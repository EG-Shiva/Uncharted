import time
import numpy as np
from scipy.optimize import minimize, linprog
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
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        # cap size for speed
        if np.count_nonzero(sel) > 2500:
            thr = np.sort(c)[-2500]
            sel = c >= thr
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def lbfgs_smooth(x_init, tau, maxiter):
    n, d = x_init.shape
    iu0 = np.triu_indices(n, 1)

    def fun(y_flat):
        Y = y_flat.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        g = G[iu0]
        mmax = g.max()
        e = np.exp((g - mmax) / tau)
        S = e.sum()
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        # build symmetric weight matrix action: Svec = W @ X
        # W has w on off-diagonals
        # efficient: use accumulation
        Wx = np.zeros_like(X)
        # scatter: for each pair k: Wx[I]+=w*x[J], Wx[J]+=w*x[I]
        I = iu0[0]
        J = iu0[1]
        # vectorized
        Wx[I] += w[:, None] * X[J]
        Wx[J] += w[:, None] * X[I]
        # project
        dot = np.sum(Wx * X, axis=1, keepdims=True)
        grad_Y = (Wx - X * dot) / r
        return F, grad_Y.ravel()

    y0 = x_init.ravel().copy()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-10, "maxfun": 10000})
        Y = res.x.reshape(n, d)
    except Exception:
        return x_init
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    X = Y / nrm
    if not np.all(np.isfinite(X)):
        return x_init
    return X


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        x_best = np.load("best.npy")
    except Exception:
        try:
            x_best = np.load("record.npy")
        except Exception:
            x_best = rng.standard_normal((74, 13))
    x_best = np.asarray(x_best, dtype=float)
    # ensure shape (74,13)
    if x_best.shape != (74, 13):
        # try to fix by truncation/padding with random
        tmp = rng.standard_normal((74, 13))
        m = min(x_best.shape[0], 74)
        k = min(x_best.shape[1], 13)
        tmp[:m, :k] = x_best[:m, :k]
        x_best = tmp
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = max_cos(x_best)
    global_best = x_best.copy()
    global_val = best_val

    n, d = 74, 13
    sigmas = [0.02, 0.05, 0.10, 0.20, 0.01, 0.15, 0.30, 0.03]
    it = 0
    # quick initial smooth polish of record to get baseline fast
    # exploration phase until ~135s
    EXPLORE_END = t0 + 135.0
    while time.time() < EXPLORE_END:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        if it % 11 == 0:
            x0 = rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif it % 7 == 0:
            # targeted: perturb only worst points (top contacts)
            G = global_best @ global_best.T
            np.fill_diagonal(G, -2)
            mx = G.max()
            cnt = np.sum(G > mx - 1e-6, axis=1)
            # perturb high-contact points more
            noise = rng.standard_normal((n, d))
            scale = (0.5 + 3.0 * (cnt > 0))[:, None] * sigma
            x0 = global_best + scale * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            noise = rng.standard_normal((n, d))
            x0 = global_best + sigma * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # annealed smoothing
        x = x0
        # adapt schedule by iteration
        if it % 2 == 0:
            sched = [(0.015, 80), (0.006, 150), (0.0025, 250)]
        else:
            sched = [(0.01, 100), (0.004, 200), (0.0012, 300)]
        for tau, mi in sched:
            if time.time() > EXPLORE_END:
                break
            x = lbfgs_smooth(x, tau, mi)
        v = max_cos(x)
        if v < global_val:
            global_val = v
            global_best = x.copy()
        # periodic short LP polish
        if it % 3 == 0 and time.time() < EXPLORE_END:
            remain = min(2.5, EXPLORE_END - time.time())
            if remain > 0.5:
                xp = lp_polish(x, remain)
                vp = max_cos(xp)
                if vp < global_val:
                    global_val = vp
                    global_best = xp.copy()
        # early stop if excellent? continue to use time for further gain
    # final polish with remaining time
    remaining = DEADLINE - time.time()
    if remaining > 5:
        # first a fine L-BFGS at very small tau
        try:
            if remaining > 25:
                global_best = lbfgs_smooth(global_best, 0.001, 400)
                gv = max_cos(global_best)
                if gv < global_val:
                    global_val = gv
                remaining = DEADLINE - time.time()
        except Exception:
            pass
        if remaining > 4:
            # leave 2s margin
            global_best = lp_polish(global_best, max(1.0, remaining - 2.0))
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best

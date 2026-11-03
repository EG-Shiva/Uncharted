import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def solve():
    t0 = time.time()
    TIME_BUDGET = 165.0
    np.random.seed(0)
    rng = np.random.RandomState(1)

    # load start
    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (219, 6):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.randn(219, 6)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = _max_cos(x_best)

    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    def run_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()

        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = g.max()
            # stable log-sum-exp
            e = np.exp(beta * (g - m))
            S = e.sum()
            loss = m + np.log(S) / beta
            w = e / S
            # grad_X = W @ X
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()

        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12,
                                "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def lp_polish(x, seconds):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = 2e-3
        t_end = time.time() + seconds
        while time.time() < t_end and radius > 1e-13:
            G = x @ x.T
            c = G[iu0, iu1]
            m = c.max()
            thr = m - max(20.0 * radius, 2e-9)
            sel = c > thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 4000:
                # keep tightest 4000
                idx = np.argpartition(-c, 4000)[:4000]
                ns = np.zeros_like(sel)
                ns[idx] = True
                # only keep those above thr anyway; but force top 4000
                sel = ns
                # ensure max included
                cnt = 4000
            I = iu0[sel]
            J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.02)
                # keep global best in sync
            else:
                radius *= 0.5
        return x

    # Stage 1: escalate sharpness on current best
    for beta in [80.0, 150.0, 300.0, 600.0]:
        if time.time() - t0 > TIME_BUDGET - 45:
            break
        try:
            x_opt = run_lbfgs(x_best, beta, maxiter=200)
        except Exception:
            continue
        v = _max_cos(x_opt)
        if v < best_val:
            x_best = x_opt
            best_val = v
        if time.time() - t0 > TIME_BUDGET - 45:
            break

    # Stage 2: perturbed smoothing trials
    sigmas = [0.025, 0.012, 0.006, 0.035, 0.018, 0.008]
    trial = 0
    while time.time() - t0 < TIME_BUDGET - 35:
        sigma = sigmas[trial % len(sigmas)]
        # alternate beta
        beta1 = 150.0 if (trial % 2 == 0) else 220.0
        beta2 = 450.0 if (trial % 3 == 0) else 320.0
        pert = rng.randn(n, d)
        # tangent-ish perturbation to avoid radial waste
        proj = (pert * x_best).sum(axis=1, keepdims=True) * x_best
        pert = pert - proj
        x_init = x_best + sigma * pert
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        try:
            x_opt = run_lbfgs(x_init, beta1, maxiter=100)
            # sharpen
            x_opt = run_lbfgs(x_opt, beta2, maxiter=100)
        except Exception:
            trial += 1
            continue
        v = _max_cos(x_opt)
        if v < best_val:
            x_best = x_opt
            best_val = v
        trial += 1
        # small time guard per trial
        if trial > 60:
            break

    # Stage 3: final LP polish with remaining time
    remain = TIME_BUDGET - (time.time() - t0)
    # reserve a few seconds margin, at least 10s polish if possible
    if remain > 5:
        polish_sec = min(remain, 40.0)
        try:
            x_pol = lp_polish(x_best, polish_sec)
            v = _max_cos(x_pol)
            if v < best_val:
                x_best = x_pol
                best_val = v
            else:
                # keep polished version if essentially equal (more tied)
                # only keep if not worse by >1e-9
                if v <= best_val + 1e-7:
                    # prefer more tied? keep best max anyway
                    pass
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

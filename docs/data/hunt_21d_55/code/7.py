import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu0]
        m = cfull.max()
        sel = cfull > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def _smooth_descent(x_init, steps, lr0, beta0, beta1, rng, noise=0.0):
    x = x_init.copy()
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 * (1.0 - 0.6 * f)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ x
        dot = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dot * x
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt /= nrm
        if noise > 0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            Gt = (1.0 - noise) * Gt + noise * nz
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _hole_for_others(X_others, rng, n_cand=20000):
    d = X_others.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_others.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]


def solve():
    rng = np.random.default_rng(20260607)
    t0 = time.time()
    LIMIT = 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((55, 21))
    xb = np.asarray(xb, dtype=np.float64)
    # fix shape if needed
    if xb.shape != (55, 21):
        tmp = rng.standard_normal((55, 21))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        # try to copy overlap
        try:
            m = min(xb.shape[0], 55)
            dd = min(xb.shape[1], 21)
            tmp[:m, :dd] = xb[:m, :dd]
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        except Exception:
            pass
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 55, 21
    best = _max_cos(xb)
    x_best = xb.copy()

    it = 0
    # exploration until ~148s, reserve ~24s final polish
    while time.time() - t0 < 148.0:
        it += 1
        remain = LIMIT - (time.time() - t0)
        if remain < 8:
            break
        mode = it % 6
        x_try = x_best.copy()
        if mode == 0:
            sigma = 0.01 + 0.02 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=400, lr0=0.004,
                                    beta0=250.0, beta1=900.0, rng=rng, noise=0.0)
        elif mode == 1:
            sigma = 0.06 + 0.08 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=600, lr0=0.006,
                                    beta0=200.0, beta1=800.0, rng=rng, noise=0.02)
        elif mode == 2:
            sigma = 0.18 + 0.15 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=700, lr0=0.007,
                                    beta0=150.0, beta1=700.0, rng=rng, noise=0.05)
        elif mode == 3:
            # subset kick on random 3-7 points
            k = int(rng.integers(3, 8))
            idx = rng.choice(n, size=k, replace=False)
            x_try[idx] += (0.4 + 0.6 * rng.random()) * rng.standard_normal((k, d))
            # small global jitter
            x_try += 0.01 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=600, lr0=0.006,
                                    beta0=200.0, beta1=850.0, rng=rng, noise=0.02)
        elif mode == 4:
            # hole reinsertion of 1-2 overloaded points
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # contact count with tol 2e-3 to find hubs
            cnt = np.sum(G > m - 0.002, axis=1)
            # victim: argmax cnt (random tie break)
            order = np.argsort(-cnt + 0.1 * rng.random(n))
            nv = 1 if rng.random() < 0.6 else 2
            victims = order[:nv]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Xk = x_try[keep]
            for v in victims:
                h = _hole_for_others(Xk, rng, n_cand=20000)
                x_try[v] = h
                Xk = np.vstack([Xk, h[None, :]])
            # small shake then smooth
            x_try += 0.005 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=600, lr0=0.005,
                                    beta0=250.0, beta1=900.0, rng=rng, noise=0.01)
        else:
            # push apart current worst pair + smooth
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            flat = np.argmax(G)
            ii, jj = flat // n, flat % n
            # kick those two apart tangentially
            v = x_try[ii] - x_try[jj]
            v -= np.dot(v, x_try[ii]) * x_try[ii]
            vn = np.linalg.norm(v) + 1e-12
            x_try[ii] += 0.05 * v / vn
            x_try[jj] -= 0.05 * v / vn
            x_try += 0.02 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=500, lr0=0.005,
                                    beta0=220.0, beta1=850.0, rng=rng, noise=0.03)

        v_smooth = _max_cos(x_try)
        # only LP-polish if promising (within 12m of best)
        thresh = best + 0.012
        if v_smooth < thresh:
            # short LP budget
            budget = min(2.0, max(0.8, remain - 7.0))
            # don't spend too much if far
            if v_smooth > best + 0.005:
                budget = min(budget, 0.9)
            x_pol = _lp_polish(x_try, seconds=budget, radius_init=8e-4)
            v_pol = _max_cos(x_pol)
            if v_pol < best - 1e-12:
                best = v_pol
                x_best = x_pol.copy()
        else:
            # still accept raw smooth if it beats best (rare)
            if v_smooth < best - 1e-12:
                x_pol = _lp_polish(x_try, seconds=1.5, radius_init=8e-4)
                v_pol = _max_cos(x_pol)
                if v_pol < best:
                    best = v_pol
                    x_best = x_pol.copy()

    # final heavy polish of best
    remain = LIMIT - (time.time() - t0)
    if remain > 3:
        x_best = _lp_polish(x_best, seconds=min(remain - 1.0, 25.0), radius_init=6e-4)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

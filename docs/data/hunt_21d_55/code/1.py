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
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        # cap LP size
        if np.count_nonzero(sel) > 1100:
            # keep top 1100
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1100]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
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


def _hole_for_others(X_others, rng, n_cand=15000):
    d = X_others.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_others.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]


def solve():
    rng = np.random.default_rng(12345)
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
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if (n, d) != (55, 21):
        # fix shape if needed
        tmp = rng.standard_normal((55, 21))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
        n, d = 55, 21
    best = _max_cos(xb)
    x_best = xb.copy()

    it = 0
    # quick initial polish of loaded best with tiny budget to stabilize
    # (skip long polish, parent already polished)
    while time.time() - t0 < 150.0:
        it += 1
        remain = LIMIT - (time.time() - t0)
        if remain < 5:
            break
        mode = it % 5
        x_try = x_best.copy()
        if mode == 1:
            # global shake small
            sigma = 0.015 + 0.01 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=500, lr0=0.006,
                                    beta0=200.0, beta1=700.0, rng=rng, noise=0.02)
        elif mode == 2:
            # global shake larger
            sigma = 0.05 + 0.05 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=700, lr0=0.008,
                                    beta0=150.0, beta1=900.0, rng=rng, noise=0.05)
        elif mode == 3:
            # local shake of worst points
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = 4 + int(rng.integers(0, 4))
            worst = order[:k]
            sig = np.full(n, 0.004)
            sig[worst] = 0.18 + 0.1 * rng.random()
            # also shake neighbors of worst[0]
            neigh = np.argsort(-G[worst[0]])[:8]
            for j in neigh:
                sig[j] = max(sig[j], 0.05)
            x_try = x_try + sig[:, None] * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=600, lr0=0.007,
                                    beta0=250.0, beta1=800.0, rng=rng, noise=0.03)
        elif mode == 4:
            # reinsert 1-2 worst points into holes
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            nr = 1 + int(rng.integers(0, 2))  # 1 or 2
            for kk in range(nr):
                w = order[kk]
                mask = np.ones(n, dtype=bool)
                mask[w] = False
                hole = _hole_for_others(x_try[mask], rng, n_cand=15000)
                # small jitter around hole
                hole = hole + 0.01 * rng.standard_normal(d)
                hole /= np.linalg.norm(hole)
                x_try[w] = hole
            x_try = _smooth_descent(x_try, steps=500, lr0=0.006,
                                    beta0=200.0, beta1=800.0, rng=rng, noise=0.02)
        else:
            # random subset shake
            frac = 0.25 + 0.3 * rng.random()
            k = int(n * frac)
            subset = rng.choice(n, size=k, replace=False)
            sigma = 0.06 + 0.08 * rng.random()
            x_try[subset] += sigma * rng.standard_normal((k, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = _smooth_descent(x_try, steps=600, lr0=0.007,
                                    beta0=180.0, beta1=850.0, rng=rng, noise=0.04)

        v_smooth = _max_cos(x_try)
        # screen: only LP-polish if promising
        if v_smooth > best + 0.012:
            continue
        # short LP polish
        budget = 1.2 if v_smooth > best + 0.001 else 2.5
        remain2 = LIMIT - (time.time() - t0) - 12.0  # reserve final polish
        if remain2 < budget:
            budget = max(0.5, remain2)
        if budget <= 0.3:
            # no time for LP, just check smooth
            if v_smooth < best:
                x_best, best = x_try, v_smooth
            continue
        x_pol = _lp_polish(x_try, seconds=min(budget, 3.0),
                           radius_init=8e-4 if v_smooth > best else 1.2e-3)
        v_pol = _max_cos(x_pol)
        if v_pol < best - 1e-12:
            x_best, best = x_pol, v_pol
        elif v_smooth < best - 1e-12 and v_smooth < v_pol:
            x_best, best = x_try, v_smooth

    # final long polish of best
    remain = LIMIT - (time.time() - t0)
    if remain > 2.0:
        x_best = _lp_polish(x_best, seconds=min(remain - 0.5, 18.0), radius_init=1e-3)
        best = _max_cos(x_best)

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

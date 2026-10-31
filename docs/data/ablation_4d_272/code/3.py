import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for linprog itself
        if time.time() - start > seconds - 0.05:
            break
        gfull = x @ x.T
        c = gfull[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if not np.any(sel):
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        # cap constraints to hardest ones if too many (speed)
        if p > 4000:
            # keep largest 4000
            csel = c[sel]
            idx = np.argpartition(csel, -4000)[-4000:]
            I = I[idx]; J = J[idx]
            csel2 = csel[idx]
            p = 4000
        else:
            csel2 = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel2,
                          A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_anneal(X, iters=1800, k0=12.0, k1=700.0,
                   lr0=0.015, lr1=0.0015, beta=0.85):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    vel = np.zeros_like(X)
    lk0 = np.log(k0)
    lk1 = np.log(k1)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * frac))
        lr = float(lr0 + (lr1 - lr0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights exp(k*(G-m)), stable (arg <=0)
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ X) * (2.0 / S)
        # tangent projection
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme rows for stability
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.5 / nr)
        # only scale down, never up: scale is <=1 when nr>0.5
        # to avoid overhead keep simple: clip if needed
        if float(nr.max()) > 0.5:
            Gt = Gt * scale
        vel = beta * vel - lr * Gt
        X = X + vel
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def _largest_hole(X, exclude=None, ns=20000):
    # random search for point farthest from set
    Z = np.random.randn(ns, 4)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    D = Z @ X.T
    if exclude is not None:
        # exclude one or list
        if np.isscalar(exclude):
            D[:, int(exclude)] = -2.0
        else:
            D[:, np.asarray(exclude, dtype=int)] = -2.0
    m = D.max(axis=1)
    idx = int(m.argmin())
    return Z[idx], float(m[idx])


def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    np.random.seed(1)

    # load start
    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (272, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    X_best = a.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = np.random.randn(272, 4)
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = _max_cos(X_best)

    sigmas = [0.015, 0.03, 0.06, 0.008, 0.045, 0.12, 0.02, 0.09]

    trial = 0
    # main basin-hopping loop, reserve ~45s for final polish
    while True:
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET - 45.0:
            break
        if trial > 200:
            break
        mode = trial % 4
        Y = None
        if mode == 2 or mode == 3:
            # hole-kick: relocate one bad point to largest hole
            try:
                ppm = _per_point_max(X_best)
                # choose index: worst or random among worst
                if mode == 2:
                    # worst point
                    idx = int(np.argmax(ppm))
                else:
                    # random among top-20 worst for diversity
                    order = np.argsort(ppm)[-20:]
                    idx = int(order[np.random.randint(0, len(order))])
                hole, hval = _largest_hole(X_best, exclude=idx, ns=20000)
                Y = X_best.copy()
                # small jitter to break symmetry
                hole = hole + 1e-4 * np.random.randn(4)
                hole /= np.linalg.norm(hole)
                Y[idx] = hole
                # slight global jitter too
                Y = Y + 0.003 * np.random.randn(*Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y = _smooth_anneal(Y, iters=1400, k0=12.0, k1=700.0,
                                   lr0=0.012, lr1=0.0012, beta=0.85)
            except Exception:
                Y = None
        else:
            sigma = sigmas[trial % len(sigmas)]
            try:
                Y = X_best + sigma * np.random.randn(*X_best.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                # vary iters/k slightly by trial for diversity
                if trial % 2 == 0:
                    Y = _smooth_anneal(Y, iters=1800, k0=12.0, k1=700.0,
                                       lr0=0.015, lr1=0.0015, beta=0.85)
                else:
                    Y = _smooth_anneal(Y, iters=1500, k0=25.0, k1=1200.0,
                                       lr0=0.010, lr1=0.0010, beta=0.90)
            except Exception:
                Y = None
        trial += 1
        if Y is None:
            continue
        try:
            v = _max_cos(Y)
        except Exception:
            continue
        # promising? smooth result is not fully polished, allow margin
        if v < best_val + 0.008:
            remain = TIME_BUDGET - 45.0 - (time.time() - t0)
            if remain < 2.0:
                # no time for LP, just check direct improvement
                if v < best_val - 1e-12:
                    X_best = Y
                    best_val = v
                continue
            lp_time = min(6.0, max(2.0, remain / 3.0))
            try:
                Y2 = _lp_polish(Y, seconds=lp_time, r0=2e-3)
                v2 = _max_cos(Y2)
                if v2 < best_val - 1e-12:
                    X_best = Y2
                    best_val = v2
            except Exception:
                if v < best_val - 1e-12:
                    X_best = Y
                    best_val = v
        else:
            # not promising, skip LP
            pass

    # rattler sweep on current best if time allows
    try:
        if time.time() - t0 < TIME_BUDGET - 40.0:
            ppm = _per_point_max(X_best)
            ratt = np.where(ppm < best_val - 5e-5)[0]
            # try up to 3 rattler moves
            for k, idx in enumerate(list(ratt[:3])):
                if time.time() - t0 > TIME_BUDGET - 40.0:
                    break
                hole, hval = _largest_hole(X_best, exclude=int(idx), ns=25000)
                if hval < best_val - 1e-6:
                    Y = X_best.copy()
                    Y[int(idx)] = hole / np.linalg.norm(hole)
                    Y = _smooth_anneal(Y, iters=800, k0=15.0, k1=500.0,
                                       lr0=0.008, lr1=0.001, beta=0.85)
                    Y = _lp_polish(Y, seconds=4.0, r0=1.5e-3)
                    v = _max_cos(Y)
                    if v < best_val - 1e-12:
                        X_best = Y
                        best_val = v
    except Exception:
        pass

    # final thorough polish with remaining time
    try:
        remain = TIME_BUDGET - (time.time() - t0) - 2.0
        if remain > 5.0:
            # cap to avoid overrun, two stages: coarse then fine
            if remain > 30.0:
                X_best = _lp_polish(X_best, seconds=remain * 0.5, r0=2e-3)
                remain = TIME_BUDGET - (time.time() - t0) - 2.0
                if remain > 5.0:
                    X_best = _lp_polish(X_best, seconds=remain, r0=8e-4)
            else:
                X_best = _lp_polish(X_best, seconds=remain, r0=8e-4)
            best_val = _max_cos(X_best)
    except Exception:
        pass

    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

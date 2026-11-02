import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds, t_end_limit):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end_limit:
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def _smooth_adam(X, steps, K0, K1, lr0, lr1):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrk = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 else 0.0
    for t in range(1, steps + 1):
        frac = t / steps
        K = K0 + (K1 - K0) * frac
        cur_lr = lr0 * np.exp(lrk * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        E = np.exp(K * (G - mv))
        np.fill_diagonal(E, 0.0)
        s = E.sum()
        if not np.isfinite(s) or s <= 0:
            break
        Ge = (E @ X) / s
        dot = np.sum(Ge * X, axis=1, keepdims=True)
        Gr = Ge - dot * X
        m1 = b1 * m1 + (1 - b1) * Gr
        v1 = b2 * v1 + (1 - b2) * (Gr * Gr)
        mhat = m1 / (1 - b1 ** t)
        vhat = v1 / (1 - b2 ** t)
        upd = cur_lr * mhat / (np.sqrt(vhat) + eps)
        dd = np.sum(upd * X, axis=1, keepdims=True)
        upd_t = upd - dd * X
        X = X - upd_t
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(0)

    Xbest = None
    best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (58, 22):
                continue
            A = A / np.linalg.norm(A, axis=1, keepdims=True)
            if not np.all(np.isfinite(A)):
                continue
            v = _max_cos(A)
            if v < best:
                best = v
                Xbest = A.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = rng.standard_normal((58, 22))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
        best = _max_cos(Xbest)
    n, d = Xbest.shape

    sigmas = [0.004, 0.009, 0.018, 0.035, 0.07]
    trial = 0
    # reserve ~30s for final LP
    while time.time() < t0 + 138.0:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = (trial - 1) % 4
        X = Xbest.copy()
        if mode == 0:
            X = X + sigma * rng.standard_normal(X.shape)
        elif mode == 1:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            k = 4 + ((trial - 1) % 5)
            idx = np.argsort(worst)[-k:]
            X[idx] += (sigma * 2.5) * rng.standard_normal((k, d))
            X += (sigma * 0.2) * rng.standard_normal(X.shape)
        elif mode == 2:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            i = int(np.argmax(worst))
            cand = rng.standard_normal((2000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ X.T
            dots[:, i] = -2.0
            j = int(np.argmin(dots.max(axis=1)))
            X[i] = cand[j] + 0.01 * rng.standard_normal(d)
            X += (sigma * 0.3) * rng.standard_normal(X.shape)
        else:
            X = X + sigma * rng.standard_normal(X.shape)
            # random small rotation in a random 2D subspace to change topology
            if trial % 2 == 0:
                a = rng.standard_normal(d)
                a /= np.linalg.norm(a)
                b = rng.standard_normal(d)
                b = b - (b @ a) * a
                b /= np.linalg.norm(b) + 1e-12
                ang = sigma * 2.0 * float(rng.standard_normal())
                ca, sa = np.cos(ang), np.sin(ang)
                proj_a = X @ a
                proj_b = X @ b
                X = X + (proj_a * (ca - 1) - proj_b * sa)[:, None] * a[None, :] + \
                    (proj_a * sa + proj_b * (ca - 1))[:, None] * b[None, :]
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # two-phase annealed smoothing
        Xopt = _smooth_adam(X, steps=1400, K0=100.0, K1=500.0,
                            lr0=0.008, lr1=0.002)
        if time.time() > t0 + 138.0:
            v = _max_cos(Xopt)
            if v < best:
                best = v
                Xbest = Xopt.copy()
            break
        Xopt = _smooth_adam(Xopt, steps=900, K0=500.0, K1=2000.0,
                            lr0=0.002, lr1=0.0003)
        v = _max_cos(Xopt)
        if v < best:
            best = v
            Xbest = Xopt / np.linalg.norm(Xopt, axis=1, keepdims=True)
        if time.time() >= t_end - 2.0:
            break

    # final trust-region LP squeeze on the best found
    remain = t_end - time.time() - 1.0
    if remain > 3.0:
        Xbest = _lp_polish(Xbest, seconds=min(remain, 30.0), t_end_limit=t_end - 0.5)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

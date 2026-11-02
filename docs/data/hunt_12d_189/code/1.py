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

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c_all[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d + k, J*d + k, last
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        cols_T = np.full(p, n * d)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2*d+1] = 0  # placeholder to allocate, will overwrite via concat logic below
        # build cols vectorized
        tmp = np.concatenate([cols_I.reshape(p, d), cols_J.reshape(p, d), cols_T.reshape(p, 1)], axis=1).ravel()
        cols = tmp
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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

def _find_hole(X, rng, n_samples=25000, refine_iters=250):
    d = X.shape[1]
    Y = rng.normal(size=(n_samples, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D = Y @ X.T
    m = D.max(axis=1)
    i0 = int(np.argmin(m))
    y = Y[i0].copy()
    best_val = float(m[i0])
    best_y = y.copy()
    lr = 0.05
    for _ in range(refine_iters):
        dv = X @ y
        j = int(np.argmax(dv))
        gj = X[j] - float(X[j] @ y) * y
        gn = np.linalg.norm(gj)
        if gn < 1e-12 or not np.isfinite(gn):
            break
        y = y - lr * gj
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            y = best_y.copy()
            lr *= 0.5
            continue
        y /= nrm
        lr *= 0.995
        val = float((X @ y).max())
        if val < best_val:
            best_val = val
            best_y = y.copy()
    return best_y

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    cands = []
    if xb is not None and xb.shape == (189, 12):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (189, 12):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        Y = rng.normal(size=(189, 12))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        cands.append(Y)

    x_best = min(cands, key=_max_cos)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    M_best = _max_cos(x_best)

    it = 0
    # reserve ~12s for final polish
    while time.time() < deadline - 12.0:
        it += 1
        remain = (deadline - 12.0) - time.time()
        if remain < 1.5:
            break
        polish_sec = min(2.2, remain * 0.4)
        if polish_sec < 0.8:
            polish_sec = min(remain, 1.0)

        mode = it % 4
        x_try = None
        try:
            mi = _per_point_max(x_best)
            order = np.argsort(-mi)
            if mode == 0:
                k = 1
                R = order[:k]
                Xrem = np.delete(x_best, R, axis=0)
                Xa = Xrem.copy()
                holes = []
                for _ in range(k):
                    h = _find_hole(Xa, rng, n_samples=22000, refine_iters=220)
                    holes.append(h)
                    Xa = np.vstack([Xa, h[None, :]])
                x_try = np.vstack([Xrem, np.vstack(holes)])
            elif mode == 1:
                k = 2 if (it % 8 != 1) else 3
                R = order[:k].copy()
                # mix in one random to diversify every other time
                if it % 2 == 0:
                    R[-1] = rng.integers(0, 189)
                    R = np.unique(R)
                    # if duplicate reduced size, pad with random
                    while len(R) < k:
                        r = int(rng.integers(0, 189))
                        if r not in R:
                            R = np.append(R, r)
                Xrem = np.delete(x_best, R, axis=0)
                Xa = Xrem.copy()
                holes = []
                for _ in range(len(R)):
                    h = _find_hole(Xa, rng, n_samples=22000, refine_iters=200)
                    holes.append(h)
                    Xa = np.vstack([Xa, h[None, :]])
                x_try = np.vstack([Xrem, np.vstack(holes)])
            elif mode == 2:
                # biased kick
                Mcur = float(mi.max())
                w = np.exp((mi - Mcur) / 0.005) + 0.05
                w = w / w.sum()
                k = int(rng.integers(4, 12))
                idx = rng.choice(189, size=k, replace=False, p=w)
                sigma = float(rng.uniform(0.03, 0.16))
                x_try = x_best.copy()
                noise = rng.normal(size=(k, 12)) * sigma
                x_try[idx] += noise
                x_try[idx] /= np.linalg.norm(x_try[idx], axis=1, keepdims=True)
            else:
                # broader kick
                Mcur = float(mi.max())
                w = np.exp((mi - Mcur) / 0.008) + 0.08
                w = w / w.sum()
                k = int(rng.integers(10, 24))
                idx = rng.choice(189, size=k, replace=False, p=w)
                sigma = float(rng.uniform(0.08, 0.28))
                x_try = x_best.copy()
                noise = rng.normal(size=(k, 12)) * sigma
                x_try[idx] += noise
                x_try[idx] /= np.linalg.norm(x_try[idx], axis=1, keepdims=True)
                # occasional global tiny jitter
                if it % 8 == 7:
                    x_try = x_try + rng.normal(size=x_try.shape) * 0.008
                    x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        except Exception:
            continue

        try:
            x_new = _lp_polish(x_try, polish_sec, t_deadline=deadline - 11.0)
            M_new = _max_cos(x_new)
        except Exception:
            continue

        if M_new < M_best - 1e-12 and np.all(np.isfinite(x_new)):
            x_best = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            M_best = float(M_new)

    # final polish with remaining time
    try:
        rem = deadline - time.time() - 0.5
        if rem > 1.0:
            x_best = _lp_polish(x_best, rem, t_deadline=deadline - 0.2)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nrm = np.linalg.norm(x_best, axis=1, keepdims=True)
    nrm = np.where(nrm == 0, 1.0, nrm)
    x_best = x_best / nrm
    return x_best

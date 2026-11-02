import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_refine(X, iters=1200, k0=80.0, k1=600.0, lr0=0.03, lr1=0.005):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = k * (G - m)
        # sparse exp: ignore far pairs
        W = np.exp(np.clip(E, -9.0, 0.0))
        W[E < -9.0] = 0.0
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if S < 1e-12:
            continue
        Grad = (W @ X) / S
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        X -= lr * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def lp_polish(x, seconds, init_radius=5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.array(xb, dtype=float))
    if xr is not None:
        cands.append(np.array(xr, dtype=float))
    if not cands:
        cands.append(rng.standard_normal((58, 22)))
    # pick best starting point
    x_best = None
    best = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best:
            best = v
            x_best = c
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    trial = 0
    # exploration phase: basin hopping
    while time.time() - t0 < 150.0:
        trial += 1
        mode = trial % 5
        Xt = x_best.copy()
        n, d = Xt.shape
        if mode == 0:  # global small shake
            Xt += 0.02 * rng.standard_normal(Xt.shape)
        elif mode == 1:  # global medium shake
            Xt += 0.07 * rng.standard_normal(Xt.shape)
        elif mode == 2:  # few points strong shake
            kk = 4
            idx = rng.choice(n, kk, replace=False)
            Xt[idx] += 0.40 * rng.standard_normal((kk, d))
        elif mode == 3:  # single point re-insert
            idx = int(rng.integers(n))
            Xt[idx] = rng.standard_normal(d)
        else:  # medium subset shake
            kk = 8
            idx = rng.choice(n, kk, replace=False)
            Xt[idx] += 0.15 * rng.standard_normal((kk, d))
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        # smooth annealed descent to new basin
        if mode in (2, 3):
            Xt2 = smooth_refine(Xt, iters=1400, k0=60.0, k1=700.0, lr0=0.035, lr1=0.004)
        else:
            Xt2 = smooth_refine(Xt, iters=1100, k0=80.0, k1=600.0, lr0=0.03, lr1=0.005)
        v2 = max_cos(Xt2)
        # only LP-polish promising basins
        thresh = best + 0.0015
        if v2 < thresh:
            # short LP to get true local optimum
            remain = 150.0 - (time.time() - t0)
            if remain < 2.0:
                break
            Xt3 = lp_polish(Xt2, seconds=min(1.6, remain - 0.2), init_radius=0.006)
            v3 = max_cos(Xt3)
            if v3 < best - 1e-12:
                x_best = Xt3
                best = v3
        # safety time check
        if time.time() - t0 > 150.0:
            break

    # final long polish of the best found
    elapsed = time.time() - t0
    remaining = 176.0 - elapsed
    if remaining > 4.0:
        x_best = lp_polish(x_best, seconds=min(remaining, 22.0), init_radius=0.002)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

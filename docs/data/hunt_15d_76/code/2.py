import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    tstart = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - tstart < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
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


def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    phaseA_end = t0 + 105.0

    # load start points
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (76, 15):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append(c)
    rng_global = np.random.default_rng(0)
    if len(cands) == 0:
        X0 = rng_global.standard_normal(size=(76, 15))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    n, d = 76, 15
    rng = np.random.default_rng(12345)

    Ts = np.array([0.015, 0.007, 0.003, 0.0012, 0.0005], dtype=float)
    steps_list = [200, 200, 200, 200, 250]
    hard_steps = 250

    trial = 0
    # preallocate not needed; allocate per trial (small)
    while time.time() < phaseA_end and time.time() < deadline - 20:
        trial += 1
        X = best.copy()
        # perturbation: random subset shake + occasional global jitter
        k = int(rng.integers(4, 18))
        idx = rng.choice(n, size=k, replace=False)
        sigma = float(rng.uniform(0.08, 0.50))
        if rng.random() < 0.5:
            X += rng.standard_normal(size=(n, d)) * 0.02
        X[idx] += rng.standard_normal(size=(k, d)) * sigma
        # occasionally fully randomize a few rattler-like points
        if rng.random() < 0.25:
            k2 = int(rng.integers(1, 4))
            idx2 = rng.choice(n, size=k2, replace=False)
            X[idx2] = rng.standard_normal(size=(k2, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        lr = float(rng.uniform(0.005, 0.014))

        # ---- smooth anneal with Adam on tangent ----
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        p1 = 1.0
        p2 = 1.0
        for Ti, st in zip(Ts, steps_list):
            invT = 1.0 / Ti
            for _ in range(st):
                p1 *= 0.9
                p2 *= 0.999
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                Mx = float(np.max(G))
                # exp((G-Mx)/T)
                E = np.exp((G - Mx) * invT)
                np.fill_diagonal(E, 0.0)
                S = float(np.sum(E))
                if not np.isfinite(S) or S == 0.0:
                    continue
                Grad = (E @ X) * (2.0 / S)
                dot = np.sum(Grad * X, axis=1, keepdims=True)
                Gt = Grad - dot * X
                m *= 0.9
                m += 0.1 * Gt
                v *= 0.999
                v += 0.001 * (Gt * Gt)
                mhat = m / (1.0 - p1)
                vhat = v / (1.0 - p2)
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        # ---- hard max-pair tail ----
        for _ in range(hard_steps):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            Mx = float(np.max(G))
            W = G > (Mx - 0.0008)
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            X -= 0.0003 * Gt
            X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))

        val = _max_cos(X)
        if val < best_val - 1e-12 and np.all(np.isfinite(X)):
            best = X
            best_val = val
            # early exit if big breakthrough: spend polish immediately? continue searching
            if best_val < 0.160:
                break

    # final trust-region LP polish of the best
    remaining = deadline - time.time() - 1.0
    if remaining > 3.0:
        best = _lp_polish(best, seconds=min(remaining, 55.0), deadline=deadline - 0.5)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

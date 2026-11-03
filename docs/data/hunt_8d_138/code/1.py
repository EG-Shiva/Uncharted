import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights in (0,1], max=1
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    # preallocate bound pattern base
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        # limit size for speed
        if np.count_nonzero(sel) > 2500:
            # keep tightest 2500
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    it = 0
    # main basin-hopping loop, leave ~6s for final polish
    while time.time() < deadline - 6.0 and it < 200:
        it += 1
        now = time.time()
        if it % 7 == 0:
            # fresh random basin
            y = rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=1200, beta0=10.0, beta1=300.0, lr0=0.03, lr1=0.002)
            y = _lp_polish(y, min(deadline - 6.0, now + 3.0))
        else:
            if it % 3 == 0:
                sigma = 0.25
            elif it % 3 == 1:
                sigma = 0.08
            else:
                sigma = 0.02
            y = best + sigma * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if it % 4 == 0:
                k = 8
                idx = rng.choice(n, size=k, replace=False)
                rp = rng.normal(size=(k, d))
                rp = rp / np.linalg.norm(rp, axis=1, keepdims=True)
                y[idx] = rp
            y = _anneal(y, steps=700, beta0=20.0, beta1=250.0, lr0=0.02, lr1=0.002)
            y = _lp_polish(y, min(deadline - 6.0, now + 2.0))
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()

    # final tight polish of incumbent
    try:
        best = _lp_polish(best, min(deadline, time.time() + 5.0))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

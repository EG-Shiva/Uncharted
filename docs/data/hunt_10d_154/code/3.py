import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
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


def smooth_refine(X, steps, a0, a1, lr0):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            break
        Grad = (2.0 / S) * (E @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mhat = m / c1
        vhat = v / c2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    # load starts
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        Xbest = rng.standard_normal((154, 10))
    else:
        vals = []
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            vals.append(max_cos(c))
        Xbest = cands[int(np.argmin(vals))].copy()
        Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    n, d = 154, 10
    if Xbest.shape != (n, d):
        # fallback: random or tile
        Xbest = rng.standard_normal((n, d))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    best_val = max_cos(Xbest)

    trial = 0
    # hopping phase, reserve time for final polish
    while time.time() - t0 < 132:
        trial += 1
        mode = trial % 4
        # contact degrees for bias
        Gc = Xbest @ Xbest.T
        np.fill_diagonal(Gc, -2.0)
        thr = best_val - 0.01
        deg = np.sum(Gc > thr, axis=1).astype(float)
        rowmax = np.max(Gc, axis=1)
        if mode == 0:
            sigma = 0.06 + 0.02 * ((trial // 4) % 3)
            Xtry = Xbest + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1200, 30.0, 400.0, 0.015
        elif mode == 1:
            prob = deg + 1.0
            prob = prob / prob.sum()
            k = int(rng.integers(6, 14))
            idx = rng.choice(n, size=k, replace=False, p=prob)
            Xtry = Xbest.copy()
            Xtry[idx] = rng.standard_normal((k, d))
            Xtry += 0.015 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1500, 25.0, 500.0, 0.015
        elif mode == 2:
            # loose points = smallest rowmax (rattlers), tight = largest
            order = np.argsort(rowmax)
            k_loose = 7
            idx_loose = order[:k_loose]
            Xtry = Xbest.copy()
            Xtry[idx_loose] = rng.standard_normal((k_loose, d))
            # shake tight points a bit
            idx_tight = order[-4:]
            Xtry[idx_tight] += 0.08 * rng.standard_normal((4, d))
            Xtry += 0.01 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1500, 25.0, 500.0, 0.012
        else:
            Xtry = rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 2200, 20.0, 500.0, 0.02

        Xref = smooth_refine(Xtry, steps, a0, a1, lr0)
        val = max_cos(Xref)
        if val < best_val - 1e-12:
            # snap to local optimum with short LP
            time_left = 165 - (time.time() - t0)
            if time_left < 2:
                Xbest, best_val = Xref, val
            else:
                sec = min(4.0, max(1.5, time_left / 12.0))
                Xpol = lp_polish(Xref, sec)
                vpol = max_cos(Xpol)
                if vpol < best_val:
                    Xbest, best_val = Xpol, vpol
                elif val < best_val:
                    Xbest, best_val = Xref, val
        elif val < best_val + 0.004 and (trial % 12 == 0):
            if 165 - (time.time() - t0) > 4:
                Xpol = lp_polish(Xref, 2.0)
                vpol = max_cos(Xpol)
                if vpol < best_val:
                    Xbest, best_val = Xpol, vpol
        # safety
        if time.time() - t0 > 133:
            break

    remaining = 178 - (time.time() - t0)
    if remaining > 4:
        Xbest = lp_polish(Xbest, remaining - 1.0)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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

def gd_refine(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights, detached gmax for stability
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) * beta
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1a * m + (1.0 - b1a) * Gt
        vv = b2a * vv + (1.0 - b2a) * (Gt * Gt)
        c1 = 1.0 - b1a ** t
        c2 = 1.0 - b2a ** t
        mhat = m / c1
        vhat = vv / c2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # load starting points
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
        cands.append(_norm(xb))
    if xr is not None:
        cands.append(_norm(xr))
    if len(cands) == 0:
        bestX = _norm(rng.normal(size=(168, 4)))
        best = max_cos(bestX)
    else:
        vals = [max_cos(c) for c in cands]
        bi = int(np.argmin(vals))
        bestX = cands[bi].copy()
        best = float(vals[bi])
    n, d = 168, 4
    # quick contact info to guide: find rattlers occasionally (not strictly needed)
    it = 0
    # hopping budget: leave ~25s for final polish
    HOP_END = 150.0  # seconds from t0
    # adapt if time already spent (loading is instant)
    while True:
        now = time.time() - t0
        if now > HOP_END:
            break
        it += 1
        # choose perturbation mode deterministically + rng
        mode = it % 12
        if mode == 0:
            # fresh random start
            Xtry = _norm(rng.normal(size=(n, d)))
            steps, b0, b1e, lr0, lr1 = 2200, 20.0, 250.0, 0.02, 0.002
        elif mode == 5 or mode == 10:
            # large kick
            sigma = 0.35
            Xtry = bestX + sigma * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
            # resample a few points fully to change topology
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = _norm(rng.normal(size=(k, d)))
            steps, b0, b1e, lr0, lr1 = 1800, 25.0, 280.0, 0.015, 0.0015
        else:
            sigmas = [0.015, 0.04, 0.10]
            sigma = sigmas[it % 3]
            Xtry = bestX + sigma * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
            if it % 3 == 0:
                k = 2 + (it % 5)
                idx = rng.choice(n, size=k, replace=False)
                Xtry[idx] = _norm(rng.normal(size=(k, d)))
            # vary schedule slightly
            if it % 2 == 0:
                steps, b0, b1e, lr0, lr1 = 1500, 30.0, 200.0, 0.02, 0.002
            else:
                steps, b0, b1e, lr0, lr1 = 1800, 20.0, 300.0, 0.012, 0.0012
        Xopt = gd_refine(Xtry, steps, b0, b1e, lr0, lr1)
        v = max_cos(Xopt)
        # decide whether to LP-polish: if better, or close and occasional
        need_polish = False
        polish_time = 1.2
        if v < best:
            need_polish = True
            polish_time = 2.0
        elif v < best + 0.002 and (it % 4 == 0):
            need_polish = True
            polish_time = 1.0
        elif v < best + 0.008 and (it % 15 == 0):
            need_polish = True
            polish_time = 0.8
        if need_polish:
            # guard remaining time
            if time.time() - t0 > HOP_END - 2.0:
                # still evaluate without polish
                if v < best:
                    best = v
                    bestX = Xopt.copy()
                break
            try:
                Xp = lp_polish(Xopt, polish_time)
            except Exception:
                Xp = Xopt
            vp = max_cos(Xp)
            if vp < best:
                best = vp
                bestX = Xp.copy()
            elif v < best:
                best = v
                bestX = Xopt.copy()
        else:
            if v < best:
                best = v
                bestX = Xopt.copy()
        # safety: abort if exceeding time
        if time.time() - t0 > HOP_END:
            break
    # final polish with remaining time
    rem = 178.0 - (time.time() - t0)
    if rem > 3.0:
        rem = min(rem, 25.0)
        try:
            bestX = lp_polish(bestX, rem)
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX

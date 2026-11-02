import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def quench(Y, steps, beta0, beta1, lr0, rng=None):
    n, d = Y.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac
        # cap beta to avoid overflow
        if beta > 600:
            beta = 600.0
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # stable exp
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ Y) / S
        dot = (Grad * Y).sum(axis=1, keepdims=True)
        Tang = Grad - dot * Y
        # clip huge steps
        tn = np.linalg.norm(Tang, axis=1, keepdims=True)
        # avoid explosion
        lr = lr0 * (1.0 - 0.6 * frac)
        Y = Y - lr * Tang
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        # limit constraints to keep linprog fast
        if p > 1200:
            # keep tightest p
            idx = np.argpartition(-c[sel], 1200)[:1200]
            I, J = I[idx], J[idx]
            p = len(I)
            csel = c[sel][idx]
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    # fast basin hopping with void reinsertion
    # budget ~115s for hopping, rest for LP
    HOP_BUDGET = 115.0
    it = 0
    # precompute triu for counts
    while time.time() - t0 < HOP_BUDGET and it < 120:
        it += 1
        # contact counts on best
        Gc = bestX @ bestX.T
        np.fill_diagonal(Gc, -2.0)
        mc = float(Gc.max())
        C = (Gc > mc - 1e-7).sum(axis=1)
        order = np.argsort(C, kind='stable')
        Xtry = bestX.copy()
        if it % 6 == 0:
            sigma = 0.025 if (it % 12 == 0) else 0.012
            Xtry = Xtry + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 160
            b0, b1, lr0 = 60.0, 350.0, 0.008
        else:
            k = 1 if (it % 3 != 0) else 2
            pool = order[:14]
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for cc in chosen:
                cand = rng.standard_normal((1500, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xtry.T
                dots[:, cc] = -2.0
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                Xtry[cc] = cand[bi]
            Xtry = Xtry + 0.004 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 200
            # anneal beta with iteration
            b0, b1, lr0 = 80.0, 450.0, 0.006
            if it % 3 == 0:
                lr0 = 0.009
        Xq = quench(Xtry, steps=steps, beta0=b0, beta1=b1, lr0=lr0)
        vq = max_cos(Xq)
        if vq < bestV:
            bestV = vq
            bestX = Xq.copy()
        # also try mild global jitter quench without void occasionally to refine
        if time.time() - t0 > HOP_BUDGET:
            break
    # final polish: short LP
    time_left = LIMIT - (time.time() - t0)
    # reserve 2s margin
    lp_time = min(28.0, max(4.0, time_left - 2.0))
    if lp_time > 1.0:
        try:
            Xp = lp_polish(bestX, lp_time)
            vp = max_cos(Xp)
            if vp < bestV:
                bestX = Xp
                bestV = vp
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

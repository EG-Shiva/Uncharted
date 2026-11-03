import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def smooth_refine(X, steps, T0, T1, lr0, lr1, mom=0.88):
    n, d = X.shape
    V = np.zeros_like(X)
    logT0 = np.log(T0)
    logT1 = np.log(T1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        T = np.exp(logT0 * (1.0 - frac) + logT1 * frac)
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        E = G
        E -= m
        E /= T
        np.clip(E, -50.0, 0.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            continue
        E /= S
        Gr = E @ X
        Gr *= 2.0
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gr -= dot * X
        V *= mom
        V -= lr * Gr
        X += V
        nr = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
        nr[nr == 0] = 1.0
        X /= nr
    return X


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
    rng = np.random.RandomState(0)
    X0 = None
    R0 = None
    try:
        X0 = np.load("best.npy")
    except Exception:
        X0 = None
    try:
        R0 = np.load("record.npy")
    except Exception:
        R0 = None
    bestX = None
    bestV = np.inf
    for c in (X0, R0):
        if c is None:
            continue
        try:
            c = np.asarray(c, dtype=np.float64)
            if c.shape != (102, 12):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            v = max_cos(c)
            if v < bestV:
                bestV = v
                bestX = c.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.randn(102, 12)
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)

    trial = 0
    # exploration budget
    while time.time() - t0 < 138:
        trial += 1
        r = rng.rand()
        if r < 0.55:
            sigma = 10 ** rng.uniform(-2.2, -0.7)
            Xt = bestX + sigma * rng.randn(102, 12)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0p, T1p, lr0, lr1 = 600, 0.015, 0.0012, 0.025, 0.002
            mom = 0.85 + rng.rand() * 0.08
        elif r < 0.75:
            Xt = bestX.copy()
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = (G > m - 0.005).sum(axis=1)
            order = np.argsort(-cnt)
            pool = order[:20]
            if rng.rand() < 0.6:
                k = 1
            elif rng.rand() < 0.8:
                k = 2
            else:
                k = 3
            chosen = rng.choice(pool, size=k, replace=False)
            for idx in chosen:
                mask = np.ones(102, dtype=bool)
                mask[idx] = False
                Xo = Xt[mask]
                n_try = 2000
                Rc = rng.randn(n_try, 12)
                Rc /= np.linalg.norm(Rc, axis=1, keepdims=True)
                scores = np.max(Rc @ Xo.T, axis=1)
                bi = int(np.argmin(scores))
                Xt[idx] = Rc[bi]
            steps, T0p, T1p, lr0, lr1 = 800, 0.012, 0.0009, 0.02, 0.0015
            mom = 0.88
        elif r < 0.90:
            Xt = bestX.copy()
            k = int(rng.randint(5, 20))
            idxs = rng.choice(102, size=k, replace=False)
            sigma = 10 ** rng.uniform(-1.5, -0.5)
            Xt[idxs] += sigma * rng.randn(k, 12)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0p, T1p, lr0, lr1 = 700, 0.015, 0.001, 0.03, 0.002
            mom = 0.90
        else:
            Xt = rng.randn(102, 12)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0p, T1p, lr0, lr1 = 1200, 0.03, 0.0015, 0.04, 0.002
            mom = 0.90
        try:
            Xt = smooth_refine(Xt, steps, T0p, T1p, lr0, lr1, mom)
            Xt = smooth_refine(Xt, 200, 0.0008, 0.0004, 0.001, 0.0005, 0.90)
        except Exception:
            continue
        if not np.all(np.isfinite(Xt)):
            continue
        v = max_cos(Xt)
        if v < bestV - 1e-12:
            bestV = v
            bestX = Xt.copy()
        if time.time() - t0 > 137:
            break

    if time.time() - t0 < 168:
        try:
            Xt = smooth_refine(bestX.copy(), 700, 0.008, 0.0005, 0.008, 0.0005, 0.90)
            v = max_cos(Xt)
            if v < bestV and np.all(np.isfinite(Xt)):
                bestV = v
                bestX = Xt
        except Exception:
            pass

    remaining = 178.0 - (time.time() - t0)
    if remaining > 6:
        try:
            bestX = lp_polish(bestX, min(remaining - 2.0, 25.0))
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

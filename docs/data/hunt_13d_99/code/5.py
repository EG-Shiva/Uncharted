import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def anneal(X, steps, T0, T1, s0, s1, mu):
    n, d = X.shape
    V = np.zeros_like(X)
    logT0, logT1 = np.log(T0), np.log(T1)
    logs0, logs1 = np.log(s0), np.log(s1)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        T = float(np.exp(logT0 + (logT1 - logT0) * frac))
        st = float(np.exp(logs0 + (logs1 - logs0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(np.max(G))
        G -= gmax
        G /= T
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        rs = G.sum(axis=1, keepdims=True) + 1e-30
        Grad = G @ X
        Grad /= rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Grad -= dot * X
        V *= mu
        V += (1.0 - mu) * Grad
        X -= st * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xbest = np.load("record.npy").astype(float)
        except Exception:
            Xbest = rng.standard_normal((99, 13))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    n, d = Xbest.shape
    if (n, d) != (99, 13):
        # fix shape defensively
        tmp = rng.standard_normal((99, 13))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        Xbest = tmp
    best_val = max_cos(Xbest)

    trial = 0
    # exploration budget ~125s, reserve rest for final polish
    while time.time() - t0 < 125:
        trial += 1
        X = Xbest.copy()
        mode = trial % 4
        if mode == 0:
            sigma = 0.07 + 0.01 * ((trial // 4) % 3)
            X += sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.22 + 0.03 * ((trial // 4) % 3)
            X += sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 2:
            k = 2 + ((trial // 4) % 2)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            idx = np.argsort(rowmax)[-k:]
            for ii in idx:
                cand = rng.standard_normal((2000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                s = cand @ X.T
                s[:, ii] = -2.0
                m = s.max(axis=1)
                X[ii] = cand[np.argmin(m)]
            X += 0.02 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            sel = rng.choice(n, 20, replace=False)
            X[sel] += 0.12 * rng.standard_normal((20, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)

        steps = 1800 + (trial % 3) * 800
        # vary temperature slightly for diversity
        T0 = 0.022 + 0.006 * ((trial // 4) % 3) / 2.0
        T1 = 0.0025
        s0 = 0.06
        s1 = 0.004
        mu = 0.85 if (trial % 2 == 0) else 0.8

        X = anneal(X, steps, T0, T1, s0, s1, mu)
        v = max_cos(X)

        improved = False
        if v < best_val:
            Xp = lp_polish(X, 2.5)
            vp = max_cos(Xp)
            if vp < best_val:
                Xbest, best_val = Xp, vp
                improved = True
            elif v < best_val:
                Xbest, best_val = X, v
                improved = True
        elif v < best_val + 0.010 and (trial % 2 == 0):
            # near miss: check basin with quick polish
            Xp = lp_polish(X, 1.2)
            vp = max_cos(Xp)
            if vp < best_val:
                Xbest, best_val = Xp, vp
                improved = True
        # time guard
        if time.time() - t0 > 126:
            break

    remain = 170 - (time.time() - t0)
    if remain < 5:
        remain = 5
    if remain > 40:
        remain = 40
    try:
        Xbest = lp_polish(Xbest, remain)
    except Exception:
        pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

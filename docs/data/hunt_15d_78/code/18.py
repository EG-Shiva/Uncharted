import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    cnts = C.sum(axis=1)
    return cnts, m, G

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu0]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _fast_quench(X0, iu0, iu1, beta1=120.0, steps1=220, lr1=0.07, beta2=500.0, steps2=220, lr2=0.018):
    X = X0.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    mu = 0.85
    # stage 1
    for s in range(steps1):
        G = X @ X.T
        gv = G[iu0, iu1]
        m = gv.max()
        e = np.exp(beta1 * (gv - m))
        S = e.sum()
        if S == 0 or not np.isfinite(S):
            break
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        # clip
        lr = lr1 * (1.0 - 0.7 * s / max(1, steps1))
        vel = mu * vel + Gt
        X = X - lr * vel
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    vel[:] = 0.0
    mu2 = 0.8
    for s in range(steps2):
        G = X @ X.T
        gv = G[iu0, iu1]
        m = gv.max()
        e = np.exp(beta2 * (gv - m))
        S = e.sum()
        if S == 0 or not np.isfinite(S):
            break
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        lr = lr2 * (1.0 - 0.7 * s / max(1, steps2))
        vel = mu2 * vel + Gt
        X = X - lr * vel
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _smooth_refine(X0, betas, maxiter_each=180):
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Xcur = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xf = Xcur.ravel().copy()
    for beta in betas:
        def fun(g):
            X = g.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            gv = G[iu0]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            S = e.sum()
            F = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0] = w
            W = W + W.T
            GrY = W @ Y
            dot = (GrY * Y).sum(axis=1, keepdims=True)
            GrX = (GrY - dot * Y) / nr
            return float(F), GrX.ravel()
        try:
            res = minimize(fun, xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-7})
            xf = res.x
        except Exception:
            break
    Xo = xf.reshape(n, d)
    Xo /= np.linalg.norm(Xo, axis=1, keepdims=True)
    return Xo

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    iu = np.triu_indices(n, 1)
    iu0, iu1 = iu[0], iu[1]
    # initial snap
    if time.time() - t0 < LIMIT - 150:
        try:
            budget = min(3.0, max(0.5, LIMIT - (time.time() - t0) - 150))
            bestX = _lp_polish(bestX, budget)
            bestV = _max_cos(bestX)
        except Exception:
            pass
    it = 0
    # main loop: many cheap hops + short LP
    while time.time() - t0 < LIMIT - 16:
        it += 1
        tleft = LIMIT - 16 - (time.time() - t0)
        if tleft < 2.5:
            break
        try:
            cnts, mcur, _ = _contacts(bestX, tol=1e-9)
        except Exception:
            cnts = np.zeros(n)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 6
        if mode == 5:
            sigma = float(rng.uniform(0.10, 0.26))
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            if mode in (0, 3):
                k = 1
            elif mode in (1, 4):
                k = 2
            else:
                k = 3
            # occasionally hit overloaded point too
            if it % 11 == 0:
                pool_low = order[:12]
                pool_high = np.argsort(-cnts, kind='stable')[:4]
                chosen_low = rng.choice(pool_low, size=min(k, len(pool_low)), replace=False) if k > 1 else rng.choice(pool_low, size=1, replace=False)
                chosen_high = rng.choice(pool_high, size=1, replace=False)
                chosen = np.concatenate([chosen_low, chosen_high])[:k+1]
            else:
                pool = order[:16]
                chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for idx in chosen:
                idx = int(idx)
                ncand = 300
                cands = rng.standard_normal((ncand, d))
                cands /= np.linalg.norm(cands, axis=1, keepdims=True)
                Gc = cands @ Xtry.T
                Gc[:, idx] = -2.0
                scores = Gc.max(axis=1)
                # pick best, sometimes 2nd best for diversity
                if rng.random() < 0.2:
                    bi = int(np.argsort(scores)[1])
                else:
                    bi = int(np.argmin(scores))
                Xtry[idx] = cands[bi]
            # small shake of rest
            Xtry = Xtry + rng.standard_normal((n, d)) * 0.006
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # fast quench
        try:
            Xq = _fast_quench(Xtry, iu0, iu1)
        except Exception:
            continue
        # short LP squeeze
        budget = 0.9
        if tleft < 8:
            budget = 0.5
        try:
            Xp = _lp_polish(Xq, min(budget, tleft - 0.6))
        except Exception:
            Xp = Xq
        try:
            v = _max_cos(Xp)
        except Exception:
            continue
        if v < bestV - 1e-12:
            bestX = Xp
            bestV = v
        if time.time() - t0 > LIMIT - 16:
            break
    # final intensive polish
    try:
        rem = LIMIT - (time.time() - t0) - 0.8
        if rem > 2:
            bestX = _lp_polish(bestX, min(rem * 0.6, 9.0))
            rem2 = LIMIT - (time.time() - t0) - 0.5
            if rem2 > 4:
                try:
                    Xs = _smooth_refine(bestX, betas=[600.0], maxiter_each=180)
                    Xp2 = _lp_polish(Xs, min(4.0, rem2 - 1.0))
                    if _max_cos(Xp2) < _max_cos(bestX):
                        bestX = Xp2
                except Exception:
                    pass
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

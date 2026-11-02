import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def contact_counts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    return C.sum(axis=1), m, G

def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def smooth_refine(X0, betas, maxiter_each=200):
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
            W = np.zeros((n, n))
            W[iu0] = w
            W = W + W.T
            GradY = W @ Y
            dot = (GradY * Y).sum(axis=1, keepdims=True)
            GradX = (GradY - dot * Y) / nr
            return float(F), GradX.ravel()
        try:
            res = minimize(fun, xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-7})
            xf = res.x
        except Exception:
            break
    Xo = xf.reshape(n, d)
    Xo /= np.linalg.norm(Xo, axis=1, keepdims=True)
    return Xo

def hole_vector(Fixed, rng, S=15000, beta=1800.0):
    d = Fixed.shape[1]
    C = rng.standard_normal((S, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    scores = (C @ Fixed.T).max(axis=1)
    u0 = C[int(np.argmin(scores))].copy()
    # refine single vector via LSE on sphere (unconstrained param)
    v0 = u0.copy()
    def fun(v):
        nrm = max(np.linalg.norm(v), 1e-12)
        u = v / nrm
        gv = Fixed @ u
        m = gv.max()
        e = np.exp(beta * (gv - m))
        Sm = e.sum()
        F = m + np.log(Sm) / beta
        w = e / Sm
        gu = Fixed.T @ w
        gvgrad = (gu - (gu @ u) * u) / nrm
        return float(F), gvgrad
    try:
        res = minimize(fun, v0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
        u = res.x / max(np.linalg.norm(res.x), 1e-12)
    except Exception:
        u = u0
    u /= np.linalg.norm(u)
    return u

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    # quick snap
    if time.time() - t0 < LIMIT - 150:
        bestX = lp_polish(bestX, min(2.0, max(0.5, LIMIT - (time.time()-t0) - 150)))
        bestV = max_cos(bestX)
    curX = bestX.copy(); curV = bestV
    it = 0
    T0 = 0.0015
    while time.time() - t0 < LIMIT - 12:
        it += 1
        tleft = LIMIT - 12 - (time.time() - t0)
        if tleft < 4:
            break
        # temperature decay
        Temp = T0 * np.exp(-(time.time()-t0)/90.0) + 1e-5
        cnts, mcur, Gcur = contact_counts(curX, tol=1e-9)
        order_low = np.argsort(cnts, kind='stable')
        order_high = order_low[::-1]
        r = rng.random()
        Xtry = curX.copy()
        if r < 0.18:
            # global shake
            sigma = float(rng.uniform(0.03, 0.22))
            Xtry = curX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.78:
            # targeted hole re-insertion of 1-3 low / occasional high
            if rng.random() < 0.82:
                pool = order_low[:14]
            else:
                pool = order_high[:6]
            rr = rng.random()
            if rr < 0.55: k = 1
            elif rr < 0.88: k = 2
            else: k = 3
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            remaining = np.ones(n, dtype=bool); remaining[chosen] = False
            Fixed = Xtry[remaining].copy()
            for c in rng.permutation(chosen):
                u = hole_vector(Fixed, rng, S=15000, beta=1800.0)
                Xtry[c] = u
                Fixed = np.vstack([Fixed, u[None, :]])
            # tiny jitter others to unlock
            jit = 0.003 * rng.standard_normal((n, d))
            Xtry = Xtry + jit
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            # block move: joint hole region (move 4-6 lowest together with noise + re-opt)
            pool = order_low[:18]
            k = int(rng.integers(4, 7))
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            sigma = float(rng.uniform(0.08, 0.25))
            Xtry[chosen] = Xtry[chosen] + sigma * rng.standard_normal((len(chosen), d))
            Xtry[chosen] /= np.linalg.norm(Xtry[chosen], axis=1, keepdims=True)
            # also small global jitter
            Xtry += 0.002 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # smooth + polish with adaptive budget
        sbudget = 1
        try:
            # two-stage betas: smaller beta first for topology, larger for max
            if it % 3 == 0:
                betas = [500.0, 2000.0]
            else:
                betas = [800.0, 2500.0]
            Xtry = smooth_refine(Xtry, betas, maxiter_each=180)
        except Exception:
            continue
        lp_time = min(2.0, max(0.8, tleft - 3.0))
        # alternate longer/shorter polish
        if it % 4 == 0:
            lp_time = min(3.0, max(1.0, tleft - 3.0))
        Xtry = lp_polish(Xtry, lp_time)
        v = max_cos(Xtry)
        if v < bestV - 1e-12:
            bestV = v; bestX = Xtry.copy()
            curX = Xtry.copy(); curV = v
            # save progress not allowed? just keep
        elif v < curV or rng.random() < np.exp(-(v - curV) / max(Temp, 1e-9)):
            curX = Xtry.copy(); curV = v
        # occasional restart from best if wandering too high
        if curV > bestV + 0.004 and rng.random() < 0.5:
            curX = bestX.copy(); curV = bestV
    # final snap on best
    trem = LIMIT - (time.time() - t0) - 1.0
    if trem > 1.0:
        bestX = lp_polish(bestX, min(trem, 5.0))
        bestV = max_cos(bestX)
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    # tiny shrink guard against boundary ties due to rounding
    return bestX

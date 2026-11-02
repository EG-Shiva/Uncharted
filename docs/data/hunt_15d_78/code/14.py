import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
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

def _lse_refine(X0, betas, maxiter_each):
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
            m = float(gv.max())
            e = np.exp(beta * (gv - m))
            S = float(e.sum())
            F = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0] = w
            W = W + W.T
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(F), GX.ravel()
        try:
            res = minimize(fun, xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-6})
            xf = res.x
        except Exception:
            break
    Xo = xf.reshape(n, d)
    Xo = Xo / np.linalg.norm(Xo, axis=1, keepdims=True)
    return Xo

def solve():
    t0 = time.time()
    HARD = 174.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    # quick snap of champion
    try:
        bestX = _lp_polish(bestX, min(t0 + 3.0, t0 + HARD - 20))
        bestV = _max_cos(bestX)
    except Exception:
        pass

    iu_all = np.triu_indices(n, 1)
    it = 0
    stall = 0
    while time.time() - t0 < HARD - 14:
        it += 1
        tl = HARD - 14 - (time.time() - t0)
        if tl < 2.5:
            break
        # contact analysis with loose tol to find tail
        try:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            C = (G > mcur - 1e-7)
            cnts = C.sum(axis=1)
        except Exception:
            cnts = np.zeros(n); mcur = bestV
        order = np.argsort(cnts, kind='stable')
        order_hi = np.argsort(-cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 5
        if mode == 4:
            # global shake
            sig = 0.10 + 0.08 * rng.random()
            Xtry = bestX + sig * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            if mode == 0: k = 1
            elif mode == 1: k = 2
            elif mode == 2: k = 2
            else: k = 3
            # boost kick if stalled
            if stall > 12:
                k = min(4, k + 1)
            lows = order[:16]
            chosen = rng.choice(lows, size=min(k, len(lows)), replace=False)
            # reinsert each chosen into largest void sequentially
            for idx in chosen:
                # candidate voids
                nc = 2500
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                Gc = cand @ Xtry.T
                # ignore the point being replaced
                Gc[:, int(idx)] = -2.0
                scores = Gc.max(axis=1)
                bi = int(np.argmin(scores))
                # diversity: sometimes take 2nd/3rd best
                if rng.random() < 0.25:
                    part = np.argpartition(scores, 5)[:5]
                    bi = int(rng.choice(part))
                Xtry[int(idx)] = cand[bi]
            # nudge 1 overloaded hub tangentially
            try:
                hub = int(order_hi[0])
                if cnts[hub] > 20:
                    Xtry[hub] = Xtry[hub] + rng.standard_normal(d) * 0.03
                    Xtry[hub] /= np.linalg.norm(Xtry[hub])
                # tiny global jitter to break symmetry
                Xtry = Xtry + rng.standard_normal((n, d)) * 0.006
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            except Exception:
                pass
        # anneal: one moderate + one high beta, few iters for speed
        try:
            Xsm = _lse_refine(Xtry, betas=[150.0, 500.0], maxiter_each=110)
        except Exception:
            stall += 1
            continue
        # short LP squeeze
        try:
            bdg = min(0.9, max(0.4, tl - 1.0))
            Xpol = _lp_polish(Xsm, time.time() + bdg)
        except Exception:
            Xpol = Xsm
        try:
            v = _max_cos(Xpol)
        except Exception:
            stall += 1
            continue
        if v < bestV - 1e-12:
            bestX = Xpol; bestV = v; stall = 0
        else:
            stall += 1
        if time.time() - t0 > HARD - 14:
            break
    # final intensive polish
    try:
        rem = HARD - (time.time() - t0) - 1.0
        if rem > 1.5:
            bestX = _lp_polish(bestX, time.time() + min(rem, 9.0))
            bestV = _max_cos(bestX)
            rem2 = HARD - (time.time() - t0) - 0.5
            if rem2 > 3.0:
                try:
                    Xsm = _lse_refine(bestX, betas=[700.0], maxiter_each=120)
                    Xpol = _lp_polish(Xsm, time.time() + min(3.0, rem2 - 0.5))
                    if _max_cos(Xpol) < bestV:
                        bestX = Xpol
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

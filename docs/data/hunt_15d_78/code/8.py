import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - tol).sum(axis=1)
    return C, m, G

def _lp_polish(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        cfull = G[iu0]
        m = float(cfull.max())
        thr = max(20.0 * radius, 1e-9)
        sel = cfull > m - thr
        # cap size for speed
        if np.count_nonzero(sel) > 1400:
            # keep tightest 1400
            idx = np.argpartition(cfull, -1400)[-1400:]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        csel = cfull[sel]
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            # keep X at bestX? move to y if not worse much to allow sliding
            radius *= 0.5
            X = bestX.copy() if best < _max_cos(X) else X
        if time.time() >= deadline:
            break
    return bestX

def _smooth(X0, betas, maxiter_each, deadline):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Xcur = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xf = Xcur.ravel().copy()
    for beta in betas:
        if time.time() >= deadline:
            break
        def fun(g):
            X = g.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            gv = G[iu]
            m = float(gv.max())
            e = np.exp(beta * (gv - m))
            S = float(e.sum())
            F = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(F), GX.ravel()
        try:
            res = minimize(fun, xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_each, 'ftol': 1e-13, 'gtol': 1e-6})
            xf = res.x
        except Exception:
            break
        if time.time() >= deadline:
            break
    Xo = xf.reshape(n, d)
    Xo = Xo / np.linalg.norm(Xo, axis=1, keepdims=True)
    return Xo

def _hole_fill(X, idx_list, rng, n_cand=4000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[np.array(idx_list, dtype=int)] = False
    Xrest = X[mask]
    for idx in idx_list:
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        # score = max dot to rest + already filled
        S = C @ X.T
        # ignore columns being replaced (set very low) except current rest
        # simplest: max over rest only plus already-updated points
        S2 = C @ X[mask].T if np.count_nonzero(mask) > 0 else S[:, :1]*0-2
        # Actually use current X with idx excluded:
        Sall = C @ X.T
        Sall[:, idx] = -2.0
        # also ignore other to-be-replaced not yet filled? keep them as blockers
        # score = max
        sc = Sall.max(axis=1)
        bi = int(np.argmin(sc))
        # diversity: sometimes take among top 5 at random
        if rng.random() < 0.3:
            order = np.argpartition(sc, 5)[:5]
            bi = int(rng.choice(order))
        X[idx] = C[bi]
    return X

def solve():
    t0 = time.time()
    LIMIT = 168.0
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
    # initial snap
    try:
        bestX = _lp_polish(bestX, t0 + min(4.0, LIMIT - (time.time()-t0) - 140))
        bestV = _max_cos(bestX)
    except Exception:
        pass

    it = 0
    while time.time() - t0 < LIMIT - 14:
        it += 1
        time_left = LIMIT - 14 - (time.time() - t0)
        if time_left < 4:
            break
        try:
            cnts, mcur, Gcur = _contacts(bestX, tol=1e-7)
        except Exception:
            cnts = np.full(n, 10)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 7
        if mode == 0:
            # global shake large
            sigma = float(rng.uniform(0.10, 0.25))
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 6:
            # shake two loose + jitter
            k = 2
            pool = order[:16]
            chosen = rng.choice(pool, size=k, replace=False).tolist()
            Xtry = _hole_fill(Xtry, chosen, rng, n_cand=5000)
            Xtry = Xtry + rng.standard_normal((n, d)) * 0.006
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            # single kill-and-fill of rattler/loose, sometimes 2
            k = 1 if (it % 3 != 0) else 2
            pool = order[:12]
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
            # occasionally also target overloaded hub to unlock
            if it % 5 == 4:
                hub = int(np.argmax(cnts))
                if hub not in chosen:
                    chosen[-1] = hub
                    # hub gets small move not full reinsert
                    Xtry[hub] = Xtry[hub] + rng.standard_normal(d) * 0.05
                    Xtry[hub] /= np.linalg.norm(Xtry[hub])
                    chosen = [c for c in chosen if c != hub]
                    if len(chosen) == 0:
                        chosen = [int(order[0])]
            if len(chosen) > 0:
                Xtry = _hole_fill(Xtry, chosen, rng, n_cand=5000)
            Xtry = Xtry + rng.standard_normal((n, d)) * 0.007
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # smooth anneal with limited budget
        sdead = time.time() + min(1.2, max(0.4, time_left - 2.0))
        try:
            if mode == 0:
                Xsm = _smooth(Xtry, betas=[80.0, 220.0], maxiter_each=120, deadline=sdead)
            else:
                Xsm = _smooth(Xtry, betas=[100.0, 300.0], maxiter_each=130, deadline=sdead)
        except Exception:
            continue
        if time.time() - t0 > LIMIT - 14:
            # evaluate without LP if no time
            v = _max_cos(Xsm)
            if v < bestV - 1e-12:
                bestX, bestV = Xsm, v
            break
        pdead = time.time() + min(1.6, LIMIT - 14 - (time.time() - t0) - 0.3)
        try:
            if pdead > time.time() + 0.2:
                Xpol = _lp_polish(Xsm, pdead)
            else:
                Xpol = Xsm
        except Exception:
            Xpol = Xsm
        try:
            v = _max_cos(Xpol)
        except Exception:
            continue
        if v < bestV - 1e-12:
            bestX = Xpol
            bestV = v

    # final intensive polish
    try:
        rem = LIMIT - (time.time() - t0) - 0.8
        if rem > 1.5:
            bestX = _lp_polish(bestX, time.time() + min(rem, 10.0))
            bestV = _max_cos(bestX)
            rem2 = LIMIT - (time.time() - t0) - 0.5
            if rem2 > 3:
                Xsm = _smooth(bestX, betas=[700.0], maxiter_each=150, deadline=time.time()+min(2.0, rem2-2.0))
                Xpol = _lp_polish(Xsm, time.time() + min(3.0, LIMIT - (time.time()-t0) - 0.4))
                if _max_cos(Xpol) < bestV:
                    bestX = Xpol
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

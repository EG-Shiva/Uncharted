import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < 300:
        it += 1
        G = x @ x.T
        cfull = G[iu0]
        m = cfull.max()
        thresh = max(20 * radius, 1e-9)
        sel = cfull > m - thresh
        # cap constraints for speed
        cnt = int(sel.sum())
        if cnt > 2500:
            # keep tightest 2500
            idx = np.argpartition(-cfull, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        if cnt == 0:
            radius *= 0.5
            continue
        c = cfull[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8

    # load starts
    candidates = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                candidates.append(a)
        except Exception:
            pass
    if len(candidates) == 0:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    else:
        vals = [_max_cos(a) for a in candidates]
        X0 = candidates[int(np.argmin(vals))].copy()
    Xbest = X0.copy()
    Vbest = _max_cos(Xbest)

    iu = np.triu_indices(n, 1)
    Iall = iu[0]
    Jall = iu[1]

    def make_fun(beta):
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = c.max()
            e = np.exp(beta * (c - m))
            s = e.sum()
            f = m + np.log(s) / beta
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    betas = [3000.0, 15000.0, 80000.0, 400000.0]
    it = 0
    # short initial smooth polish without perturbation
    # main hopping loop
    while time.time() < DEADLINE - 12:
        it += 1
        time_left = DEADLINE - 12 - time.time()
        if time_left < 2:
            break
        if it == 1:
            Xtry = Xbest.copy()
        else:
            mode = it % 5
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            if mode == 1:
                sigma = 0.008
                Xtry = Xbest + sigma * rng.standard_normal(size=(n, d))
            elif mode == 2:
                sigma = 0.025
                Xtry = Xbest + sigma * rng.standard_normal(size=(n, d))
            elif mode == 3:
                # kick worst points hard
                k = 10
                worst = order[:k]
                Xtry = Xbest.copy()
                Xtry[worst] += 0.12 * rng.standard_normal(size=(k, d))
                Xtry += 0.005 * rng.standard_normal(size=(n, d))
            elif mode == 4:
                # re-randomize worst few
                k = 6
                worst = order[:k]
                Xtry = Xbest.copy()
                R = rng.standard_normal(size=(k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                alpha = 0.6
                Xtry[worst] = (1 - alpha) * Xtry[worst] + alpha * R
            else:
                sigma = 0.015
                # weight noise by badness
                wgt = (rowmax - rowmax.min())
                wgt = wgt / (wgt.max() + 1e-12)
                wgt = 0.5 + 1.5 * wgt
                Xtry = Xbest + sigma * wgt[:, None] * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # annealed smooth L-BFGS
        for bi, beta in enumerate(betas):
            if time.time() > DEADLINE - 12:
                break
            fun = make_fun(beta)
            # limit iterations to fit time
            try:
                res = minimize(fun, Xtry.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-9, "maxfun": 100})
                Xn = res.x.reshape(n, d)
                nn = np.linalg.norm(Xn, axis=1, keepdims=True)
                nn = np.maximum(nn, 1e-12)
                Xn = Xn / nn
                # only accept if not NaN
                if np.all(np.isfinite(Xn)):
                    Xtry = Xn
            except Exception:
                break
        Vtry = _max_cos(Xtry)
        if Vtry < Vbest and np.all(np.isfinite(Xtry)):
            Xbest = Xtry.copy()
            Vbest = Vtry
            # quick LP squeeze (2-3s)
            tl = min(4.0, max(1.0, DEADLINE - 12 - time.time()))
            if tl > 0.5:
                Xp = _lp_polish(Xbest, tl)
                Vp = _max_cos(Xp)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
        # opportunistic early LP if stuck very close: every 8 iters polish best
        if it % 8 == 0 and time.time() < DEADLINE - 15:
            Xp = _lp_polish(Xbest, 1.5)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp

    # final polish with remaining time
    rem = DEADLINE - time.time() - 1.0
    if rem > 1.0:
        Xp = _lp_polish(Xbest, min(rem, 25.0))
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

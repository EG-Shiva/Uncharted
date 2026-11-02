import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _lp_polish(x, deadline, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 8e-4
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 400:
        it += 1
        G = x @ x.T
        cfull = G[iu0]
        m = cfull.max()
        thresh = max(25.0 * radius, 2e-9)
        sel = cfull > m - thresh
        cnt = int(sel.sum())
        if cnt > cap:
            idx = np.argpartition(-cfull, cap)[:cap]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = cap
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
            bx = y.copy()
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return bx

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    else:
        vals = [_max_cos(a) for a in cands]
        X0 = cands[int(np.argmin(vals))].copy()
    Xbest = X0.copy()
    Vbest = _max_cos(Xbest)

    iu = np.triu_indices(n, 1)
    Iall = iu[0]; Jall = iu[1]

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

    betas_exp = [2000.0, 20000.0, 200000.0]
    betas_fine = [8000.0, 80000.0, 400000.0]

    def lbfgs_run(Xtry, betas, iters):
        for beta in betas:
            if time.time() > DEADLINE - 22:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, Xtry.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": iters, "ftol": 1e-14, "gtol": 1e-9, "maxfun": iters + 20})
                Xn = res.x.reshape(n, d)
                nn = np.linalg.norm(Xn, axis=1, keepdims=True)
                nn = np.maximum(nn, 1e-12)
                Xn = Xn / nn
                if np.all(np.isfinite(Xn)):
                    Xtry = Xn
            except Exception:
                break
        return Xtry

    def void_replace(X, k, Nc):
        # replace k worst points by farthest voids (sequential greedy)
        Y = X.copy()
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        order = np.argsort(-rm)
        targets = order[:k].copy()
        for t in targets:
            # fixed set without t
            mask = np.ones(n, dtype=bool)
            mask[t] = False
            Yf = Y[mask]
            C = rng.standard_normal(size=(Nc, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Yf.T
            cm = D.max(axis=1)
            bi = int(np.argmin(cm))
            Y[t] = C[bi]
        return Y

    it = 0
    stag = 0
    while time.time() < DEADLINE - 22:
        it += 1
        time_left = DEADLINE - 22 - time.time()
        if time_left < 1.5:
            break
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        mode = it % 7
        if it == 1:
            Xtry = Xbest.copy()
            cur_betas = betas_fine
            cur_iter = 60
        elif mode == 0:
            Xtry = Xbest + 0.008 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            cur_betas = betas_fine; cur_iter = 50
        elif mode == 1:
            Xtry = Xbest + 0.028 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            cur_betas = betas_exp; cur_iter = 50
        elif mode == 2:
            k = 8
            worst = order[:k]
            Xtry = Xbest.copy()
            Xtry[worst] += 0.10 * rng.standard_normal(size=(k, d))
            Xtry += 0.004 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            cur_betas = betas_exp; cur_iter = 55
        elif mode == 3:
            Xtry = void_replace(Xbest, 1, 25000)
            Xtry += 0.003 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            cur_betas = betas_fine; cur_iter = 55
        elif mode == 4:
            kk = 3 if stag < 20 else 5
            Xtry = void_replace(Xbest, kk, 15000)
            Xtry += 0.005 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            cur_betas = betas_exp; cur_iter = 55
        elif mode == 5:
            # weighted noise by badness
            wgt = rowmax - rowmax.min()
            wgt = wgt / (wgt.max() + 1e-12)
            wgt = 0.4 + 1.6 * wgt
            Xtry = Xbest + 0.016 * wgt[:, None] * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            cur_betas = betas_fine; cur_iter = 50
        else:
            # strong shake of rattlers + worst Keep: re-randomize a few mid points
            if stag > 30:
                kk = 6
                worst = order[:kk]
                Xtry = Xbest.copy()
                R = rng.standard_normal(size=(kk, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                alpha = 0.55
                Xtry[worst] = (1 - alpha) * Xtry[worst] + alpha * R
                Xtry += 0.006 * rng.standard_normal(size=(n, d))
            else:
                k = 5
                worst = order[:k]
                Xtry = Xbest.copy()
                Xtry[worst] += 0.15 * rng.standard_normal(size=(k, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            cur_betas = betas_exp; cur_iter = 55

        Xtry = lbfgs_run(Xtry, cur_betas, cur_iter)
        Vtry = _max_cos(Xtry)
        if Vtry < Vbest - 1e-12:
            Xbest = Xtry.copy()
            Vbest = Vtry
            stag = 0
            tl = min(3.5, max(0.8, DEADLINE - 22 - time.time()))
            if tl > 0.6 and time.time() < DEADLINE - 23:
                Xp = _lp_polish(Xbest, time.time() + tl)
                Vp = _max_cos(Xp)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
        else:
            stag += 1
        if it % 9 == 0 and time.time() < DEADLINE - 25:
            Xp = _lp_polish(Xbest, time.time() + 1.2)
            Vp = _max_cos(Xp)
            if Vp < Vbest - 1e-12:
                Xbest, Vbest = Xp, Vp
                stag = 0

    rem = DEADLINE - time.time() - 1.0
    if rem > 1.0:
        Xp = _lp_polish(Xbest, time.time() + min(rem, 28.0))
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

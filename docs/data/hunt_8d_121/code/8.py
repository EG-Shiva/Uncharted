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
    radius = 1e-3
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
        cnt = int(sel.sum())
        if cnt > 3000:
            idx = np.argpartition(-cfull, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 3000
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8

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
    Xcur = Xbest.copy()
    Vcur = Vbest

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

    betas = [3000.0, 15000.0, 80000.0, 300000.0]

    def smooth_opt(Xtry):
        for beta in betas:
            if time.time() > DEADLINE - 12:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, Xtry.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": 45, "ftol": 1e-14, "gtol": 1e-9, "maxfun": 80})
                Xn = res.x.reshape(n, d)
                nn = np.linalg.norm(Xn, axis=1, keepdims=True)
                nn = np.maximum(nn, 1e-12)
                Xn = Xn / nn
                if np.all(np.isfinite(Xn)):
                    Xtry = Xn
            except Exception:
                break
        return Xtry

    def refine_hole(Yoth, c, steps=250):
        # projected softmax descent for single point
        for s in range(steps):
            frac = s / max(1, steps - 1)
            T = 0.02 * ((0.002 / 0.02) ** frac)
            lr = 0.15 / (1.0 + 2.0 * frac)
            Gv = Yoth @ c
            M = Gv.max()
            E = np.exp((Gv - M) / T)
            S = E.sum()
            if not np.isfinite(S) or S <= 0:
                break
            w = E / S
            grad = w @ Yoth
            gdot = float(grad @ c)
            gt = grad - gdot * c
            c = c - lr * gt
            nn = np.linalg.norm(c)
            if nn == 0 or not np.isfinite(nn):
                break
            c = c / nn
        return c

    def hole_for_idx(Y, idx, n_samples=6000):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        Yoth = Y[mask]
        C = rng.standard_normal(size=(n_samples, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Yoth.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        c0 = C[k].copy()
        c1 = refine_hole(Yoth, c0, steps=220)
        return c1

    def hole_attack(Y):
        # find endpoints of top pairs + worst rowmax
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        # top pairs
        flat = G[iu]
        k = min(12, flat.size)
        topk = np.argpartition(-flat, k)[:k]
        cand = set()
        for t in topk:
            cand.add(int(Iall[t])); cand.add(int(Jall[t]))
        order = np.argsort(-rowmax)
        for q in order[:4]:
            cand.add(int(q))
        cand = list(cand)[:10]
        # pairs among others baseline to evaluate
        best_gain = 0
        best_idx = None
        best_c = None
        cur_global = float(flat.max())
        # precompute global without each candidate? just evaluate replacement
        for idx in cand:
            try:
                cnew = hole_for_idx(Y, idx, n_samples=5000)
            except Exception:
                continue
            Y2 = Y.copy()
            Y2[idx] = cnew
            v2 = _max_cos(Y2)
            gain = cur_global - v2
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_c = cnew
        if best_idx is not None and best_gain > 1e-12:
            Ynew = Y.copy()
            Ynew[best_idx] = best_c
            return Ynew / np.linalg.norm(Ynew, axis=1, keepdims=True)
        # no improving hole: force relocation of worst point to its best hole (basin change)
        w = int(order[0])
        try:
            cnew = hole_for_idx(Y, w, n_samples=5000)
            Ynew = Y.copy()
            Ynew[w] = cnew
            # also relocate second worst to random hole to diversify
            if len(order) > 1 and rng.random() < 0.5:
                w2 = int(order[1])
                c2 = hole_for_idx(Ynew, w2, n_samples=3000)
                Ynew[w2] = c2
            Ynew /= np.linalg.norm(Ynew, axis=1, keepdims=True)
            return Ynew
        except Exception:
            return Y.copy()

    it = 0
    # initial quick smooth on best to ensure local opt
    # main loop
    while time.time() < DEADLINE - 12:
        it += 1
        time_left = DEADLINE - 12 - time.time()
        if time_left < 2:
            break
        mode = it % 6
        if mode == 0:
            sigma = 0.008
            Xtry = Xcur + sigma * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.025
            Xtry = Xcur + sigma * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 2:
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = 10
            worst = order[:k]
            Xtry = Xcur.copy()
            Xtry[worst] += 0.12 * rng.standard_normal(size=(k, d))
            Xtry += 0.005 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 3:
            try:
                Xtry = hole_attack(Xcur)
            except Exception:
                Xtry = Xcur + 0.01 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 4:
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = 4
            worst = order[:k]
            Xtry = Xcur.copy()
            R = rng.standard_normal(size=(k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            alpha = 0.7
            Xtry[worst] = (1 - alpha) * Xtry[worst] + alpha * R
            Xtry += 0.004 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            wgt = (rowmax - rowmax.min())
            wgt = wgt / (wgt.max() + 1e-12)
            wgt = 0.5 + 1.5 * wgt
            sigma = 0.015
            Xtry = Xcur + sigma * wgt[:, None] * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        Xopt = smooth_opt(Xtry)
        if not np.all(np.isfinite(Xopt)):
            continue
        Vopt = _max_cos(Xopt)
        # update best
        if Vopt < Vbest:
            Xbest = Xopt.copy()
            Vbest = Vopt
            Xcur = Xopt.copy()
            Vcur = Vopt
            tl = min(4.0, max(1.0, DEADLINE - 12 - time.time()))
            if tl > 0.5:
                try:
                    Xp = _lp_polish(Xbest, tl)
                    Vp = _max_cos(Xp)
                    if Vp < Vbest:
                        Xbest, Vbest = Xp, Vp
                        Xcur, Vcur = Xp.copy(), Vp
                except Exception:
                    pass
        else:
            # wandering acceptance (allow sideways)
            tol = 2e-4 * np.exp(-it / 25.0) + 1e-6
            if Vopt < Vcur + tol:
                # accept if better than cur or with prob for sideways
                if Vopt < Vcur or rng.random() < 0.3:
                    Xcur = Xopt.copy()
                    Vcur = Vopt
            elif rng.random() < 0.02:
                Xcur = Xopt.copy()
                Vcur = Vopt
        if it % 10 == 0 and time.time() < DEADLINE - 15:
            try:
                Xp = _lp_polish(Xcur, 1.0)
                Vp = _max_cos(Xp)
                if Vp < Vcur:
                    Xcur, Vcur = Xp, Vp
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
            except Exception:
                pass

    rem = DEADLINE - time.time() - 1.0
    if rem > 1.0:
        try:
            Xp = _lp_polish(Xbest, min(rem, 25.0))
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

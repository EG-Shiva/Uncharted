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
    while time.time() - start < seconds and radius > 1e-14 and it < 200:
        it += 1
        G = x @ x.T
        cfull = G[iu0]
        m = cfull.max()
        thresh = max(20 * radius, 1e-9)
        sel = cfull > m - thresh
        cnt = int(sel.sum())
        if cnt > 2500:
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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(0)
    n, d = 121, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
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
    # quick initial LP to get true local value if time allows
    Xcur = Xbest.copy()
    Vcur = Vbest
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
    betas = [2500.0, 12000.0, 60000.0, 250000.0]

    def smooth_opt(Xtry):
        for beta in betas:
            if time.time() > DEADLINE - 12:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, Xtry.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": 50, "ftol": 1e-14, "gtol": 1e-9, "maxfun": 90})
                Xn = res.x.reshape(n, d)
                nn = np.linalg.norm(Xn, axis=1, keepdims=True)
                nn = np.maximum(nn, 1e-12)
                Xn = Xn / nn
                if np.all(np.isfinite(Xn)):
                    Xtry = Xn
            except Exception:
                break
        return Xtry

    def refine_hole(Yoth, c, steps=200):
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

    def best_hole(Yoth, n_samples=12000):
        C = rng.standard_normal(size=(n_samples, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Yoth.T
        cm = D.max(axis=1)
        # take best few and refine best
        k = int(np.argmin(cm))
        c0 = C[k].copy()
        c1 = refine_hole(Yoth, c0, steps=180)
        # check refined value vs second candidate
        v1 = float((Yoth @ c1).max())
        # try one more candidate for robustness
        order = np.argpartition(cm, 3)[:3]
        for q in order:
            if int(q) == k:
                continue
            cq = refine_hole(Yoth, C[int(q)].copy(), steps=120)
            vq = float((Yoth @ cq).max())
            if vq < v1:
                v1 = vq
                c1 = cq
        return c1

    def ruin_recreate(Y, k, mode):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        if mode == 0:
            # worst k
            rem = list(order[:k])
        elif mode == 1:
            # worst 2 + random from top-25
            rem = list(order[:min(2, k)])
            pool = list(order[:25])
            while len(rem) < k:
                q = pool[int(rng.integers(0, len(pool)))]
                if q not in rem:
                    rem.append(int(q))
        else:
            # spatial cluster: pick random center, remove nearest neighbours
            ctr = int(rng.integers(0, n))
            sims = (Y @ Y[ctr])
            sims[ctr] = 2.0
            # nearest = largest dot
            nn_order = np.argsort(-sims)
            rem = [ctr] + [int(q) for q in nn_order[:k-1]]
            # ensure worst included sometimes
            if rng.random() < 0.5 and int(order[0]) not in rem:
                rem[-1] = int(order[0])
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.array(rem, dtype=int)] = False
        Ykeep = Y[keep_mask]  # (n-k, d)
        # sequential farthest insertion
        cur = Ykeep
        newpts = []
        for _ in range(k):
            cnew = best_hole(cur, n_samples=10000)
            newpts.append(cnew)
            cur = np.vstack([cur, cnew[None, :]])
        Ynew = Y.copy()
        for idx, cnew in zip(rem, newpts):
            Ynew[int(idx)] = cnew
        Ynew /= np.linalg.norm(Ynew, axis=1, keepdims=True)
        return Ynew

    it = 0
    # main basin-hopping loop
    while time.time() < DEADLINE - 12:
        it += 1
        if time.time() > DEADLINE - 13:
            break
        mode = it % 7
        try:
            if mode == 0:
                sigma = 0.008
                Xtry = Xcur + sigma * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            elif mode == 1:
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                wgt = (rowmax - rowmax.min())
                wgt = wgt / (wgt.max() + 1e-12)
                wgt = 0.4 + 1.6 * wgt
                Xtry = Xcur + 0.016 * wgt[:, None] * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            elif mode == 2:
                k = int(rng.integers(1, 3))  # 1-2
                Xtry = ruin_recreate(Xcur, k, mode=0)
            elif mode == 3:
                k = int(rng.integers(2, 6))  # 2-5
                Xtry = ruin_recreate(Xcur, k, mode=1)
            elif mode == 4:
                k = int(rng.integers(3, 8))
                Xtry = ruin_recreate(Xcur, k, mode=2)
            elif mode == 5:
                # shake spatial neighbourhood jointly
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                ctr = int(order[0])
                sims = Xcur @ Xcur[ctr]
                nn_order = np.argsort(-sims)[:12]
                Xtry = Xcur.copy()
                Xtry[nn_order] += 0.10 * rng.standard_normal(size=(len(nn_order), d))
                Xtry += 0.004 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            else:
                # strong random reinit of few points
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                k = 4
                worst = order[:k]
                Xtry = Xcur.copy()
                R = rng.standard_normal(size=(k, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                alpha = 0.6 + 0.3 * rng.random()
                Xtry[worst] = (1 - alpha) * Xtry[worst] + alpha * R
                Xtry += 0.005 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        except Exception:
            Xtry = Xcur + 0.01 * rng.standard_normal(size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        Xopt = smooth_opt(Xtry)
        if not np.all(np.isfinite(Xopt)):
            continue
        Vopt = _max_cos(Xopt)
        if Vopt < Vbest:
            Xbest = Xopt.copy()
            Vbest = Vopt
            Xcur = Xopt.copy()
            Vcur = Vopt
            tl = min(3.5, max(0.8, DEADLINE - 12 - time.time()))
            if tl > 0.6 and time.time() < DEADLINE - 14:
                try:
                    Xp = _lp_polish(Xbest, tl)
                    Vp = _max_cos(Xp)
                    if Vp < Vbest:
                        Xbest, Vbest = Xp, Vp
                        Xcur, Vcur = Xp.copy(), Vp
                except Exception:
                    pass
        else:
            tol = 2e-4 * np.exp(-it / 30.0) + 2e-6
            if Vopt < Vcur + tol:
                if Vopt < Vcur or rng.random() < 0.35:
                    Xcur = Xopt.copy()
                    Vcur = Vopt
            elif rng.random() < 0.02:
                Xcur = Xopt.copy()
                Vcur = Vopt
        if it % 9 == 0 and time.time() < DEADLINE - 15:
            try:
                Xp = _lp_polish(Xcur, 0.9)
                Vp = _max_cos(Xp)
                if Vp < Vcur:
                    Xcur, Vcur = Xp, Vp
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
            except Exception:
                pass
    rem = DEADLINE - time.time() - 1.0
    if rem > 0.8:
        try:
            Xp = _lp_polish(Xbest, min(rem, 22.0))
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

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
    radius = 1.2e-4
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
        if cnt > 2800:
            idx = np.argpartition(-cfull, 2800)[:2800]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2800
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _search_hole(Y, idx, cur, m_other, rng, n_sample=20000):
    n, d = Y.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Z = Y[mask]
    best_s = np.inf
    best_y = None
    chunk = 5000
    done = 0
    while done < n_sample:
        m = min(chunk, n_sample - done)
        done += m
        C = rng.standard_normal(size=(m, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = (C @ Z.T).max(axis=1)
        k = int(np.argmin(S))
        if S[k] < best_s:
            best_s = float(S[k])
            best_y = C[k].copy()
    if not np.isfinite(best_s) or best_s >= cur:
        return None, np.inf
    # hill-climb refinement around best_y
    y = best_y
    s = best_s
    for sigma in (0.12, 0.06, 0.03, 0.015, 0.007, 0.003):
        for _ in range(2):
            P = rng.standard_normal(size=(2000, d)) * sigma + y[None, :]
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            S = (P @ Z.T).max(axis=1)
            k = int(np.argmin(S))
            if S[k] < s:
                s = float(S[k])
                y = P[k].copy()
            else:
                break
    return y, float(s)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
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

    def smooth_polish(Xtry, betas, maxiter=60):
        for beta in betas:
            if time.time() > DEADLINE - 12:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, Xtry.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9, "maxfun": 120})
                Xn = res.x.reshape(n, d)
                Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(Xn)):
                    Xtry = Xn
            except Exception:
                break
        return Xtry

    betas_full = [3000.0, 15000.0, 80000.0, 300000.0]
    betas_quick = [15000.0, 120000.0]

    # brief initial squeeze if time allows (1s)
    if DEADLINE - time.time() > 20:
        Xp = _lp_polish(Xbest, 1.0)
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp

    it = 0
    while time.time() < DEADLINE - 12:
        it += 1
        # --- targeted single-point hole search on worst points ---
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        cur = float(G.max())
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        improved = False
        # only top 12 worst
        for idx in order[:12]:
            if time.time() > DEADLINE - 12:
                break
            # max without idx
            Gm = G.copy()
            Gm[idx, :] = -2.0
            Gm[:, idx] = -2.0
            m_other = float(Gm.max())
            if m_other >= cur - 1e-12:
                continue
            y_new, s_new = _search_hole(Xbest, int(idx), cur, m_other, rng, n_sample=16000)
            if y_new is None:
                continue
            new_global = m_other if m_other > s_new else s_new
            if new_global < cur - 1e-12:
                Y2 = Xbest.copy()
                Y2[int(idx)] = y_new
                V2 = _max_cos(Y2)
                if V2 < Vbest - 1e-12:
                    Xbest = Y2
                    Vbest = V2
                    improved = True
                    break
        if improved:
            Xbest = smooth_polish(Xbest, betas_quick, maxiter=40)
            Vtry = _max_cos(Xbest)
            if Vtry < Vbest:
                Vbest = Vtry
            tl = min(2.0, max(0.5, DEADLINE - 12 - time.time()))
            if tl > 0.4:
                Xp = _lp_polish(Xbest, tl)
                Vp = _max_cos(Xp)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
            continue

        # --- basin hop with annealed smoothing ---
        mode = it % 6
        if mode == 1:
            sigma = 0.008
            Xtry = Xbest + sigma * rng.standard_normal(size=(n, d))
        elif mode == 2:
            sigma = 0.025
            Xtry = Xbest + sigma * rng.standard_normal(size=(n, d))
        elif mode == 3:
            worst = order[:10]
            Xtry = Xbest.copy()
            Xtry[worst] += 0.12 * rng.standard_normal(size=(10, d))
            Xtry += 0.005 * rng.standard_normal(size=(n, d))
        elif mode == 4:
            worst = order[:6]
            Xtry = Xbest.copy()
            R = rng.standard_normal(size=(6, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            alpha = 0.6
            Xtry[worst] = (1 - alpha) * Xtry[worst] + alpha * R
        elif mode == 5:
            # void-seeking: move worst points toward sampled holes partially
            Xtry = Xbest.copy()
            for idx in order[:4]:
                Gm = G.copy(); Gm[idx,:]=-2.0; Gm[:,idx]=-2.0
                m_other = float(Gm.max())
                y_new, s_new = _search_hole(Xbest, int(idx), 10.0, m_other, rng, n_sample=4000)
                if y_new is not None:
                    Xtry[int(idx)] = 0.5*Xtry[int(idx)]+0.5*y_new
            Xtry += 0.008 * rng.standard_normal(size=(n, d))
        else:
            wgt = (rowmax - rowmax.min())
            wgt = wgt / (wgt.max() + 1e-12)
            wgt = 0.5 + 1.5 * wgt
            Xtry = Xbest + 0.015 * wgt[:, None] * rng.standard_normal(size=(n, d))
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        Xtry = smooth_polish(Xtry, betas_full, maxiter=55)
        Vtry = _max_cos(Xtry)
        if Vtry < Vbest and np.all(np.isfinite(Xtry)):
            Xbest = Xtry.copy()
            Vbest = Vtry
            tl = min(3.0, max(0.8, DEADLINE - 12 - time.time()))
            if tl > 0.4:
                Xp = _lp_polish(Xbest, tl)
                Vp = _max_cos(Xp)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
        if it % 8 == 0 and time.time() < DEADLINE - 15:
            Xp = _lp_polish(Xbest, 1.2)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp

    rem = DEADLINE - time.time() - 1.0
    if rem > 1.0:
        Xp = _lp_polish(Xbest, min(rem, 25.0))
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

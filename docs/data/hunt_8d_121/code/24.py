import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(1)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall = iu[0]; Jall = iu[1]

    def _max_cos(x):
        g = x @ x.T
        np.fill_diagonal(g, -2.0)
        return float(g.max())

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))
    Xcur = Xbest.copy()
    Vcur = Vbest

    def make_fun(beta):
        b = float(beta)
        def fun(flat):
            Xr = flat.reshape(n, d)
            nr = np.linalg.norm(Xr, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xr / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(b * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def lbfgs_anneal(X, betas, iters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 10.0:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-14, "gtol": 1e-9, "maxfun": int(mi*6+20)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        if not np.all(np.isfinite(Xn)):
            return X
        return Xn

    def lp_polish(x, deadline, cap=2800):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        bx = x.copy()
        radius = 5e-4
        it = 0
        iu0 = iu
        while time.time() < deadline and radius > 1e-14 and it < 300:
            it += 1
            G = x @ x.T
            cf = G[iu0]
            m = float(cf.max())
            thresh = max(22.0 * radius, 2e-9)
            sel = cf > m - thresh
            cnt = int(sel.sum())
            if cnt > cap:
                idx = np.argpartition(-cf, cap)[:cap]
                mask = np.zeros_like(sel)
                mask[idx] = True
                sel = mask
                cnt = cap
            if cnt == 0:
                radius *= 0.5
                continue
            c = cf[sel]
            I = iu0[0][sel]; J = iu0[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
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
                radius = min(radius*1.6, 0.02)
            else:
                radius *= 0.5
        return bx

    def multi_void(X, targets, Nc=20000):
        Y = X.copy()
        k = len(targets)
        tset = set([int(t) for t in targets])
        fixed_idx = [i for i in range(n) if i not in tset]
        Yf = Y[fixed_idx]
        C = rng.standard_normal(size=(Nc, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        Df = C @ Yf.T
        mf = Df.max(axis=1)
        picked = []
        used = np.zeros(Nc, dtype=bool)
        Cm = mf.copy()
        # iterative farthest with repulsion among picks
        P = []
        for s in range(k):
            # penalize closeness to already picked
            if P:
                Pmat = np.stack(P, axis=0)  # s x d
                Dp = C @ Pmat.T
                mp = Dp.max(axis=1)
                score = np.maximum(mf, mp)
            else:
                score = mf
            score = np.where(used, 1e9, score)
            bi = int(np.argmin(score))
            used[bi] = True
            # suppress neighbours of bi to enforce diversity
            dotc = C @ C[bi]
            used = used | (dotc > 0.92)
            P.append(C[bi].copy())
        for ti, p in zip(targets, P):
            Y[int(ti)] = p
        return Y

    def cluster_targets(X, m):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        hub = int(np.argmax(rm))
        order = np.argsort(-G[hub])
        tids = [hub] + [int(o) for o in order[:m-1]]
        return tids

    def cap_rotate(X, scale=0.6):
        Y = X.copy()
        c = rng.standard_normal(size=d); c /= np.linalg.norm(c)
        dots = Y @ c
        # take top ~12 points
        k = 12
        idx = np.argpartition(-dots, k)[:k]
        # random skew
        A = rng.standard_normal(size=(d, d))
        S = (A - A.T) * 0.5
        s = float(scale) * (0.5 + rng.random()*0.8)
        try:
            R = np.linalg.solve(np.eye(d) - 0.5*s*S, np.eye(d) + 0.5*s*S)
        except Exception:
            return Y
        Y[idx] = Y[idx] @ R.T
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    betas_exp = [800.0, 4000.0, 20000.0, 100000.0, 400000.0]
    iters_exp = [30, 30, 35, 40, 45]
    betas_fine = [2000.0, 20000.0, 200000.0]
    iters_fine = [40, 45, 50]

    # quick initial squeeze
    if time.time() < DEADLINE - 30:
        Xp = lp_polish(Xbest, time.time()+2.0)
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
            Xcur, Vcur = Xp.copy(), Vp

    it = 0
    stag = 0
    Temp = 0.0025
    while time.time() < DEADLINE - 14:
        it += 1
        if time.time() > DEADLINE - 15:
            break
        # base: mostly from cur, sometimes from best
        if rng.random() < 0.35 or stag > 25:
            base = Xbest
            baseV = Vbest
        else:
            base = Xcur
            baseV = Vcur
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        mode = it % 8
        try:
            if mode == 0:
                Xtry = base + 0.008 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_fine, iters_fine
            elif mode == 1:
                Xtry = base + 0.030 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_exp, iters_exp
            elif mode == 2:
                k = 1
                tids = order[:k]
                Xtry = multi_void(base, tids, Nc=25000)
                Xtry += 0.003 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_fine, iters_fine
            elif mode == 3:
                kk = 4 if stag < 25 else 6
                tids = order[:kk]
                # joint void with same pool
                Xtry = multi_void(base, tids, Nc=22000)
                Xtry += 0.004 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_exp, iters_exp
            elif mode == 4:
                m = 5 if stag < 20 else 7
                tids = cluster_targets(base, m)
                Xtry = multi_void(base, tids, Nc=22000)
                Xtry += 0.005 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_exp, iters_exp
            elif mode == 5:
                Xtry = cap_rotate(base, scale=0.7)
                Xtry += 0.004 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_exp, iters_exp
            elif mode == 6:
                wgt = rowmax - rowmax.min()
                wgt = wgt / (wgt.max() + 1e-12)
                wgt = 0.4 + 1.8 * wgt
                Xtry = base + 0.018 * wgt[:, None] * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_fine, iters_fine
            else:
                # heavy re-randomize of a few worst
                kk = 5
                tids = order[:kk]
                Xtry = base.copy()
                R = rng.standard_normal(size=(kk, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                alpha = 0.6
                for ii, ti in enumerate(tids):
                    Xtry[int(ti)] = (1-alpha)*Xtry[int(ti)] + alpha*R[ii]
                Xtry += 0.006 * rng.standard_normal(size=(n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                cur_b, cur_i = betas_exp, iters_exp
        except Exception:
            continue
        Xtry = lbfgs_anneal(Xtry, cur_b, cur_i)
        try:
            Vtry = _max_cos(Xtry)
        except Exception:
            stag += 1
            continue
        if Vtry < Vbest - 1e-12:
            Xbest = Xtry.copy(); Vbest = Vtry
            Xcur = Xtry.copy(); Vcur = Vtry
            stag = 0
            tl = min(3.0, max(0.7, DEADLINE - 14 - time.time()))
            if tl > 0.6 and time.time() < DEADLINE - 15:
                Xp = lp_polish(Xbest, time.time()+tl)
                Vp = _max_cos(Xp)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
                    Xcur, Vcur = Xp.copy(), Vp
        elif Vtry < Vcur - 1e-12:
            Xcur = Xtry.copy(); Vcur = Vtry
            stag = max(0, stag-1)
        else:
            # metropolis on cur
            if rng.random() < np.exp(-(Vtry - Vcur) / Temp):
                Xcur = Xtry.copy(); Vcur = Vtry
            stag += 1
            if stag > 40:
                # restart cur from best
                Xcur = Xbest.copy(); Vcur = Vbest
                stag = 0
                Temp = max(0.0008, Temp*0.95)
        if it % 10 == 0 and time.time() < DEADLINE - 18:
            Xp = lp_polish(Xbest, time.time()+1.0)
            Vp = _max_cos(Xp)
            if Vp < Vbest - 1e-12:
                Xbest, Vbest = Xp, Vp
                Xcur, Vcur = Xp.copy(), Vp
                stag = 0

    rem = DEADLINE - time.time() - 0.8
    if rem > 1.0:
        Xp = lp_polish(Xbest, time.time()+min(rem, 25.0))
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    n, d = 249, 6
    rng = np.random.default_rng(42)
    np.random.seed(42)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    cur = bestX.copy()

    iu0, iu1 = np.triu_indices(n, 1)

    def full_soft(Xinit, taus, iters, creg=0.2):
        nn = Xinit.shape[0]
        I0, J0 = np.triu_indices(nn, 1)
        xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - 28:
                break
            def fg(xflat, tau=float(tau)):
                X = xflat.reshape(nn, d)
                r = np.sqrt(np.sum(X*X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g-gm)/tau)
                S = np.sum(e)
                Lsoft = gm + tau*np.log(S)
                w = e / S
                W = np.zeros((nn, nn))
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A*Xn, axis=1)
                G0 = (A - proj[:, None]*Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg*np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:, None]*(X/r[:, None])
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
                xf = res.x
            except Exception:
                break
        Xo = xf.reshape(nn, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def subset_polish(Xfull, free_idx, taus, iters, creg=0.2):
        free_idx = np.asarray(free_idx, dtype=int)
        m = len(free_idx)
        if m == 0:
            return Xfull.copy()
        if m >= n:
            return full_soft(Xfull, taus, iters, creg)
        mask = np.ones(n, dtype=bool)
        mask[free_idx] = False
        fix_idx = np.where(mask)[0]
        Xfix = Xfull[fix_idx].copy()
        if m > 1:
            Im, Jm = np.triu_indices(m, 1)
            nff = len(Im)
        else:
            Im = Jm = np.zeros(0, dtype=int)
            nff = 0
        yflat = Xfull[free_idx].copy().ravel()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - 28:
                break
            def fg(yf, tau=float(tau)):
                Y = yf.reshape(m, d)
                r = np.sqrt(np.sum(Y*Y, axis=1))
                r = np.maximum(r, 1e-12)
                Yn = Y / r[:, None]
                D = Yn @ Xfix.T
                if nff > 0:
                    Gf = Yn @ Yn.T
                    gff = Gf[Im, Jm]
                    gm1 = np.max(gff)
                else:
                    gff = np.zeros(0)
                    gm1 = -2.0
                gm2 = np.max(D) if D.size else -2.0
                gm = gm1 if gm1 > gm2 else gm2
                e1 = np.exp((gff-gm)/tau) if nff else np.zeros(0)
                e2 = np.exp((D-gm)/tau)
                S = np.sum(e1) + np.sum(e2)
                Lsoft = gm + tau*np.log(S)
                w1 = e1 / S
                w2 = e2 / S
                A = np.zeros((m, d))
                if nff:
                    W = np.zeros((m, m))
                    W[Im, Jm] = w1
                    W = W + W.T
                    A += W @ Yn
                A += w2 @ Xfix
                proj = np.sum(A*Yn, axis=1)
                G0 = (A - proj[:, None]*Yn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg*np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:, None]*(Y/r[:, None])
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, yflat, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                yflat = res.x
            except Exception:
                break
        Y = yflat.reshape(m, d)
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Out = Xfull.copy()
        Out[free_idx] = Y
        return Out

    def single_refine(Ffix, y0, tau=0.003, creg=1.0):
        yf = y0.copy()
        # normalize start
        def fg(yv, tau=float(tau)):
            r = float(np.sqrt(np.sum(yv*yv)))
            r = max(r, 1e-12)
            yn = yv / r
            dd = Ffix @ yn
            gm = np.max(dd)
            e = np.exp((dd-gm)/tau)
            S = np.sum(e)
            Lsoft = gm + tau*np.log(S)
            w = e / S
            A = w @ Ffix
            a_dot = float(A @ yn)
            G0 = (A - a_dot*yn) / r
            L = Lsoft + creg*(r-1.0)**2
            Gr = G0 + 2.0*creg*(r-1.0)*yn
            return float(L), Gr
        try:
            res = minimize(fg, yf, jac=True, method='L-BFGS-B',
                           options={'maxiter': 120, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-9})
            yf = res.x
        except Exception:
            pass
        yn = yf / max(np.linalg.norm(yf), 1e-12)
        return yn

    def worst_indices(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        # contact counts in tight window for hub detection
        m = float(G.max())
        C = (G > m - 2e-4).sum(axis=1)
        order = np.lexsort((C, rowmax))[::-1]
        return order[:k], rowmax, C, G

    def hole_reinsert(Xbase, r):
        X = Xbase.copy()
        worst, _, _, _ = worst_indices(X, r)
        worst = np.sort(worst)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[worst] = False
        Xkeep = X[keep_mask]  # (n-r) x d
        # sequentially reinsert
        new_pts = []
        curF = Xkeep.copy()
        for k in range(r):
            # sample candidates
            S = 20000
            Cand = rng.standard_normal(size=(S, d))
            Cand /= np.linalg.norm(Cand, axis=1, keepdims=True)
            dots = Cand @ curF.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            y0 = Cand[bi]
            yopt = single_refine(curF, y0, tau=0.004)
            # also try a couple random starts, keep best true max
            for _ in range(2):
                yalt0 = rng.standard_normal(size=(d,))
                yalt0 /= np.linalg.norm(yalt0)
                yalt = single_refine(curF, yalt0, tau=0.004)
                # true max comparison
                m1 = float(np.max(curF @ yopt))
                m2 = float(np.max(curF @ yalt))
                if m2 < m1:
                    yopt = yalt
            new_pts.append(yopt)
            curF = np.vstack([curF, yopt[None, :]])
        Out = np.empty_like(X)
        Out[keep_mask] = Xkeep
        Out[worst] = np.array(new_pts)
        return Out

    # ---- main basin loop ----
    # quick initial jam of loaded best to ensure local optimum
    # (only if time allows, short)
    niter = 0
    # neighbourhood sizes / strategies cycle
    while time.time() < T_END - 32:
        niter += 1
        mode = niter % 4
        try:
            if mode == 1:
                # hub neighbourhood re-opt
                k = int(rng.integers(2, 4))  # remove size small, but here neighbourhood
                worst, _, _, G = worst_indices(bestX, 1)
                i0 = int(worst[0])
                # neighbours: largest dots with i0
                row = G[i0].copy()
                # exclude self (already -2)
                nb = np.argsort(row)[-22:][::-1]  # ~22 neighbours + center
                free = np.unique(np.concatenate([[i0], nb]))
                # add a random jitter to free points to change basin
                trial = bestX.copy()
                trial[free] += 0.004 * rng.standard_normal(size=(len(free), d))
                trial[free] /= np.linalg.norm(trial[free], axis=1, keepdims=True)
                trial = subset_polish(trial, free, taus=[0.008, 0.003, 0.0012], iters=[120, 150, 180])
                # short global soft to rejamm lightly
                trial = full_soft(trial, taus=[0.001], iters=[80])
            elif mode == 2:
                # hole reinsertion of 2-4 worst hubs
                r = int(rng.choice([2, 3, 3, 4]))
                trial = hole_reinsert(bestX, r)
                # polish reinserted + neighbours
                worst2, _, _, G2 = worst_indices(trial, 12)
                # free = reinserted region + close neighbours: take worst 20
                free = worst2[:20]
                trial = subset_polish(trial, free, taus=[0.008, 0.003, 0.001], iters=[120, 150, 200])
                trial = full_soft(trial, taus=[0.0008], iters=[100])
            elif mode == 3:
                # random spherical cap re-opt
                center = rng.standard_normal(size=(d,))
                center /= np.linalg.norm(center)
                dots = bestX @ center
                # cap size random
                thr = float(rng.choice([0.55, 0.65, 0.75, 0.85]))
                free = np.where(dots > thr)[0]
                if len(free) < 5 or len(free) > 60:
                    # fallback: random subset of 20
                    free = rng.choice(n, size=20, replace=False)
                trial = bestX.copy()
                trial[free] += 0.006 * rng.standard_normal(size=(len(free), d))
                trial[free] /= np.linalg.norm(trial[free], axis=1, keepdims=True)
                trial = subset_polish(trial, free, taus=[0.006, 0.002, 0.0008], iters=[120, 150, 200])
            else:
                # global shake + anneal
                sig = float(rng.choice([0.002, 0.004, 0.008]))
                trial = bestX + sig * rng.standard_normal(size=(n, d))
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = full_soft(trial, taus=[0.006, 0.0025, 0.001], iters=[150, 180, 220])
            v = _max_cos(trial)
            if v < bestVal:
                bestVal = v
                bestX = trial.copy()
                cur = trial.copy()
        except Exception:
            continue
        if niter > 60:
            break

    # final intensive annealing from best
    if time.time() < T_END - 20:
        try:
            t = full_soft(bestX, taus=[0.005, 0.002, 0.0009, 0.00045], iters=[200, 250, 280, 320])
            v = _max_cos(t)
            if v < bestVal:
                bestVal = v
                bestX = t
        except Exception:
            pass

    # final trust-region LP jam
    def lp_polish(Xinit, deadline):
        X = Xinit.copy()
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        best = _max_cos(X)
        bx = X.copy()
        radius = 8e-4
        iuA, iuB = np.triu_indices(n, 1)
        while time.time() < deadline and radius > 1e-14:
            if time.time() > T_END - 1.5:
                break
            Gf = X @ X.T
            c = Gf[iuA, iuB]
            m = float(c.max())
            sel = c > m - max(20*radius, 4e-7)
            cnt = int(np.sum(sel))
            if cnt == 0:
                break
            if cnt > 6000:
                idx = np.argpartition(c, -6000)[-6000:]
                msel = np.zeros_like(sel)
                msel[idx] = True
                sel = msel
                cnt = 6000
            I = iuA[sel].astype(int); J = iuB[sel].astype(int)
            p = cnt; nv = n*d + 1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d + np.arange(d), J[:,None]*d + np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
            try:
                A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
                A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
                obj = np.zeros(nv); obj[-1]=1.0
                tl = max(1.0, deadline - time.time() - 0.5)
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                              options={"time_limit": tl})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n,d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                best = v
                bx = y.copy()
                X = y
                radius = min(radius*1.5, 0.01)
            else:
                radius *= 0.5
        return bx, best

    if time.time() < T_END - 3:
        try:
            deadline = T_END - 0.8
            t2, v2 = lp_polish(bestX, deadline)
            if v2 < bestVal:
                bestVal = v2
                bestX = t2
        except Exception:
            pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

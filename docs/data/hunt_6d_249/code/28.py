import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET_END = t0 + 172.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 249, 6

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
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestVal = float(vals[bi])
    cur = bestX.copy()
    curV = bestVal

    def note(X):
        nonlocal bestX, bestVal
        v = _max_cos(X)
        if v < bestVal:
            bestVal = v
            bestX = X.copy()
        return v

    iu0 = np.triu_indices(n, 1)
    I0A, J0A = iu0

    def full_soft(Xinit, tau, maxiter, creg=0.05):
        nn, dd = Xinit.shape
        def fg(xf):
            X = xf.reshape(nn, dd)
            r = np.sqrt(np.sum(X*X, axis=1))
            r = np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            g = G[I0A, J0A]
            gm = np.max(g)
            e = np.exp((g - gm) / tau)
            S = np.sum(e)
            Lsoft = gm + tau*np.log(S)
            w = e / S
            W = np.zeros((nn, nn))
            W[I0A, J0A] = w
            W = W + W.T
            A = W @ Xn
            proj = np.sum(A*Xn, axis=1)
            G0 = (A - proj[:, None]*Xn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + creg*np.sum(diff*diff)
            Gr = G0 + (2.0*creg*diff)[:, None]*(X/r[:, None])
            return float(L), Gr.ravel()
        xf = Xinit.ravel().copy()
        try:
            res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(maxiter), 'maxfun': int(maxiter*4),
                                    'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
            xf = res.x
        except Exception:
            pass
        Xo = xf.reshape(nn, dd)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def subset_polish(Xfull, free_idx, taus, iters, creg=0.05):
        free_idx = np.asarray(free_idx, dtype=int)
        m = len(free_idx)
        if m == 0:
            return Xfull.copy()
        mask = np.ones(n, dtype=bool)
        mask[free_idx] = False
        fix_idx = np.where(mask)[0]
        Xfix = Xfull[fix_idx].copy()
        if m > 1:
            iu = np.triu_indices(m, 1)
            Im, Jm = iu
            nff = len(Im)
        else:
            Im = Jm = np.zeros(0, dtype=int)
            nff = 0
        yflat = Xfull[free_idx].copy().ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > BUDGET_END - 34:
                break
            def fg(yf, tau=tau):
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
                e1 = np.exp((gff - gm)/tau) if nff else np.zeros(0)
                e2 = np.exp((D - gm)/tau)
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
            Ym = yflat.reshape(m, d)
            Ym /= np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
            yflat = Ym.ravel().copy()
        Yf = yflat.reshape(m, d)
        Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
        Xout = Xfull.copy()
        Xout[free_idx] = Yf
        return Xout

    def pick_cap(X, m_lo=12, m_hi=34):
        u = rng.standard_normal(size=(d,))
        if rng.random() < 0.7:
            u = X[int(rng.integers(0, n))] + 0.35*rng.standard_normal(size=(d,))
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            u = rng.standard_normal(size=(d,)); nu = np.linalg.norm(u)
        u /= nu
        dots = X @ u
        m = int(rng.integers(m_lo, m_hi+1))
        thr = np.sort(dots)[n-m]
        free = np.where(dots >= thr)[0]
        return free

    def worst_neighbourhood(X, k=None, m=None):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = np.max(G, axis=1)
        order = np.argsort(-rm)
        if k is None:
            k = int(rng.integers(3, 9))
        worst = order[:k]
        cent = np.mean(X[worst], axis=0)
        cent /= (np.linalg.norm(cent)+1e-12)
        dots = X @ cent
        if m is None:
            m = int(rng.integers(16, 30))
        thr = np.sort(dots)[n-m]
        free = np.where(dots >= thr)[0]
        free = np.union1d(free, worst)
        return free, worst

    def init_holes(Xfix, k):
        # largest-hole greedy diverse init
        C = rng.standard_normal(size=(6000, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xfix.T
        cm = np.max(D, axis=1)
        order = np.argsort(cm)
        picked = []
        for idx in order:
            if len(picked) >= k:
                break
            c = C[idx]
            ok = True
            for p in picked:
                if float(c @ C[p]) > 0.86:
                    ok = False
                    break
            if ok:
                picked.append(int(idx))
        if len(picked) < k:
            for idx in order:
                if int(idx) not in picked:
                    picked.append(int(idx))
                if len(picked) >= k:
                    break
        return C[picked[:k]]

    # diversification
    it = 0
    while time.time() < BUDGET_END - 38:
        it += 1
        rsel = rng.random()
        try:
            if rsel < 0.42:
                free = pick_cap(cur)
                trial0 = cur.copy()
                sig = float(rng.choice([0.04, 0.07, 0.11]))
                P = rng.standard_normal(size=(len(free), d))*sig
                for ii, idx in enumerate(free):
                    v = P[ii] - np.dot(P[ii], trial0[idx])*trial0[idx]
                    trial0[idx] = trial0[idx] + v
                    trial0[idx] /= np.linalg.norm(trial0[idx])
                trial = subset_polish(trial0, free, [0.005, 0.0016, 0.0006], [80, 90, 110])
            elif rsel < 0.74:
                free, worst = worst_neighbourhood(cur)
                trial0 = cur.copy()
                sig = float(rng.choice([0.05, 0.08, 0.12]))
                P = rng.standard_normal(size=(len(free), d))*sig
                for ii, idx in enumerate(free):
                    v = P[ii] - np.dot(P[ii], trial0[idx])*trial0[idx]
                    trial0[idx] = trial0[idx] + v
                    trial0[idx] /= np.linalg.norm(trial0[idx])
                trial = subset_polish(trial0, free, [0.005, 0.0016, 0.00055], [80, 90, 110])
            else:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rm = np.max(G, axis=1)
                order = np.argsort(-rm)
                k = 1 if rng.random() < 0.6 else 2
                if rng.random() < 0.18:
                    k = 3
                rem = order[:k]
                keep = np.ones(n, dtype=bool); keep[rem] = False
                Xfix = cur[keep]
                H0 = init_holes(Xfix, k)
                trial0 = cur.copy()
                trial0[rem] = H0
                # polish holes + a few neighbours for topology change
                dots = trial0 @ np.mean(H0, axis=0)
                m2 = min(n, 14 + k*4)
                thr = np.sort(dots)[n-m2]
                free2 = np.where(dots >= thr)[0]
                free2 = np.union1d(free2, rem)
                trial = subset_polish(trial0, free2, [0.005, 0.0016, 0.00055], [70, 80, 100])
        except Exception:
            continue
        vtry = _max_cos(trial)
        if vtry < bestVal - 1e-12:
            bestVal = vtry
            bestX = trial.copy()
            cur = trial.copy(); curV = vtry
        elif vtry < curV - 1e-12:
            cur = trial.copy(); curV = vtry
        elif rng.random() < 0.12 and vtry < curV + 0.004:
            cur = trial.copy(); curV = vtry
        if it % 12 == 0 and (curV > bestVal + 0.003 or rng.random() < 0.4):
            cur = bestX.copy(); curV = bestVal
            free = pick_cap(cur)
            P = rng.standard_normal(size=(len(free), d))*0.10
            for ii, idx in enumerate(free):
                v = P[ii] - np.dot(P[ii], cur[idx])*cur[idx]
                cur[idx] = cur[idx] + v
                cur[idx] /= np.linalg.norm(cur[idx])
            curV = _max_cos(cur)

    # final jam: annealed full soft-max
    Xfin = bestX.copy()
    if time.time() < BUDGET_END - 30:
        for tau, mi in [(0.0035, 120), (0.0014, 140), (0.00055, 160), (0.00025, 120)]:
            if time.time() > BUDGET_END - 26:
                break
            Xfin = full_soft(Xfin, tau, mi)
            note(Xfin)

    def greedy_polish(Xinit, t_end, step0=2e-3):
        cur2 = Xinit.copy()
        cur2 /= np.maximum(np.linalg.norm(cur2, axis=1, keepdims=True), 1e-12)
        cv = _max_cos(cur2)
        step = step0
        while time.time() < t_end and step > 1e-13:
            G = cur2 @ cur2.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            window = max(10.0*step, 4e-4)
            if step < 5e-5:
                window = max(1.5e-4, 10.0*step)
            mask = G > (m - window)
            F = mask.astype(float) @ cur2
            dot = np.sum(F*cur2, axis=1)
            Ft = F - dot[:, None]*cur2
            trial = cur2 - step*Ft
            trial /= np.maximum(np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            v = _max_cos(trial)
            if v < cv:
                cur2 = trial; cv = v
                step = min(step*1.25, 0.01)
            else:
                step *= 0.5
        return cur2, cv

    if time.time() < BUDGET_END - 22:
        Xfin, vf = greedy_polish(Xfin, min(time.time()+4.0, BUDGET_END-18))
        note(Xfin)

    def lp_polish(x, t_end):
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        best = _max_cos(x)
        bestXl = x.copy()
        radius = 1.5e-3
        iu = np.triu_indices(n, 1)
        ar = np.arange(d)
        rep = np.repeat(np.arange(n), d)
        while time.time() < t_end - 0.5:
            if radius < 1e-14:
                break
            if time.time() > BUDGET_END - 1.5:
                break
            Gfull = x @ x.T
            c_all = Gfull[iu]
            m = float(c_all.max())
            thresh = m - max(20.0*radius, 5e-10)
            sel = np.where(c_all > thresh)[0]
            if sel.size == 0:
                break
            if sel.size > 5000:
                part = np.argpartition(-c_all[sel], 5000)[:5000]
                sel = sel[part]
            I = iu[0][sel]; J = iu[1][sel]
            c_sel = c_all[sel]
            p = sel.size
            nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols_block = np.concatenate([(I[:, None]*d+ar), (J[:, None]*d+ar), np.full((p, 1), n*d)], axis=1).ravel()
            vals_block = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = coo_matrix((vals_block, (rows, cols_block)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (rep, np.arange(n*d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                bounds = [(-radius, radius)]*(n*d)+[(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            nr = np.linalg.norm(y, axis=1, keepdims=True); nr[nr == 0] = 1.0
            y /= nr
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bestXl = y.copy()
                note(y)
                radius = min(radius*1.5, 0.01)
            else:
                radius *= 0.5
        return bestXl

    if time.time() < BUDGET_END - 2:
        try:
            Xlp = lp_polish(Xfin, BUDGET_END - 0.5)
            note(Xlp)
        except Exception:
            pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX

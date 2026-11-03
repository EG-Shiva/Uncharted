import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

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
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
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

    iu0_all = np.triu_indices(n, 1)
    I0A, J0A = iu0_all

    def full_soft(Xinit, tau, maxiter, creg=0.15):
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
        Xo /= np.linalg.norm(Xo, axis=1, keepdims=True)
        return Xo

    def subset_polish(Xfull, free_idx, taus, iters, creg=0.15):
        free_idx = np.asarray(free_idx, dtype=int)
        m = len(free_idx)
        if m == 0:
            return Xfull.copy()
        mask = np.ones(n, dtype=bool)
        mask[free_idx] = False
        fix_idx = np.where(mask)[0]
        Xfix = Xfull[fix_idx].copy()
        nf = len(fix_idx)
        if m > 1:
            iu = np.triu_indices(m, 1)
            Im, Jm = iu
            nff = len(Im)
        else:
            Im = Jm = np.zeros(0, dtype=int)
            nff = 0
        y = Xfull[free_idx].copy()
        yflat = y.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > BUDGET_END - 34:
                break
            def fg(yf, tau=tau):
                Y = yf.reshape(m, d)
                r = np.sqrt(np.sum(Y*Y, axis=1))
                r = np.maximum(r, 1e-12)
                Yn = Y / r[:, None]
                # free-fixed dots
                D = Yn @ Xfix.T  # m x nf
                if nff > 0:
                    Gf = Yn @ Yn.T
                    gff = Gf[Im, Jm]
                    gm1 = np.max(gff) if nff else -2.0
                else:
                    gff = np.zeros(0)
                    gm1 = -2.0
                gm2 = np.max(D) if D.size else -2.0
                gm = gm1 if gm1 > gm2 else gm2
                e1 = np.exp((gff - gm)/tau) if nff else np.zeros(0)
                e2 = np.exp((D - gm)/tau)
                S = np.sum(e1) + np.sum(e2)
                Lsoft = gm + tau*np.log(S)
                # weights
                w1 = e1 / S
                w2 = e2 / S
                # A = contrib
                A = np.zeros((m, d))
                if nff:
                    W = np.zeros((m, m))
                    W[Im, Jm] = w1
                    W = W + W.T
                    A += W @ Yn
                # cross: sum_l w2[k,l] Xfix[l]
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
            Ym /= np.linalg.norm(Ym, axis=1, keepdims=True)
            yflat = Ym.ravel().copy()
        Yf = yflat.reshape(m, d)
        Yf /= np.linalg.norm(Yf, axis=1, keepdims=True)
        Xout = Xfull.copy()
        Xout[free_idx] = Yf
        return Xout

    def pick_cap(X, m_lo=12, m_hi=34):
        u = rng.standard_normal(size=(d,))
        # bias toward existing points to hit dense regions
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
            m = int(rng.integers(16, 28))
        thr = np.sort(dots)[n-m]
        free = np.where(dots >= thr)[0]
        # ensure worst included
        free = np.union1d(free, worst)
        return free, worst

    def init_holes(Xfix, k):
        # sample candidates, greedy diverse best
        C = rng.standard_normal(size=(5000, d))
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
            # if too clustered, after one pass allow closer
        if len(picked) < k:
            for idx in order:
                if int(idx) not in picked:
                    picked.append(int(idx))
                if len(picked) >= k:
                    break
        return C[picked[:k]]

    # ---------- diversification ----------
    it = 0
    while time.time() < BUDGET_END - 38:
        it += 1
        rsel = rng.random()
        try:
            if rsel < 0.44:
                free = pick_cap(cur)
                trial = subset_polish(cur, free, [0.006, 0.002, 0.0007], [90, 90, 120])
            elif rsel < 0.74:
                free, worst = worst_neighbourhood(cur)
                # tangential shake before polish
                trial0 = cur.copy()
                sig = float(rng.choice([0.05, 0.08, 0.12]))
                P = rng.standard_normal(size=(len(free), d))*sig
                for ii, idx in enumerate(free):
                    v = P[ii] - np.dot(P[ii], trial0[idx])*trial0[idx]
                    trial0[idx] = trial0[idx] + v
                    trial0[idx] /= np.linalg.norm(trial0[idx])
                trial = subset_polish(trial0, free, [0.005, 0.0018, 0.0006], [80, 90, 120])
            else:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rm = np.max(G, axis=1)
                order = np.argsort(-rm)
                k = 1 if rng.random() < 0.65 else 2
                if rng.random() < 0.15:
                    k = 3
                rem = order[:k]
                keep = np.ones(n, dtype=bool); keep[rem] = False
                Xfix = cur[keep]
                H0 = init_holes(Xfix, k)
                trial0 = cur.copy()
                trial0[rem] = H0
                trial = subset_polish(trial0, rem, [0.006, 0.002, 0.0006, 0.00035], [80, 80, 100, 100])
        except Exception:
            continue
        vtry = _max_cos(trial)
        if vtry < bestVal - 1e-12:
            bestVal = vtry
            bestX = trial.copy()
            cur = trial.copy(); curV = vtry
            note(trial)
        elif vtry < curV - 1e-12:
            cur = trial.copy(); curV = vtry
        elif rng.random() < 0.14 and vtry < curV + 0.004:
            cur = trial.copy(); curV = vtry
        if it % 12 == 0 and (curV > bestVal + 0.003 or rng.random() < 0.4):
            # jump back to best with a kick to avoid drift
            cur = bestX.copy(); curV = bestVal
            # big kick on a cap so next iter starts elsewhere
            free = pick_cap(cur)
            P = rng.standard_normal(size=(len(free), d))*0.10
            for ii, idx in enumerate(free):
                v = P[ii] - np.dot(P[ii], cur[idx])*cur[idx]
                cur[idx] = cur[idx] + v
                cur[idx] /= np.linalg.norm(cur[idx])
            curV = _max_cos(cur)

    # ---------- final jam ----------
    Xfin = bestX.copy()
    if time.time() < BUDGET_END - 30:
        for tau, mi in [(0.003, 120), (0.0012, 140), (0.0005, 160)]:
            if time.time() > BUDGET_END - 28:
                break
            Xfin = full_soft(Xfin, tau, mi)
            v = note(Xfin)
            # keep jammed track
    # exact squeeze
    def exact_polish(Xinit, t_end, step0=2e-3):
        cur2 = Xinit.copy()
        cur2 /= np.linalg.norm(cur2, axis=1, keepdims=True)
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
    if time.time() < BUDGET_END - 24:
        Xfin, vf = exact_polish(Xfin, min(time.time()+4.0, BUDGET_END-20))
        note(Xfin)
    # LP trust region polish
    def lp_polish(x, t_end):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = 8e-4
        iu = np.triu_indices(n, 1)
        while time.time() < t_end - 0.5:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu]
            m = float(np.max(c))
            sel = c > m - max(20*radius, 5e-10)
            I = iu[0][sel]; J = iu[1][sel]
            p = len(I)
            if p == 0:
                radius *= 0.5; continue
            if p > 6000:
                idx = np.argsort(-c[sel])[:6000]
                I = I[idx]; J = J[idx]; csv = c[sel][idx]
            else:
                csv = c[sel]
                p = len(I)
            nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
            except Exception:
                radius *= 0.5; continue
            if res.status != 0:
                radius *= 0.5; continue
            y = x + res.x[:-1].reshape(n,d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.03)
                note(x)
            else:
                radius *= 0.5
        return x
    rem = BUDGET_END - time.time() - 1.0
    if rem > 6:
        try:
            Xlp = lp_polish(Xfin, BUDGET_END-0.8)
            vl = _max_cos(Xlp)
            if vl < bestVal:
                bestX = Xlp; bestVal = vl
            else:
                # keep whichever is better between Xfin and best
                vf2 = _max_cos(Xfin)
                if vf2 < bestVal:
                    bestX = Xfin; bestVal = vf2
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

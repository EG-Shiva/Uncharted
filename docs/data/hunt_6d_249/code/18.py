import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _soft(Xinit, tau, maxiter, creg=0.1):
    n, d = Xinit.shape
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0
    def fg(xf):
        X = xf.reshape(n, d)
        r = np.sqrt(np.sum(X*X, axis=1))
        r = np.maximum(r, 1e-12)
        Xn = X / r[:, None]
        G = Xn @ Xn.T
        g = G[I0, J0]
        gmax = np.max(g)
        e = np.exp((g - gmax) / tau)
        S = np.sum(e)
        Lsoft = gmax + tau*np.log(S)
        w = e / S
        W = np.zeros((n, n), dtype=float)
        W[I0, J0] = w
        W = W + W.T
        A = W @ Xn
        proj = np.sum(A*Xn, axis=1)
        Grad0 = (A - proj[:, None]*Xn) / r[:, None]
        diff = r - 1.0
        L = Lsoft + creg*np.sum(diff*diff)
        Grad = Grad0 + (2.0*creg*diff)[:, None]*(X/r[:, None])
        return float(L), Grad.ravel()
    Xf = Xinit.ravel().copy()
    try:
        res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'maxfun': int(maxiter*4),
                                'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
        Xf = res.x
    except Exception:
        pass
    Xout = Xf.reshape(n, d)
    Xout /= np.linalg.norm(Xout, axis=1, keepdims=True)
    return Xout

def _exact(Xinit, t_lim, step0=2e-3):
    cur = Xinit.copy()
    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
    curVal = _max_cos(cur)
    step = step0
    while time.time() < t_lim:
        if step < 1e-13:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        window = max(10.0*step, 5e-4)
        if step < 5e-5:
            window = max(2e-4, 10.0*step)
        mask = G > (m - window)
        F = mask.astype(float) @ cur
        dot = np.sum(F*cur, axis=1)
        Ft = F - dot[:, None]*cur
        trial = cur - step*Ft
        rn = np.linalg.norm(trial, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        trial /= rn
        v = _max_cos(trial)
        if v < curVal:
            cur = trial
            curVal = v
            step = min(step*1.25, 0.01)
        else:
            step *= 0.5
            if step < 1e-7:
                break
    return cur, curVal

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20*radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 6000:
            idx = np.argsort(-c[sel])[:6000]
            I = I[idx]; J = J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
            p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _degrees(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    row_max = np.max(G, axis=1)
    deg = np.sum(G > (m - tol), axis=1)
    return G, row_max, deg

def solve():
    t0 = time.time()
    BUDGET_END = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 249, 6
    cands = []
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape==(n,d):
                a = a/np.linalg.norm(a,axis=1,keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal(size=(n,d))
        X0/=np.linalg.norm(X0,axis=1,keepdims=True)
        cands.append(X0)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestVal = float(vals[bi])
    cur = bestX.copy()

    def note(X):
        nonlocal bestX, bestVal
        v = _max_cos(X)
        if v < bestVal:
            bestVal = v
            bestX = X.copy()
        return v

    # quick squeeze of incumbent
    cur = _soft(cur, 0.002, 80)
    note(cur)
    cur, _ = _exact(cur, min(time.time()+1.5, BUDGET_END-120))
    note(cur)

    # helpers for moves
    def hole_for_idx(x, idx, n_cand=15000):
        mask = np.ones(n, dtype=bool); mask[idx]=False
        Xo = x[mask]
        C = rng.standard_normal(size=(n_cand, d))
        C/=np.linalg.norm(C,axis=1,keepdims=True)
        D = C @ Xo.T
        cmax = np.max(D, axis=1)
        b = int(np.argmin(cmax))
        return C[b], float(cmax[b])

    def contact_pairs(x, m, k=1500):
        G = x @ x.T
        np.fill_diagonal(G,-2.0)
        iu = np.triu_indices(n,1)
        c = G[iu]
        idx = np.argsort(-c)[:k]
        return iu[0][idx], iu[1][idx], G

    it = 0
    T_EXPLORE = BUDGET_END - 42.0
    while time.time() < T_EXPLORE:
        it += 1
        # compute degrees of cur/best base
        use_best = (it % 3 == 0)
        base = bestX if use_best and rng.random()<0.6 else cur
        G0 = base @ base.T
        np.fill_diagonal(G0,-2.0)
        m0 = float(G0.max())
        row_max = np.max(G0,axis=1)
        # degree with adaptive tol
        tol = 5e-4 if m0 > bestVal+1e-4 else 2e-9
        deg = np.sum(G0 > (m0-tol), axis=1)
        order_deg = np.argsort(-deg)
        order_row = np.argsort(-row_max)
        mode = it % 6
        Xtry = base.copy()
        if mode == 0:
            # shake high-degree hubs
            k = int(rng.integers(3,8))
            # mix deg and row order
            pool = order_deg[:20] if rng.random()<0.7 else order_row[:20]
            sel = rng.choice(pool, size=min(k,len(pool)), replace=False)
            sigma = float(rng.choice([0.015,0.025,0.04]))
            pert = rng.standard_normal(size=(len(sel),d))*sigma
            for ii,idx in enumerate(sel):
                v = pert[ii]-np.dot(pert[ii],Xtry[idx])*Xtry[idx]
                Xtry[idx]=Xtry[idx]+v
            j = rng.standard_normal(size=(n,d))*(sigma*0.12)
            j-=np.sum(j*Xtry,axis=1,keepdims=True)*Xtry
            Xtry = Xtry+j
        elif mode == 1:
            # cluster move around top hub
            hub = int(order_deg[0] if rng.random()<0.8 else order_row[0])
            grow = G0[hub] > (m0 - max(2e-3, (m0-bestVal+1e-4)))
            nbrs = np.where(grow)[0]
            if len(nbrs)>12:
                nbrs = nbrs[np.argsort(-G0[hub][nbrs])[:12]]
            common = rng.standard_normal(size=(d,))*0.035
            for idx in list(nbrs)+[hub]:
                v = common + rng.standard_normal(size=(d,))*0.012
                v = v - np.dot(v,Xtry[idx])*Xtry[idx]
                Xtry[idx]=Xtry[idx]+v
        elif mode == 2:
            # hole replacement of overloaded point
            cand_pool = order_deg[:8]
            idx = int(rng.choice(cand_pool))
            # quick estimate global excl
            bv, bval = hole_for_idx(Xtry, idx, n_cand=15000)
            Xtry[idx]=bv
            # if not promising still keep as topology change occasionally
        elif mode == 3:
            # pair break: push tight pair apart
            I,J,_ = contact_pairs(base, m0, k=800)
            pi = int(rng.integers(0,len(I)))
            i,j = int(I[pi]), int(J[pi])
            eps = 0.03
            # move i away from j
            vi = -Xtry[j]; vi=vi-np.dot(vi,Xtry[i])*Xtry[i]
            vj = -Xtry[i]; vj=vj-np.dot(vj,Xtry[j])*Xtry[j]
            ni=np.linalg.norm(vi)+1e-12; nj=np.linalg.norm(vj)+1e-12
            Xtry[i]=Xtry[i]+eps*vi/ni + rng.standard_normal(size=(d,))*0.005
            Xtry[j]=Xtry[j]+eps*vj/nj + rng.standard_normal(size=(d,))*0.005
            # jitter their neighbors a bit
            for idx in (i,j):
                pass
            Xtry/=np.linalg.norm(Xtry,axis=1,keepdims=True)
        elif mode == 4:
            # global small jitter + worst shake
            k = int(rng.integers(2,6))
            sel = order_row[:k]
            sigma=0.02
            pert=rng.standard_normal(size=(len(sel),d))*sigma
            for ii,idx in enumerate(sel):
                v=pert[ii]-np.dot(pert[ii],Xtry[idx])*Xtry[idx]
                Xtry[idx]=Xtry[idx]+v
            j=rng.standard_normal(size=(n,d))*0.004
            j-=np.sum(j*Xtry,axis=1,keepdims=True)*Xtry
            Xtry=Xtry+j
        else:
            # large exploration kick on random subset
            k=int(rng.integers(10,20))
            sel=rng.choice(n,size=k,replace=False)
            sigma=0.06
            pert=rng.standard_normal(size=(k,d))*sigma
            for ii,idx in enumerate(sel):
                v=pert[ii]-np.dot(pert[ii],Xtry[idx])*Xtry[idx]
                Xtry[idx]=Xtry[idx]+v
        Xtry/=np.linalg.norm(Xtry,axis=1,keepdims=True)
        # cheap filter polish
        if time.time() > T_EXPLORE-2:
            break
        Xtry = _soft(Xtry, 0.0015, 35)
        Xtry,_ = _exact(Xtry, min(time.time()+0.45, T_EXPLORE))
        vtry = _max_cos(Xtry)
        if vtry < bestVal - 1e-9:
            # deep polish promising
            Xtry = _soft(Xtry, 0.0006, 70)
            Xtry,_ = _exact(Xtry, min(time.time()+0.9, T_EXPLORE+5))
            vtry = note(Xtry)
            cur = Xtry.copy()
        elif vtry < _max_cos(cur) - 5e-7 and rng.random()<0.4:
            cur = Xtry.copy()
        elif rng.random()<0.05:
            cur = Xtry.copy()
        if it % 10 == 0:
            cur = bestX.copy()
            # occasional big hole sweep on cur
            if time.time() < T_EXPLORE-10:
                Gc = cur @ cur.T; np.fill_diagonal(Gc,-2.0)
                mc=float(Gc.max()); rmc=np.max(Gc,axis=1)
                ord2=np.argsort(-rmc)
                improved=False
                for kk in range(4):
                    idx=int(ord2[kk])
                    bv,bval=hole_for_idx(cur, idx, n_cand=8000)
                    # compute current global
                    if bval < mc - 2e-7:
                        cur[idx]=bv
                        cur/=np.linalg.norm(cur,axis=1,keepdims=True)
                        Gc=cur@cur.T; np.fill_diagonal(Gc,-2.0)
                        mc=float(Gc.max()); rmc=np.max(Gc,axis=1); ord2=np.argsort(-rmc)
                        improved=True
                if improved:
                    cur=_soft(cur,0.001,50)
                    note(cur)

    # final polish best
    Xfin = bestX.copy()
    if time.time() < BUDGET_END-30:
        Xfin = _soft(Xfin, 0.0008, 90)
        v=_max_cos(Xfin)
        if v<bestVal:
            bestVal=v; bestX=Xfin.copy()
        else:
            Xfin=bestX.copy()
        Xfin = _soft(Xfin, 0.0004, 90)
        v=_max_cos(Xfin)
        if v<bestVal:
            bestVal=v; bestX=Xfin.copy()
        else:
            Xfin=bestX.copy()
    rem = BUDGET_END - time.time() - 1.0
    if rem > 6:
        try:
            Xlp = _lp_polish(Xfin, rem, BUDGET_END-0.5)
            vlp=_max_cos(Xlp)
            if vlp<bestVal:
                bestX=Xlp; bestVal=vlp
        except Exception:
            pass
    bestX = bestX/np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=2200):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None or len(_IU[0]) != n*(n-1)//2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0*radius, 1e-9)
        sel = c_all > thr
        if int(np.count_nonzero(sel)) == 0:
            break
        if int(np.count_nonzero(sel)) > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 800:
                ids = np.where(sel)[0][::2]
                ns = np.zeros_like(sel)
                take = ids[:cap]
                ns[take] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cd = np.arange(d)
        cols = np.concatenate([I[:,None]*d+cd, J[:,None]*d+cd, np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = _max_cos(y)
        if v < best:
            x,best=y,v
            radius=min(radius*1.6,0.05)
        else:
            radius*=0.5
    return x

def _full_smooth(x, iters, T0, T1, lr0, lr1):
    n,d = x.shape
    for k in range(iters):
        frac = k/max(iters-1,1)
        T = T0*(T1/T0)**frac
        lr = lr0*(lr1/lr0)**frac
        G = x @ x.T
        np.fill_diagonal(G,-2.0)
        m = float(G.max())
        W = np.exp((G-m)/T)
        np.fill_diagonal(W,0.0)
        grad = W @ x
        dot = np.sum(grad*x,axis=1,keepdims=True)
        gt = grad - dot*x
        rs = W.sum(axis=1,keepdims=True)
        step = gt/np.maximum(rs,1e-12)
        sn = np.linalg.norm(step,axis=1,keepdims=True)
        step = step/np.maximum(sn,1.0)
        x = x - lr*step
        x /= np.linalg.norm(x,axis=1,keepdims=True)
    return x

def _expand(reps, k, cos_t, sin_t):
    # reps (r,4), returns X (1+r*k,4)
    r = reps.shape[0]
    n = 1 + r*k
    X = np.empty((n,4))
    X[0] = np.array([1.0,0.0,0.0,0.0])
    a = reps[:,0]; b=reps[:,1]; c=reps[:,2]; d=reps[:,3]
    for t in range(k):
        ct = cos_t[t]; st = sin_t[t]
        c2 = c*ct - d*st
        d2 = c*st + d*ct
        X[1+t::k if False else 0]=0  # placeholder
    # vectorized
    idx=1
    for j in range(r):
        aj,bj,cj,dj = a[j],b[j],c[j],d[j]
        c2 = cj*cos_t - dj*sin_t
        d2 = cj*sin_t + dj*cos_t
        X[idx:idx+k,0]=aj
        X[idx:idx+k,1]=bj
        X[idx:idx+k,2]=c2
        X[idx:idx+k,3]=d2
        idx+=k
    return X

def _sym_smooth(reps, k, iters, T0, T1, lr0, lr1, cos_t, sin_t):
    r = reps.shape[0]
    n = 1 + r*k
    for it in range(iters):
        frac = it/max(iters-1,1)
        T = T0*(T1/T0)**frac
        lr = lr0*(lr1/lr0)**frac
        X = _expand(reps,k,cos_t,sin_t)
        G = X @ X.T
        np.fill_diagonal(G,-2.0)
        m = float(G.max())
        # avoid overflow: (G-m)<=0
        W = np.exp((G-m)/T)
        np.fill_diagonal(W,0.0)
        gradX = W @ X  # (n,4)
        rs = W.sum(axis=1)  # (n,)
        # pullback to reps
        gradR = np.zeros_like(reps)
        sumR = np.zeros(r)
        for t in range(k):
            ct=cos_t[t]; st=sin_t[t]
            # indices for this rotation across all j: 1+j*k+t
            # gather
            for j in range(r):
                gi = 1+j*k+t
                gx,gy,gz,gw = gradX[gi]
                # apply R_{-t} to last two comps
                gz0 = gz*ct + gw*st
                gw0 = -gz*st + gw*ct
                gradR[j,0]+=gx
                gradR[j,1]+=gy
                gradR[j,2]+=gz0
                gradR[j,3]+=gw0
                sumR[j]+=rs[gi]
        # tangent projection + step
        dot = np.sum(gradR*reps,axis=1,keepdims=True)
        gt = gradR - dot*reps
        step = gt/np.maximum(sumR[:,None],1e-12)
        sn = np.linalg.norm(step,axis=1,keepdims=True)
        step = step/np.maximum(sn,1.0)
        reps = reps - lr*step
        reps /= np.linalg.norm(reps,axis=1,keepdims=True)
    return reps

def solve():
    t0=time.time()
    LIMIT=165.0
    # load baseline
    try:
        bestX=np.load("best.npy").astype(float)
    except Exception:
        try:
            bestX=np.load("record.npy").astype(float)
        except Exception:
            rng0=np.random.default_rng(0)
            bestX=rng0.normal(size=(163,4))
    bestX=bestX/np.linalg.norm(bestX,axis=1,keepdims=True)
    bestScore=_max_cos(bestX)
    best=bestX.copy()
    # group orders dividing 162: 1 fixed + r*k
    orders=[27,18,54,81,9,6]
    seed=12345
    for k in orders:
        if time.time()-t0>LIMIT-25:
            break
        if 162%k!=0:
            continue
        r=162//k
        ang=2*np.pi*np.arange(k)/k
        cos_t=np.cos(ang); sin_t=np.sin(ang)
        ntrials=2 if k in (27,18,54) else 1
        for trial in range(ntrials):
            if time.time()-t0>LIMIT-25:
                break
            rng=np.random.default_rng(seed)
            seed+=1
            reps=rng.normal(size=(r,4))
            reps/=np.linalg.norm(reps,axis=1,keepdims=True)
            # symmetric anneal: fewer iters for large r
            if r<=6:
                iters=900
            elif r<=10:
                iters=700
            else:
                iters=500
            reps=_sym_smooth(reps,k,iters,0.03,0.002,0.08,0.008,cos_t,sin_t)
            Xsym=_expand(reps,k,cos_t,sin_t)
            # break symmetry with small noise (deterministic)
            sigma=0.01 if trial==0 else 0.025
            noise=rng.normal(size=Xsym.shape)*sigma
            Xb=Xsym+noise
            Xb/=np.linalg.norm(Xb,axis=1,keepdims=True)
            # full smooth short
            Xb=_full_smooth(Xb,250,0.02,0.002,0.03,0.005)
            # quick LP
            t_rem=LIMIT-(time.time()-t0)
            lp_t=min(4.0,max(1.0,t_rem-20.0))
            if lp_t<1.0:
                lp_t=1.0
            Xb=_lp_polish(Xb,lp_t)
            sc=_max_cos(Xb)
            if sc<bestScore:
                bestScore=sc
                best=Xb.copy()
    # final polish of best with remaining time
    t_rem=LIMIT-(time.time()-t0)
    if t_rem>3:
        # also try polishing a slightly perturbed best to jump basin once
        # keep time for final: reserve 12s for best polish
        if t_rem>18:
            try:
                rng=np.random.default_rng(999)
                Xp=best+rng.normal(size=best.shape)*0.003
                Xp/=np.linalg.norm(Xp,axis=1,keepdims=True)
                Xp=_full_smooth(Xp,200,0.01,0.001,0.02,0.003)
                Xp=_lp_polish(Xp,min(4.0,t_rem-12.0))
                sc=_max_cos(Xp)
                if sc<bestScore:
                    bestScore=sc
                    best=Xp.copy()
            except Exception:
                pass
            t_rem=LIMIT-(time.time()-t0)
        try:
            best=_lp_polish(best,max(1.0,min(12.0,t_rem-1.0)))
        except Exception:
            pass
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    return best

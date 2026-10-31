import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

_IU = None
def _iu(n):
    global _IU
    if _IU is None or _IU[0].shape[0] != n*(n-1)//2:
        _IU = np.triu_indices(n, 1)
    return _IU

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, deadline=None, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu = _iu(n)
    best = max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline - 0.5:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0*radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            # keep largest 2500
            th = np.partition(c_all, -2500)[-2500]
            sel = c_all >= th
            # subsample if still too many
            if int(np.count_nonzero(sel)) > 3200:
                idx = np.where(sel)[0]
                np.random.shuffle(idx) if False else None
                sel2 = np.zeros_like(sel)
                # take every other to reduce
                sel2[idx[::2][:3000]] = True
                sel = sel2
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.6, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_lbfgs(x_init, temps, maxiters, rng=None):
    n,d = x_init.shape
    iu = _iu(n)
    y0 = x_init.ravel().copy()
    y = y0
    for T, mit in zip(temps, maxiters):
        def fun(yf):
            Y = yf.reshape(n,d)
            nr = np.linalg.norm(Y,axis=1,keepdims=True)
            nr = np.maximum(nr,1e-12)
            X = Y/nr
            G = X @ X.T
            g = G[iu]
            m = float(g.max())
            e = np.exp((g-m)/T)
            S = float(e.sum())
            F = m + T*np.log(S+1e-300)
            W = np.zeros((n,n))
            W[iu] = e/S
            W = W + W.T
            GX = W @ X
            dot = np.sum(GX*X,axis=1,keepdims=True)
            GY = (GX - dot*X)/nr
            return F, GY.ravel()
        try:
            res = minimize(fun, y, jac=True, method="L-BFGS-B",
                           options={"maxiter": mit, "ftol":1e-14, "gtol":1e-7, "maxfun":100000})
            y = res.x
        except Exception:
            break
    Y = y.reshape(n,d)
    Y /= np.linalg.norm(Y,axis=1,keepdims=True)
    return Y

def find_voids(x_keep, m, rng, S=16000):
    y = rng.standard_normal(size=(S,4))
    y /= np.linalg.norm(y,axis=1,keepdims=True)
    scores = (y @ x_keep.T).max(axis=1)
    order = np.argsort(scores)
    picked=[]
    for idx in order:
        if len(picked)>=m: break
        cand=y[idx]
        ok=True
        for q in picked:
            if float(cand@q)>0.80:
                ok=False; break
        if ok:
            picked.append(cand)
    k=0
    while len(picked)<m:
        picked.append(y[order[k]]); k+=1
    return np.array(picked)

def solve():
    rng = np.random.default_rng(0)
    t0=time.time()
    DEADLINE=t0+172.0
    try:
        xb=np.load("best.npy")
    except Exception:
        try: xb=np.load("record.npy")
        except Exception: xb=rng.standard_normal(size=(163,4))
    xb=np.asarray(xb,dtype=float)
    if xb.shape!=(163,4):
        tmp=rng.standard_normal(size=(163,4))
        if xb.ndim==2:
            a=min(xb.shape[0],163); b=min(xb.shape[1],4)
            tmp[:a,:b]=xb[:a,:b]
        xb=tmp
    xb=xb/np.linalg.norm(xb,axis=1,keepdims=True)
    best=max_cos(xb)
    x_best=xb.copy()
    x_cur=xb.copy(); cur=best
    n,d=163,4
    it=0
    # quick initial smooth+lp to ensure local opt (cheap)
    # skip if already at record to save time
    while time.time()<DEADLINE-16.0:
        it+=1
        base = x_best if (rng.random()<0.72 or it<4) else x_cur
        base_val = best if (base is x_best) else cur
        x_try=base.copy()
        r=rng.random()
        if r<0.30:
            sig=10.0**rng.uniform(-3.0,-1.1)
            x_try=base+sig*rng.standard_normal(size=(n,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        elif r<0.58:
            G=base@base.T; np.fill_diagonal(G,-2.0)
            thr=base_val-0.012
            deg=(G>thr).sum(axis=1)
            center=int(np.argmax(deg+rng.random(n)*0.7)) if rng.random()<0.65 else int(rng.integers(n))
            k=int(rng.integers(6,24))
            dots=base@base[center]
            idx=np.argsort(-dots)[:k]
            sig=float(rng.uniform(0.015,0.10))
            x_try[idx]=base[idx]+sig*rng.standard_normal(size=(k,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        elif r<0.82:
            G=base@base.T; np.fill_diagonal(G,-2.0)
            thr=base_val-0.015
            deg=(G>thr).sum(axis=1)
            m=int(rng.integers(2,9))
            # bias to low-degree (rattlers) or high-degree (jammed)
            if rng.random()<0.5:
                reloc=np.argsort(deg+rng.random(n)*0.4)[:m]
            else:
                reloc=np.argsort(-deg+rng.random(n)*0.4)[:m]
            keep=np.ones(n,bool); keep[reloc]=False
            voids=find_voids(base[keep],m,rng,S=16000)
            voids=voids+0.004*rng.standard_normal(size=voids.shape)
            voids/=np.linalg.norm(voids,axis=1,keepdims=True)
            x_try[reloc]=voids
            if rng.random()<0.5:
                x_try[keep]=base[keep]+0.008*rng.standard_normal(size=(n-m,d))
                x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        else:
            # big kill-and-fill: 8-14 random points to random far positions
            m=int(rng.integers(8,15))
            reloc=rng.choice(n,size=m,replace=False)
            keep=np.ones(n,bool); keep[reloc]=False
            voids=find_voids(base[keep],m,rng,S=20000)
            voids/=np.linalg.norm(voids,axis=1,keepdims=True)
            x_try[reloc]=voids
            # shake rest lightly
            x_try[keep]=base[keep]+0.005*rng.standard_normal(size=(n-m,d))
            x_try/=np.linalg.norm(x_try,axis=1,keepdims=True)
        if time.time()>DEADLINE-16.5:
            break
        # annealed smooth LBFGS: choose schedule randomly
        if rng.random()<0.5:
            temps=[0.008,0.003,0.0012]; mits=[120,150,180]
        else:
            temps=[0.005,0.0015]; mits=[150,220]
        # occasionally single-temp fast pass
        if rng.random()<0.2:
            temps=[0.002]; mits=[200]
        x_try=smooth_lbfgs(x_try,temps,mits)
        v=max_cos(x_try)
        # only LP-polish if promising (saves time)
        need_lp = (v < best+0.004)
        if need_lp and time.time()<DEADLINE-15.0:
            sec=float(rng.uniform(0.7,1.6))
            sec=min(sec,max(0.5,DEADLINE-15.0-time.time()))
            x_try=lp_polish(x_try,sec,deadline=DEADLINE-14.0)
            v=max_cos(x_try)
        if v<best-1e-12:
            x_best=x_try.copy(); best=v
            x_cur=x_try.copy(); cur=v
        elif v<cur+0.002 and rng.random()<0.5:
            x_cur=x_try.copy(); cur=v
        if it%15==0 and cur>best+0.0025:
            x_cur=x_best.copy(); cur=best
    remain=DEADLINE-time.time()-0.8
    if remain>2.0:
        x_best=lp_polish(x_best,min(remain,13.0),deadline=DEADLINE)
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best

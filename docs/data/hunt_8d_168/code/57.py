import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_X(seeds, freqs, m):
    ns = seeds.shape[0]
    ar = np.arange(m, dtype=np.float64)[:, None] * np.asarray(freqs, dtype=np.float64)[None, :]
    angs = 2.0 * np.pi * ar / float(m)
    C = np.cos(angs)
    S = np.sin(angs)
    Xr = np.empty((ns, m, 8), dtype=np.float64)
    for p in range(4):
        s0 = seeds[:, 2*p]
        s1 = seeds[:, 2*p+1]
        c = C[:, p]; s = S[:, p]
        Xr[:, :, 2*p] = s0[:, None]*c[None, :] - s1[:, None]*s[None, :]
        Xr[:, :, 2*p+1] = s0[:, None]*s[None, :] + s1[:, None]*c[None, :]
    X = Xr.reshape((ns*m, 8))
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    X = X / nrm
    return X

def _seed_quench(seeds0, freqs, m, steps, beta0, beta1, lr0, lr1):
    ns = seeds0.shape[0]
    seeds = seeds0.copy()
    seeds /= (np.linalg.norm(seeds, axis=1, keepdims=True)+1e-30)
    ar = np.arange(m, dtype=np.float64)[:, None] * np.asarray(freqs, dtype=np.float64)[None, :]
    angs = 2.0*np.pi*ar/float(m)
    C = np.cos(angs); S = np.sin(angs)
    mad = np.zeros_like(seeds); vad = np.zeros_like(seeds)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    n = ns*m
    best_val=1e9
    best_seeds=seeds.copy()
    for t in range(1, steps+1):
        frac=t/steps
        beta=np.exp(lb0+(lb1-lb0)*frac)
        lr=np.exp(ll0+(ll1-ll0)*frac)
        Xr=np.empty((ns,m,8),dtype=np.float64)
        for p in range(4):
            s0=seeds[:,2*p]; s1=seeds[:,2*p+1]
            c=C[:,p]; s=S[:,p]
            Xr[:,:,2*p]=s0[:,None]*c[None,:]-s1[:,None]*s[None,:]
            Xr[:,:,2*p+1]=s0[:,None]*s[None,:]+s1[:,None]*c[None,:]
        X=Xr.reshape((n,8))
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        gmax=float(G.max())
        if gmax<best_val:
            best_val=gmax; best_seeds=seeds.copy()
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        Rs=W.sum(axis=1,keepdims=True)+1e-30
        GradX=(W@X)/Rs
        dotx=(GradX*X).sum(axis=1,keepdims=True)
        GtX=GradX-dotx*X
        Gtr=GtX.reshape((ns,m,8))
        Gs=np.empty_like(seeds)
        for p in range(4):
            g0=Gtr[:,:,2*p]; g1=Gtr[:,:,2*p+1]
            c=C[:,p]; s=S[:,p]
            Gs[:,2*p]=((c[None,:]*g0+s[None,:]*g1).sum(axis=1))/float(m)
            Gs[:,2*p+1]=((-s[None,:]*g0+c[None,:]*g1).sum(axis=1))/float(m)
        dots=(Gs*seeds).sum(axis=1,keepdims=True)
        Gt=Gs-dots*seeds
        mad*=b1; mad+=(1.0-b1)*Gt
        vad*=b2; vad+=(1.0-b2)*(Gt*Gt)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(mad/bc1)/(np.sqrt(vad/bc2)+eps)
        seeds=seeds-lr*step
        seeds/=(np.linalg.norm(seeds,axis=1,keepdims=True)+1e-30)
    return best_seeds, best_val

def _adam_full(X, steps, beta0, beta1, lr0, lr1, t_lim=1e18):
    n,d=X.shape
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    best=X.copy(); best_m=_max_cos(X)
    for t in range(1,steps+1):
        if (t&63)==0 and time.time()>t_lim:
            break
        frac=t/steps
        beta=np.exp(lb0+(lb1-lb0)*frac)
        lr=np.exp(ll0+(ll1-ll0)*frac)
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        gmax=float(G.max())
        if gmax<best_m:
            best_m=gmax; best=X.copy()
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        Rs=W.sum(axis=1,keepdims=True)+1e-30
        Grad=(W@X)/Rs
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        m*=b1; m+=(1.0-b1)*Gt
        v*=b2; v+=(1.0-b2)*(Gt*Gt)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(m/bc1)/(np.sqrt(v/bc2)+eps)
        X=X-lr*step
        X/=(np.linalg.norm(X,axis=1,keepdims=True)+1e-30)
    return best

def _hard_polish(X, steps=250, lr0=0.0032, lr1=0.00012):
    n,d=X.shape
    for s in range(steps):
        frac=s/max(steps-1,1)
        tol=3e-4+(2e-6-3e-4)*frac
        lr=lr0*(lr1/lr0)**frac
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        mm=float(G.max())
        M=(G>(mm-tol)).astype(np.float64)
        np.fill_diagonal(M,0.0)
        cnt=M.sum(axis=1,keepdims=True)+1e-12
        Grad=(M@X)/cnt
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        nn=np.sqrt((Gt*Gt).sum(axis=1,keepdims=True))+1e-12
        active=(cnt[:,0]>0.5)
        step=np.zeros_like(Gt)
        step[active]=Gt[active]/nn[active]
        X=X-lr*step
        X/=(np.linalg.norm(X,axis=1,keepdims=True)+1e-30)
    return X

def _lp_polish(x, seconds, r0=1.2e-3):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    try:
        best=_max_cos(x)
    except Exception:
        return x
    radius=r0
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        c=(x@x.T)[iu]
        m=float(c.max())
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            radius*=0.5; continue
        if p>2500:
            idx=np.argsort(c[sel])[-2500:]
            I,J=I[idx],J[idx]
            cc=c[sel][idx]; p=2500
        else:
            cc=c[sel]
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius*=0.5; continue
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cc,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=_max_cos(y)
        if vv<best:
            x,best=y,vv
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _rand_freqs(rng,m):
    for _ in range(200):
        a=rng.randint(1,m,size=4)
        ok=True
        for v in a:
            if math.gcd(int(v),m)!=1:
                ok=False; break
        if not ok:
            continue
        sig=sorted([min(int(v),m-int(v)) for v in a])
        if len(set(sig))<3:
            continue
        if len(set(int(v) for v in a))<4:
            pass
        return np.array(a,dtype=np.float64)
    a=np.array([1,3,5,7],dtype=np.float64)%m
    a[a==0]=1
    return a

def solve():
    t0=time.time()
    BUDGET=175.0
    n,d=168,8
    rng=np.random.RandomState(20261103)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape!=(n,d):
                continue
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va=_max_cos(a)
            if va<best_val:
                best_val=va; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=rng.randn(n,d); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    incumbent=best.copy(); inc_val=best_val
    configs=[(84,2,22),(56,3,22),(42,4,18),(28,6,18),(24,7,14),(21,8,14)]
    cands=[]
    race_end=t0+55.0
    for (m,ns,nt) in configs:
        for k in range(nt):
            if time.time()>race_end:
                break
            freqs=_rand_freqs(rng,m)
            s0=rng.randn(ns,8); s0/=np.linalg.norm(s0,axis=1,keepdims=True)
            try:
                bs,bv=_seed_quench(s0,freqs,m,steps=320,beta0=30.0,beta1=600.0,lr0=0.025,lr1=0.002)
            except Exception:
                continue
            cands.append((bv,bs.copy(),freqs.copy(),m))
            if bv<inc_val-1e-9:
                try:
                    Xc=_build_X(bs,freqs,m)
                    vc=_max_cos(Xc)
                    if vc<inc_val:
                        pass
                except Exception:
                    pass
        if time.time()>race_end:
            break
    if len(cands)==0:
        return incumbent
    cands.sort(key=lambda z: z[0])
    top=cands[:6]
    # full quench + symmetry breaking
    # time left
    for rank,(bv,bs,freqs,m) in enumerate(top):
        for noise in (1e-3,1e-2):
            t_left=BUDGET-(time.time()-t0)
            if t_left<18.0:
                break
            try:
                X0=_build_X(bs,freqs,m)
            except Exception:
                continue
            # symmetry-breaking noise
            try:
                pert=rng.randn(n,d)*noise
                # tangent projection per row
                pert-=(pert*X0).sum(axis=1,keepdims=True)*X0
                Xb=X0+pert
                Xb/=np.linalg.norm(Xb,axis=1,keepdims=True)
            except Exception:
                Xb=X0.copy()
            # full-dimensional quench
            tlim=time.time()+min(9.0,t_left-10.0)
            try:
                Xq=_adam_full(Xb,steps=1400,beta0=30.0,beta1=700.0,lr0=0.02,lr1=0.0015,t_lim=tlim)
                Xq=_hard_polish(Xq,steps=220)
            except Exception:
                continue
            # LP polish with remaining slice
            t_left2=BUDGET-(time.time()-t0)
            lp_sec=min(8.0,max(2.0,t_left2/ ( (len(top)-rank)*2 +1) ))
            if t_left2>12.0:
                try:
                    Xq=_lp_polish(Xq,seconds=lp_sec)
                except Exception:
                    pass
            try:
                vq=_max_cos(Xq)
            except Exception:
                continue
            if vq<inc_val:
                inc_val=vq; incumbent=Xq.copy()
            if time.time()-t0>BUDGET-8.0:
                break
        if time.time()-t0>BUDGET-10.0:
            break
    # final safety LP on incumbent if time remains and we improved via sym path? keep cheap
    # ensure strictly valid
    incumbent/=np.linalg.norm(incumbent,axis=1,keepdims=True)
    # tiny shrink guard not needed; return
    return incumbent

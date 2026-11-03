import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_X(Rn, Fn):
    Xo = np.empty((91, 13), dtype=float)
    for j in range(7):
        r = Rn[j]
        for s in range(13):
            Xo[j*13+s] = np.roll(r, s)
    return np.vstack([Xo, Fn])

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    if t_end is None:
        t_end = start + seconds
    else:
        t_end = min(t_end, start + seconds)
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n*d+1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
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
            x,best,radius = y,v,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _sym_run(R,F,steps,b0,b1,lr,t_end):
    mR=np.zeros_like(R); vR=np.zeros_like(R)
    mF=np.zeros_like(F); vF=np.zeros_like(F)
    b1a,b2a,eps=0.9,0.999,1e-8
    lb0=np.log(b0); lb1=np.log(b1)
    for t in range(1,steps+1):
        if time.time()>t_end:
            break
        beta=float(np.exp(lb0+(lb1-lb0)*t/steps))
        nR=np.linalg.norm(R,axis=1,keepdims=True); nR[nR==0]=1.0
        nF=np.linalg.norm(F,axis=1,keepdims=True); nF[nF==0]=1.0
        Rn=R/nR; Fn=F/nF
        X=_build_X(Rn,Fn)
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        mx=float(G.max())
        E=np.exp(beta*(G-mx))
        np.fill_diagonal(E,0.0)
        s=E.sum()
        if not np.isfinite(s) or s==0:
            break
        W=E/s
        Grad=W@X
        Go=Grad[:91]; Gf=Grad[91:]
        gRn=np.zeros((7,13))
        for j in range(7):
            acc=np.zeros(13)
            for ss in range(13):
                acc+=np.roll(Go[j*13+ss],-ss)
            gRn[j]=acc
        # pullback through normalisation
        dotR=np.sum(gRn*Rn,axis=1,keepdims=True)
        gR=(gRn-dotR*Rn)/nR
        dotF=np.sum(Gf*Fn,axis=1,keepdims=True)
        gF=(Gf-dotF*Fn)/nF
        mR=b1a*mR+(1-b1a)*gR; vR=b2a*vR+(1-b2a)*(gR*gR)
        mF=b1a*mF+(1-b1a)*gF; vF=b2a*vF+(1-b2a)*(gF*gF)
        mRh=mR/(1-b1a**t); vRh=vR/(1-b2a**t)
        mFh=mF/(1-b1a**t); vFh=vF/(1-b2a**t)
        R-=lr*mRh/(np.sqrt(vRh)+eps)
        F-=lr*mFh/(np.sqrt(vFh)+eps)
    nR=np.linalg.norm(R,axis=1,keepdims=True); nR[nR==0]=1.0
    nF=np.linalg.norm(F,axis=1,keepdims=True); nF[nF==0]=1.0
    return R/nR, F/nF

def _full_run(Xraw,steps,b0,b1,lr,t_end):
    m=np.zeros_like(Xraw); v=np.zeros_like(Xraw)
    b1a,b2a,eps=0.9,0.999,1e-8
    lb0=np.log(b0); lb1=np.log(b1)
    for t in range(1,steps+1):
        if time.time()>t_end:
            break
        beta=float(np.exp(lb0+(lb1-lb0)*t/steps))
        nr=np.linalg.norm(Xraw,axis=1,keepdims=True); nr[nr==0]=1.0
        X=Xraw/nr
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        mx=float(G.max())
        E=np.exp(beta*(G-mx))
        np.fill_diagonal(E,0.0)
        s=E.sum()
        if not np.isfinite(s) or s==0:
            break
        W=E/s
        Grad=W@X
        dot=np.sum(Grad*X,axis=1,keepdims=True)
        g=(Grad-dot*X)/nr
        m=b1a*m+(1-b1a)*g; v=b2a*v+(1-b2a)*(g*g)
        mh=m/(1-b1a**t); vh=v/(1-b2a**t)
        step=lr*mh/(np.sqrt(vh)+eps)
        sn=np.linalg.norm(step,axis=1,keepdims=True)+1e-12
        cap=np.minimum(1.0,0.05/sn)
        Xraw-=step*cap
    nr=np.linalg.norm(Xraw,axis=1,keepdims=True); nr[nr==0]=1.0
    return Xraw/nr

def solve():
    t0=time.time()
    deadline=t0+172.0
    rng=np.random.default_rng(0)
    try:
        xb=np.load("best.npy").astype(float)
        xb=xb/np.linalg.norm(xb,axis=1,keepdims=True)
        best_val=_max_cos(xb)
    except Exception:
        xb=rng.standard_normal(size=(95,13))
        xb/=np.linalg.norm(xb,axis=1,keepdims=True)
        best_val=_max_cos(xb)
    try:
        xr=np.load("record.npy").astype(float)
        xr=xr/np.linalg.norm(xr,axis=1,keepdims=True)
        vr=_max_cos(xr)
        if vr<best_val:
            best_val,xb=vr,xr.copy()
    except Exception:
        pass
    x_keep=xb.copy(); v_keep=best_val
    # --- symmetry-constrained search ---
    sym_best_X=None; sym_best_v=np.inf
    # time budget: leave ~70s for full+LP
    sym_end=deadline-70.0
    seeds=[0,1,7]
    lrs=[0.02,0.03,0.015]
    for si,sd in enumerate(seeds):
        if time.time()>sym_end-2:
            break
        r2=np.random.default_rng(sd)
        R=r2.standard_normal(size=(7,13))
        F=r2.standard_normal(size=(4,13))
        steps=1800
        Rn,Fn=_sym_run(R,F,steps,30.0,1000.0,lrs[si],sym_end)
        X=_build_X(Rn,Fn)
        vv=_max_cos(X)
        # duplicate check: internal orbit coincidence gives ~1
        if vv<sym_best_v:
            sym_best_v=vv; sym_best_X=X.copy()
    # second family attempt if duplicates/bad: partially constrained orbit variant
    # (use 6x13=78 + 17 free) briefly if sym very bad
    alt_X=None; alt_v=np.inf
    if sym_best_X is None or sym_best_v>0.45:
        if time.time()<sym_end-2:
            r2=np.random.default_rng(123)
            R=r2.standard_normal(size=(6,13))
            F=r2.standard_normal(size=(17,13))
            # quick full run from random as fallback alt
            X0=np.vstack([np.vstack([np.roll(R[j]/np.linalg.norm(R[j]),s) for s in range(13)]) for j in range(6)] + [F/np.linalg.norm(F,axis=1,keepdims=True)])
            # X0 is 78+17=95
            alt_X=_full_run(X0.copy(),1200,30.0,800.0,0.02,sym_end)
            alt_v=_max_cos(alt_X)
            if alt_v<sym_best_v:
                sym_best_v=alt_v; sym_best_X=alt_X.copy()
    cand_list=[]
    if sym_best_X is not None:
        cand_list.append(sym_best_X)
    # --- break symmetry: perturb + full refine ---
    full_end=deadline-28.0
    refined_best=None; refined_v=np.inf
    for ci,Xc in enumerate(cand_list):
        if time.time()>full_end-3:
            break
        for sigma in (0.003, 0.008):
            if time.time()>full_end-3:
                break
            r2=np.random.default_rng(1000+ci*10+int(sigma*1e6)%97)
            noise=r2.standard_normal(size=Xc.shape)
            noise-=np.sum(noise*Xc,axis=1,keepdims=True)*Xc
            nn=np.linalg.norm(noise,axis=1,keepdims=True)+1e-12
            noise/=nn
            Xp=Xc+sigma*noise
            Xp/=np.linalg.norm(Xp,axis=1,keepdims=True)
            Xr=_full_run(Xp.copy(),2600,80.0,1500.0,0.012,full_end)
            vv=_max_cos(Xr)
            if vv<refined_v:
                refined_v=vv; refined_best=Xr.copy()
    # LP polish refined
    if refined_best is not None:
        remain=deadline-time.time()-1.0
        if remain>3:
            refined_best=_lp_polish(refined_best,seconds=min(remain-0.5,22.0),t_end=deadline-0.8)
            refined_v=_max_cos(refined_best)
        if refined_v<v_keep-1e-12:
            v_keep,x_keep=refined_v,refined_best.copy()
    # also LP-polish the kept best briefly if time remains
    remain=deadline-time.time()-0.8
    if remain>2.0:
        # only polish if we have slack; avoid destroying record with timeout
        if v_keep>=best_val-1e-12:
            # try polishing current best a little
            xp=_lp_polish(x_keep,seconds=min(remain,6.0),t_end=deadline-0.5)
            vp=_max_cos(xp)
            if vp<v_keep:
                v_keep,x_keep=vp,xp
    x_keep/=np.linalg.norm(x_keep,axis=1,keepdims=True)
    return x_keep

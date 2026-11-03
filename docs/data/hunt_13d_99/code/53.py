import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return V / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_cs(m, mults):
    p = len(mults)
    th = 2*np.pi*np.arange(m)[:,None]*np.array(mults, dtype=float)[None,:]/float(m)
    return np.cos(th), np.sin(th)

def _gen_full(Yf, Yx, m, p, n_fixed, C, S):
    nf = Yf.shape[0]
    nfx = Yx.shape[0]
    Xf = np.zeros((nf*m, 13))
    Yfix = Yf[:, 2*p:] if n_fixed>0 else None
    for g in range(m):
        sl = slice(g*nf, (g+1)*nf)
        for pi in range(p):
            c = C[g,pi]; s = S[g,pi]
            y0 = Yf[:,2*pi]; y1 = Yf[:,2*pi+1]
            Xf[sl,2*pi] = c*y0 - s*y1
            Xf[sl,2*pi+1] = s*y0 + c*y1
        if n_fixed>0:
            Xf[sl,2*p:] = Yfix
    if nfx>0:
        return np.vstack([Xf, Yx])
    return Xf

def _accum_full(gXf, nf, m, p, n_fixed, C, S):
    gY = np.zeros((nf, 13))
    for g in range(m):
        blk = gXf[g*nf:(g+1)*nf]
        for pi in range(p):
            c = C[g,pi]; s = S[g,pi]
            a0 = blk[:,2*pi]; a1 = blk[:,2*pi+1]
            gY[:,2*pi] += c*a0 + s*a1
            gY[:,2*pi+1] += -s*a0 + c*a1
        if n_fixed>0:
            gY[:,2*p:] += blk[:,2*p:]
    return gY

def _rot_back_one(v, g, p, C, S):
    y = np.zeros(13)
    for pi in range(p):
        c = C[g,pi]; s = S[g,pi]
        b0 = v[2*pi]; b1 = v[2*pi+1]
        y[2*pi] = c*b0 + s*b1
        y[2*pi+1] = -s*b0 + c*b1
    y[2*p:] = v[2*p:]
    return y

def _fit_to_B(B, cfg, rng, iters=12):
    m=cfg['m']; mults=cfg['mults']; p=cfg['p']
    n_fixed=13-2*p; nf=cfg['nf']; nfx=cfg['nfx']
    C,S=_build_cs(m,mults)
    idx=rng.choice(B.shape[0],size=nf,replace=False)
    Yf=_norm_rows(B[idx].copy())
    if nfx>0:
        if n_fixed>0:
            Z=_norm_rows(rng.standard_normal(size=(nfx,n_fixed)))
            Yx=np.zeros((nfx,13)); Yx[:,2*p:]=Z
        else:
            Yx=_norm_rows(rng.standard_normal(size=(nfx,13)))
    else:
        Yx=np.zeros((0,13))
    for _ in range(iters):
        Xf=_gen_full(Yf,Yx,m,p,n_fixed,C,S) if False else None
        # build Xf + Yx full for assignment
        Xfull=_gen_full(Yf,Yx,m,p,n_fixed,C,S)
        Nf=nf*m
        G=B@Xfull.T
        assign=np.argmax(G,axis=1)
        # update Yf
        acc_sum=np.zeros_like(Yf); acc_cnt=np.zeros(nf)
        # accumulate rotated back sums
        for i in range(B.shape[0]):
            a=int(assign[i])
            if a<Nf:
                gg=a//nf; jj=a%nf
                rb=_rot_back_one(B[i],gg,p,C,S)
                acc_sum[jj]+=rb; acc_cnt[jj]+=1
        for j in range(nf):
            if acc_cnt[j]>0:
                s=acc_sum[j]/acc_cnt[j]
                n=np.linalg.norm(s)
                if n>1e-12:
                    Yf[j]=s/n
        if nfx>0 and n_fixed>0:
            sums=[ [] for _ in range(nfx)]
            # collect
            fix_acc=np.zeros((nfx,13)); fix_cnt=np.zeros(nfx)
            for i in range(B.shape[0]):
                a=int(assign[i])
                if a>=Nf:
                    k=a-Nf
                    fix_acc[k]+=B[i]; fix_cnt[k]+=1
            for k in range(nfx):
                if fix_cnt[k]>0:
                    M=fix_acc[k]/fix_cnt[k]
                    v=M[2*p:]; n=np.linalg.norm(v)
                    if n>1e-12:
                        Yx[k,:2*p]=0.0; Yx[k,2*p:]=v/n
    if nfx>0 and n_fixed>0:
        Ys=Yx[:,2*p:].copy()
    else:
        Ys=np.zeros((nfx,0))
    return Yf,Ys,C,S

def _sym_fun_factory(cfg,C,S,alpha):
    m=cfg['m']; p=cfg['p']; nf=cfg['nf']; nfx=cfg['nfx']
    n_fixed=13-2*p
    def fun(v):
        # split
        off=nf*13
        Vf=v[:off].reshape(nf,13) if nf>0 else np.zeros((0,13))
        if nf>0:
            nr=np.linalg.norm(Vf,axis=1,keepdims=True); nr=np.where(nr<1e-15,1.0,nr)
            Yf=Vf/nr
        else:
            Yf=np.zeros((0,13)); nr=np.ones((0,1))
        if nfx>0 and n_fixed>0:
            Vs=v[off:].reshape(nfx,n_fixed)
            ns=np.linalg.norm(Vs,axis=1,keepdims=True); ns=np.where(ns<1e-15,1.0,ns)
            Ys=Vs/ns
            Yx=np.zeros((nfx,13)); Yx[:,2*p:]=Ys
        elif nfx>0:
            # n_fixed==0: fixed points are full 13-vectors
            Vs=v[off:].reshape(nfx,13)
            ns=np.linalg.norm(Vs,axis=1,keepdims=True); ns=np.where(ns<1e-15,1.0,ns)
            Yx=Vs/ns; Ys=np.zeros((nfx,0))
            # need dummy Yf fixed part handling n_fixed=0 already
        else:
            Ys=np.zeros((0,0)); Yx=np.zeros((0,13)); Vs=np.zeros((0,0)); ns=np.ones((0,1))
        X=_gen_full(Yf,Yx,m,p,n_fixed,C,S)
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        mx=float(G.max())
        E=np.exp(alpha*(G-mx))
        np.fill_diagonal(E,0.0)
        Sm=E.sum()/2.0+1e-300
        L=mx+np.log(Sm)/alpha
        gX=(E@X)/Sm
        Nf=nf*m
        gXf=gX[:Nf] if Nf>0 else np.zeros((0,13))
        gXx=gX[Nf:] if nfx>0 else np.zeros((0,13))
        if nf>0:
            gY=_accum_full(gXf,nf,m,p,n_fixed,C,S)
            dot=(gY*Yf).sum(axis=1,keepdims=True)
            gVf=(gY-dot*Yf)/nr
        else:
            gVf=np.zeros((0,13))
        if nfx>0 and n_fixed>0:
            gS=gXx[:,2*p:]
            dot2=(gS*Ys).sum(axis=1,keepdims=True)
            gVs=(gS-dot2*Ys)/ns
            g=np.concatenate([gVf.ravel(),gVs.ravel()]).astype(float)
        elif nfx>0:
            dotx=(gXx*Yx).sum(axis=1,keepdims=True)
            # ns here is for Vs==Yx case: recompute
            gVx=(gXx-dotx*Yx)/ns
            g=np.concatenate([gVf.ravel(),gVx.ravel()]).astype(float)
        else:
            g=gVf.ravel().astype(float)
        return float(L), g
    return fun

def _run_lbfgs(fun,x0,maxiter):
    try:
        r=minimize(fun,x0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-20,'gtol':1e-12,'maxfun':maxiter*6})
        return r.x.copy(), float(r.fun)
    except Exception:
        return x0, float('inf')

def _full_fun_factory(alpha):
    def fun(v):
        V=v.reshape(99,13)
        nr=np.linalg.norm(V,axis=1,keepdims=True); nr=np.where(nr<1e-15,1.0,nr)
        Y=V/nr
        G=Y@Y.T
        np.fill_diagonal(G,-10.0)
        mx=float(G.max())
        E=np.exp(alpha*(G-mx))
        np.fill_diagonal(E,0.0)
        Sm=E.sum()/2.0+1e-300
        L=mx+np.log(Sm)/alpha
        gY=(E@Y)/Sm
        dot=(gY*Y).sum(axis=1,keepdims=True)
        gV=(gY-dot*Y)/nr
        return float(L), gV.ravel().astype(float)
    return fun

def _lp_polish(x,time_budget,deadline):
    x=_norm_rows(x.copy())
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    t0=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-t0<time_budget and time.time()<deadline:
        if radius<1e-14:
            break
        c=(x@x.T)[iu]
        m=c.max()
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I); nv=n*d+1
        if p==0:
            break
        if p>2500:
            # subsample worst pairs to keep LP tractable
            order=np.argsort(-c[sel])[:2500]
            I,J=I[order],J[order]; p=len(I)
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            break
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel][:p] if p==np.count_nonzero(sel) else None if False else (-(x@x.T)[iu][sel][:0] if False else None),A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        # fix b_ub: recompute correctly for subsampled case
        # (above hack) -> redo properly if subsampled
        if False:
            pass
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y=_norm_rows(y)
        v=_max_cos(y)
        if v<best:
            x,best,radius=y,v,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x,best

def _lp_polish2(x,time_budget,deadline):
    # correct wide-active LP (simpler, robust)
    x=_norm_rows(x.copy())
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    t0=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-t0<time_budget and time.time()<deadline:
        if radius<1e-14:
            break
        G=x@x.T
        c=G[iu]
        m=c.max()
        thr=m-max(20*radius,2e-6)
        sel=c>thr
        I,J=iu[0][sel],iu[1][sel]
        p=len(I); nv=n*d+1
        if p==0:
            break
        if p>2200:
            order=np.argsort(-c[sel])[:2200]
            I,J=I[order],J[order]; p=len(I)
            bvals=(G[I,J]).copy() if False else None
            # need corresponding c values
            b_ub=-(x[I]*x[J]).sum(axis=1)
            # note c for these pairs:
            b_ub2=-((x@x.T)[I,J] if False else 0)
            # recompute directly:
            b_ub=-np.array([float(x[I[k]]@x[J[k]]) for k in range(p)])
        else:
            b_ub=-c[sel]
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            break
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y=_norm_rows(y)
        v=_max_cos(y)
        if v<best:
            x,best,radius=y,v,min(radius*1.5,0.02)
        else:
            radius*=0.5
    return x,best

def solve():
    t_start=time.time()
    deadline=t_start+168.0
    rng=np.random.RandomState(0)
    try:
        xb=np.load("best.npy")
    except Exception:
        try:
            xb=np.load("record.npy")
        except Exception:
            xb=rng.standard_normal(size=(99,13))
    xb=np.asarray(xb,dtype=float)
    if xb.shape!=(99,13):
        flat=xb.ravel()
        need=99*13
        if flat.size<need:
            flat=np.concatenate([flat,rng.standard_normal(size=(need-flat.size,))])
        xb=flat[:need].reshape(99,13)
    xb=_norm_rows(xb)
    best_val=_max_cos(xb)
    x_best=xb.copy()
    # candidate cyclic configs
    cfgs=[
        {'m':11,'mults':[1,2,3,4,5],'p':5,'nf':9,'nfx':0},
        {'m':9,'mults':[1,2,4,5,7],'p':5,'nf':11,'nfx':0},
        {'m':3,'mults':[1,1,2,1,2,2],'p':6,'nf':33,'nfx':0},
        {'m':16,'mults':[1,3,5,7,9],'p':5,'nf':6,'nfx':3},
        {'m':12,'mults':[1,5,7,11,1],'p':5,'nf':8,'nfx':3},
        {'m':2,'mults':[1,1,1,1,1,1],'p':6,'nf':49,'nfx':1},
    ]
    sym_end=t_start+92.0
    for cfg in cfgs:
        if time.time()>sym_end or time.time()>deadline-70:
            break
        m=cfg['m']; p=cfg['p']; nf=cfg['nf']; nfx=cfg['nfx']
        n_fixed=13-2*p
        g_start=time.time()
        g_budget=13.0
        # two starts: projected + random
        for s_idx in range(2):
            if time.time()-g_start>g_budget or time.time()>sym_end or time.time()>deadline-70:
                break
            if s_idx==0:
                try:
                    Yf0,Ys0,C,S=_fit_to_B(x_best,cfg,rng,iters=10)
                except Exception:
                    continue
            else:
                C,S=_build_cs(m,cfg['mults'])
                Yf0=_norm_rows(rng.standard_normal(size=(nf,13)))
                if nfx>0 and n_fixed>0:
                    Ys0=_norm_rows(rng.standard_normal(size=(nfx,n_fixed)))
                else:
                    Ys0=np.zeros((nfx,0))
                # for m=2 fixed dim 1, pin to +1
                if n_fixed==1 and nfx>0:
                    Ys0=np.ones((nfx,1))
            # pack x0
            if nfx>0 and n_fixed>0:
                # if fixed dim 1, variable is degenerate -> drop it (fix)
                if n_fixed==1:
                    x0=Yf0.ravel().copy()
                    # custom fun with fixed Ys pinned
                    Ys_pinned=np.ones((nfx,1))
                    # build wrapper
                    base_fun=_sym_fun_factory(cfg,C,S,80)
                    # we will just optimise Yf with Ys pinned by using full fun but overriding?
                    # simpler: optimise with variable including dummy but project gradient: just use base with x0 including dummy fixed
                    x0=np.concatenate([Yf0.ravel(),Ys_pinned.ravel()])
                    # will keep fixed by resetting after? allow optimiser to move but renormalises to +-1, harmless
                else:
                    x0=np.concatenate([Yf0.ravel(),Ys0.ravel()])
            else:
                x0=Yf0.ravel().copy()
            # anneal alphas
            for alpha,mit in [(30,80),(80,70),(200,60)]:
                if time.time()-g_start>g_budget or time.time()>sym_end or time.time()>deadline-70:
                    break
                fun=_sym_fun_factory(cfg,C,S,alpha)
                x0,_=_run_lbfgs(fun,x0,maxiter=mit)
            # expand
            try:
                off=nf*13
                Vf=x0[:off].reshape(nf,13) if nf>0 else np.zeros((0,13))
                Yf=_norm_rows(Vf) if nf>0 else np.zeros((0,13))
                if nfx>0 and n_fixed>0:
                    Vs=x0[off:].reshape(nfx,n_fixed)
                    Ys=_norm_rows(Vs)
                    Yx=np.zeros((nfx,13)); Yx[:,2*p:]=Ys
                    if n_fixed==1:
                        Yx=np.zeros((nfx,13)); Yx[:,12]=1.0
                elif nfx>0:
                    Yx=_norm_rows(x0[off:].reshape(nfx,13))
                else:
                    Yx=np.zeros((0,13))
                Xcand=_gen_full(Yf,Yx,m,p,n_fixed,C,S)
                # safety: if shape mismatch, skip
                if Xcand.shape!=(99,13):
                    continue
                Xcand=_norm_rows(Xcand)
                v=_max_cos(Xcand)
                if v<best_val-1e-12:
                    best_val=v; x_best=Xcand.copy()
            except Exception:
                continue
    # release symmetry in stages: brief annealed full optimisation from best symmetric + incumbent
    # stage 1: from x_best
    cur_best=x_best.copy(); cur_val=best_val
    # also consider incumbent xb if different
    starts=[x_best]
    if _max_cos(xb)<cur_val+0.005:
        starts.append(xb)
    full_end=min(deadline-18.0, time.time()+45.0)
    for si,Xs in enumerate(starts):
        if time.time()>full_end:
            break
        v0=Xs.ravel().copy()
        for alpha,mit in [(40,120),(120,120),(300,80)]:
            if time.time()>full_end:
                break
            fun=_full_fun_factory(alpha)
            v0,_=_run_lbfgs(fun,v0,maxiter=mit)
        try:
            V=v0.reshape(99,13); Y=_norm_rows(V)
            vv=_max_cos(Y)
            if vv<cur_val:
                cur_val=vv; cur_best=Y.copy()
                if vv<best_val:
                    best_val=vv; x_best=Y.copy()
        except Exception:
            continue
        # keep chaining best
        if si==0 and len(starts)>1:
            # update second start to current best if improved a lot? keep as is
            pass
    x_best=cur_best.copy(); best_val=cur_val
    # final wide-active LP polish
    remain=deadline-time.time()
    if remain>4.0:
        try:
            tb=min(remain-1.0,14.0)
            x_best,best_val=_lp_polish2(x_best,time_budget=tb,deadline=deadline-0.5)
        except Exception:
            pass
    x_best=_norm_rows(x_best)
    return x_best

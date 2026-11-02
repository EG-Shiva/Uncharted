import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _nr(x):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-15)
    return x

def _tangent_bases(x):
    n, d = x.shape
    dm = d - 1
    Q = np.zeros((n, d, dm))
    for i in range(n):
        # null space of 1x21 row
        _, _, Vt = np.linalg.svd(x[i:i+1, :], full_matrices=True)
        # Vt is 21x21, first row spans x, rest is nullspace
        Q[i] = Vt[1:, :].T
    return Q

def _build_J(x, Q, I, J):
    n, d = x.shape
    dm = d - 1
    p = len(I)
    Ntan = n * dm
    Jmat = np.zeros((p, Ntan), dtype=np.float64)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        a = Q[i].T @ x[j]  # dm
        b = Q[j].T @ x[i]
        Jmat[k, i*dm:(i+1)*dm] = a
        Jmat[k, j*dm:(j+1)*dm] = b
    return Jmat

def _refresh(x, radius_lp=5e-4):
    x = _nr(x)
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    cfull = G[iu]
    # active within 1e-7 for topology
    sel tight = cfull > m - 1e-7
    # for stress LP use near-active
    thr = m - max(20*radius_lp, 1e-7)
    sel2 = cfull > thr
    # cap to largest 700
    if np.count_nonzero(sel2) > 700:
        idx = np.where(sel2)[0]
        order = np.argsort(-cfull[idx])[:700]
        nsel = np.zeros_like(sel2)
        nsel[idx[order]] = True
        sel2 = nsel
    I = iu[0][sel2]; J = iu[1][sel2]
    Q = _tangent_bases(x)
    Jmat = _build_J(x, Q, I, J)
    p = len(I)
    w = np.ones(p)/max(p,1)
    # LP dual multipliers as equilibrium stresses
    try:
        if p>0 and p<=900:
            nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel2], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius_lp,radius_lp)]*(n*d)+[(None,None)], method="highs")
            if res.status==0 and hasattr(res,'ineqlin') and res.ineqlin is not None:
                try:
                    marg = np.asarray(res.ineqlin.marginals, dtype=float).ravel()
                    ww = np.abs(marg)
                    ww = np.where(np.isfinite(ww), ww, 0.0)
                    s = ww.sum()
                    if s>1e-12 and np.all(np.isfinite(ww)):
                        w = ww/s
                    else:
                        # fallback: degree weighting
                        pass
                except Exception:
                    pass
    except Exception:
        pass
    # tight contact indices for output (subset of I,J with slack<1e-7)
    return Q, Jmat, I, J, w, m

def _masked_quench(x0, excl, steps, b0, b1, lr0, lr1, rng, noise=0.0):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    mom = 0.85
    for s in range(steps):
        f = s/max(steps-1,1)
        beta = b0+(b1-b0)*f
        lr = lr0+(lr1-lr0)*f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        if excl is not None:
            G[excl] = -10.0
        m = G.max()
        W = np.exp(beta*(G-m))
        np.fill_diagonal(W, 0.0)
        if excl is not None:
            W[excl] = 0.0
        S = W.sum()+1e-30
        Gw = (W @ x)/S
        dot = np.sum(Gw*x, axis=1, keepdims=True)
        Gt = Gw-dot*x
        nn = np.linalg.norm(Gt,axis=1,keepdims=True)+1e-12
        Gt = Gt/nn
        if noise>0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz*x,axis=1,keepdims=True)*x
            nz /= (np.linalg.norm(nz,axis=1,keepdims=True)+1e-12)
            Gt = (1.0-noise)*Gt+noise*nz
            Gt /= (np.linalg.norm(Gt,axis=1,keepdims=True)+1e-12)
        vel = mom*vel+(1.0-mom)*Gt
        x = x - lr*vel
        x /= (np.linalg.norm(x,axis=1,keepdims=True)+1e-15)
    return x

def _smooth_slsqp(x0, Q, Ia, Ja, beta, radius, maxiter=60):
    # epigraph-free smooth max over allowed pairs, trust box in tangent coords
    n,d = x0.shape
    dm = d-1
    Ntan = n*dm
    # precompute nothing
    def fun_grad(v):
        # v: Ntan
        V = v.reshape(n,dm)
        # dx = Q @ V per point: (n,d)
        dx = np.einsum('nik,nk->ni', Q, V)
        y = x0+dx
        y /= (np.linalg.norm(y,axis=1,keepdims=True)+1e-15)
        dots = np.sum(y[Ia]*y[Ja],axis=1)
        m = dots.max()
        # log-sum-exp
        e = np.exp(beta*(dots-m))
        S = e.sum()+1e-30
        f = m + np.log(S+1e-30)/beta
        # grad w.r.t dots
        pw = e/S  # p
        # grad w.r.t y: for each pair k, contributes pw[k] to y[Ia] via y[Ja] and vice versa
        Gy = np.zeros_like(y)
        # accumulate
        # use bincount style loop (p<=800)
        for k in range(len(Ia)):
            ik=int(Ia[k]); jk=int(Ja[k]); wk=float(pw[k])
            Gy[ik]+=wk*y[jk]
            Gy[jk]+=wk*y[ik]
        # chain through normalization: dy/dx: (I - y y^T)/norm ; norm~1
        # approximate norm factor
        # need norms before normalize
        # recompute norms
        yn = np.linalg.norm((x0+dx),axis=1,keepdims=True)+1e-15
        # Gy is dL/dy, dL/dx = (I-y y^T)/yn * Gy
        Gx = (Gy - np.sum(Gy*y,axis=1,keepdims=True)*y)/yn
        # dL/dV: Q^T Gx per point
        Gv = np.einsum('nik,ni->nk', Q, Gx).ravel()
        return float(f), Gv.astype(float)
    try:
        v0 = np.zeros(Ntan)
        bnds = [(-radius,radius)]*Ntan
        res = minimize(lambda v: fun_grad(v)[0], v0, jac=lambda v: fun_grad(v)[1],
                       method="SLSQP", bounds=bnds, options={"maxiter":maxiter,"ftol":1e-12,"disp":False})
        if not np.all(np.isfinite(res.x)):
            return x0
        V = res.x.reshape(n,dm)
        dx = np.einsum('nik,nk->ni', Q, V)
        y = _nr(x0+dx)
        return y
    except Exception:
        return x0

def _lp_polish(x, deadline, radius_init=1e-3, cap=700):
    x=_nr(x); n,d=x.shape
    best=_mc(x); radius=radius_init
    iu=np.triu_indices(n,1); niter=0
    while time.time()<deadline and radius>1e-13 and niter<40:
        niter+=1
        G=x@x.T; cfull=G[iu]; m=cfull.max()
        sel=cfull>m-max(20*radius,1e-9)
        cnt=int(np.count_nonzero(sel))
        if cnt==0: break
        if cnt>cap:
            idx=np.where(sel)[0]; order=np.argsort(-cfull[idx])[:cap]
            nsel=np.zeros_like(sel); nsel[idx[order]]=True; sel=nsel
        csel=cfull[sel]; I,J=iu[0][sel],iu[1][sel]; p=len(I); nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception: break
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d); y=_nr(y); v=_mc(y)
        if v<best:
            x,best=y,v; radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return x

def solve():
    rng=np.random.default_rng(0)
    t0=time.time()
    deadline=t0+171.0
    search_end=t0+120.0
    hard_end=t0+149.0
    try:
        xb=np.load("best.npy")
    except Exception:
        try: xb=np.load("record.npy")
        except Exception: xb=rng.standard_normal((55,21))
    xb=np.asarray(xb,dtype=np.float64)
    if xb.shape!=(55,21):
        xb=rng.standard_normal((55,21))
    xb=_nr(xb)
    best=_mc(xb)
    x_best=xb.copy()
    n,d=55,21
    dm=d-1
    Ntan=n*dm
    # initial stress info
    try:
        Q,Jmat,Ia,Ja,w,m0=_refresh(x_best)
    except Exception:
        Q=_tangent_bases(x_best); Jmat=np.zeros((0,Ntan)); Ia=np.array([],dtype=int); Ja=np.array([],dtype=int); w=np.array([]); m0=best
    def do_refresh():
        nonlocal Q,Jmat,Ia,Ja,w,m0
        try:
            Q,Jmat,Ia,Ja,w,m0=_refresh(x_best)
        except Exception:
            pass
    trial=0
    n_refresh_gap=12
    steps_list=[220,300,360]
    while time.time()<search_end:
        if time.time()>hard_end-25:
            break
        trial+=1
        if trial%n_refresh_gap==0:
            do_refresh()
        p=len(Ia)
        # select high-stress subset S to relax
        S_idx=np.array([],dtype=int)
        try:
            if p>0:
                order=np.argsort(-w)
                topk=min(p,40)
                pool=order[:topk]
                # random subset size 3..12
                ksize=3+int(rng.integers(0,10))
                ksize=min(ksize,len(pool))
                # bias to largest stress: sample with prob ~ w
                pw=w[pool]; pw=pw/pw.sum() if pw.sum()>0 else None
                if pw is not None and rng.random()<0.7:
                    S_idx=rng.choice(pool,size=ksize,replace=False,p=pw)
                else:
                    S_idx=rng.choice(pool,size=ksize,replace=False)
            else:
                S_idx=np.array([],dtype=int)
        except Exception:
            S_idx=np.array([],dtype=int)
        # build J_rest excluding S
        mask_keep=np.ones(p,dtype=bool)
        if len(S_idx)>0:
            mask_keep[S_idx]=False
        try:
            if p>0 and np.count_nonzero(mask_keep)>0:
                Jr=Jmat[mask_keep]
                pr=Jr.shape[0]
                # JJt with damping
                JJt=Jr@Jr.T+1e-8*np.eye(pr)
                # random tangent vector
                r=rng.standard_normal(Ntan)
                # nullspace projection: r_null = r - Jr^T inv(JJt) Jr r
                Jr_r=Jr@r
                try:
                    c=np.linalg.solve(JJt,Jr_r)
                except Exception:
                    c=np.linalg.lstsq(JJt,Jr_r,rcond=None)[0]
                r_null=r-Jr.T@c
                nn=np.linalg.norm(r_null)+1e-12
                r_null/=nn
                # sometimes mix two null modes
                if rng.random()<0.4:
                    r2=rng.standard_normal(Ntan)
                    Jr2=Jr@r2
                    try: c2=np.linalg.solve(JJt,Jr2)
                    except Exception: c2=np.linalg.lstsq(JJt,Jr2,rcond=None)[0]
                    r2n=r2-Jr.T@c2; r2n/= (np.linalg.norm(r2n)+1e-12)
                    alpha=rng.random()
                    r_null=alpha*r_null+(1-alpha)*r2n
                    r_null/=(np.linalg.norm(r_null)+1e-12)
                # lowest-eigenvector character: also consider small singular direction via Jr^T c with small coeff? r_null already lowest (zero)
                step_choices=np.array([0.004,0.010,0.020,0.035,0.055])
                step=float(rng.choice(step_choices))* (0.7+0.6*rng.random())
                Vnull=r_null.reshape(n,dm)
                dx=np.einsum('nik,nk->ni',Q,Vnull)
                # scale so max point move ~ step
                mx=np.linalg.norm(dx,axis=1).max()+1e-12
                dx=dx/mx*step
            else:
                dx=0.02*rng.standard_normal((n,d))
                # project tangent
                dx=dx-np.sum(dx*x_best,axis=1,keepdims=True)*x_best
                nn=np.linalg.norm(dx)+1e-12
                dx=dx/nn*0.02
        except Exception:
            dx=0.02*rng.standard_normal((n,d))
            dx=dx-np.sum(dx*x_best,axis=1,keepdims=True)*x_best
        x_init=_nr(x_best+dx)
        # exclusion mask for quench (allow S to worsen)
        excl=None
        try:
            if len(S_idx)>0:
                excl=np.zeros((n,n),dtype=bool)
                for k in S_idx:
                    i=int(Ia[k]); j=int(Ja[k])
                    excl[i,j]=True; excl[j,i]=True
        except Exception:
            excl=None
        steps=int(rng.choice(steps_list))
        b0=float(rng.choice([60.0,120.0,200.0]))
        b1=float(rng.choice([600.0,900.0,1200.0]))
        lr0=float(rng.choice([0.006,0.009,0.013]))
        lr1=lr0*0.3
        lam_choice=steps  # unused
        try:
            x_q=_masked_quench(x_init,excl,steps=steps,b0=b0,b1=b1,lr0=lr0,lr1=lr1,rng=rng,noise=0.03 if rng.random()<0.5 else 0.0)
        except Exception:
            continue
        v=_mc(x_q)
        # epigraph SLSQP/trust-constr step using only active+near-active (excluding S) if promising
        if v<best+2e-4 and time.time()<search_end-5:
            try:
                Gq=x_q@x_q.T; np.fill_diagonal(Gq,-2.0)
                mq=float(Gq.max())
                iu=np.triu_indices(n,1)
                cf=Gq[iu]
                near=cf>mq-5e-4
                # remove S pairs from allowed? S refers to old indexing; rebuild exclusion by value: allow all near except those with excl mask
                Iall=iu[0][near]; Jall=iu[1][near]
                if len(Iall)>800:
                    idx=np.argsort(-cf[near])[:800]
                    Iall=Iall[idx]; Jall=Jall[idx]
                # filter out excluded (if excl defined on old config, map by indices? keep all for new config but drop pairs that were in S index set by node pairs)
                if excl is not None:
                    keepm=np.ones(len(Iall),dtype=bool)
                    for t in range(len(Iall)):
                        if excl[Iall[t],Jall[t]]:
                            keepm[t]=False
                    # keep at least some
                    if np.count_nonzero(keepm)>10:
                        Iall=Iall[keepm]; Jall=Jall[keepm]
                Qq=_tangent_bases(x_q)
                x_s=_smooth_slsqp(x_q,Qq,Iall,Jall,beta=600.0,radius=0.015,maxiter=40)
                vs=_mc(x_s)
                if vs<v:
                    x_q,v=x_s,vs
            except Exception:
                pass
        if v<best-1e-12:
            x_best=_nr(x_q); best=v
            do_refresh()
        # timeout guard
        if deadline-time.time()<22:
            break
    # final short LP polish
    rem=hard_end-time.time()
    if rem>4:
        try:
            x_best=_lp_polish(x_best,deadline=min(hard_end-0.7,time.time()+min(rem-1,18.0)),radius_init=1e-3)
        except Exception:
            pass
    x_best=_nr(x_best)
    return x_best

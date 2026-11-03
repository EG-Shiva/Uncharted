import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def norm_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return V / n

def build_cs(m, mults):
    p = len(mults)
    th = 2*np.pi*np.arange(m)[:,None]*np.array(mults)[None,:]/float(m)
    return np.cos(th), np.sin(th)

def gen_full_block(Yf, m, mults, p, n_fixed, C, S):
    nf = Yf.shape[0]
    Xf = np.zeros((nf*m, 13))
    Yfix = Yf[:, 2*p:]  # (nf, n_fixed)
    for g in range(m):
        sl = slice(g*nf, (g+1)*nf)
        for pi in range(p):
            c = C[g, pi]; s = S[g, pi]
            y0 = Yf[:, 2*pi]; y1 = Yf[:, 2*pi+1]
            Xf[sl, 2*pi] = c*y0 - s*y1
            Xf[sl, 2*pi+1] = s*y0 + c*y1
        if n_fixed>0:
            Xf[sl, 2*p:] = Yfix
    return Xf

def accum_full(gXf, nf, m, mults, p, n_fixed, C, S):
    gY = np.zeros((nf, 13))
    for g in range(m):
        blk = gXf[g*nf:(g+1)*nf]
        for pi in range(p):
            c = C[g, pi]; s = S[g, pi]
            a0 = blk[:, 2*pi]; a1 = blk[:, 2*pi+1]
            gY[:, 2*pi] += c*a0 + s*a1
            gY[:, 2*pi+1] += -s*a0 + c*a1
        if n_fixed>0:
            gY[:, 2*p:] += blk[:, 2*p:]
    return gY

def rot_back_vec(B, g, p, C, S):
    # B (k,13) -> R_{-g}(B)
    k = B.shape[0]
    Y = np.zeros_like(B)
    for pi in range(p):
        c = C[g, pi]; s = S[g, pi]
        b0 = B[:, 2*pi]; b1 = B[:, 2*pi+1]
        Y[:, 2*pi] = c*b0 + s*b1
        Y[:, 2*pi+1] = -s*b0 + c*b1
    Y[:, 2*p:] = B[:, 2*p:]
    return Y

def symmetrise(B, cfg, rng, iters=15):
    m = cfg['m']; mults = cfg['mults']; p = cfg['p']
    n_fixed = 13-2*p
    nf = cfg['n_full']; nfx = cfg['n_fix']
    C, S = build_cs(m, mults)
    # init Yf from random subset of B
    idx = rng.choice(B.shape[0], size=nf, replace=False)
    Yf = norm_rows(B[idx].copy())
    if nfx>0:
        Z = rng.standard_normal(size=(nfx, max(n_fixed,1)))
        Z = norm_rows(Z)
        Yx = np.zeros((nfx, 13))
        if n_fixed>0:
            Yx[:, 2*p:] = Z[:, :n_fixed]
        else:
            Yx = norm_rows(rng.standard_normal(size=(nfx,13)))
    else:
        Yx = np.zeros((0,13))
    for _ in range(iters):
        Xf = gen_full_block(Yf, m, mults, p, n_fixed, C, S)
        if nfx>0:
            Xall = np.vstack([Xf, Yx])
        else:
            Xall = Xf
        G = B @ Xall.T
        assign = np.argmax(G, axis=1)
        Nf = Xf.shape[0]
        newYf = np.zeros_like(Yf)
        cntf = np.zeros(nf)
        acc = [ [] for _ in range(nf) ]
        # accumulate rotated-back vectors
        for i in range(B.shape[0]):
            a = int(assign[i])
            if a < Nf:
                gg = a // nf; jj = a % nf
                acc[jj].append((i, gg))
            # fixed assignments handled later
        for j in range(nf):
            if len(acc[j])==0:
                newYf[j] = Yf[j]
            else:
                blk = np.stack([B[ii] for (ii,gg) in acc[j]], axis=0)
                # need per-element different g; rotate each individually
                s = np.zeros(13)
                for (ii,gg) in acc[j]:
                    rb = rot_back_vec(B[ii:ii+1], gg, p, C, S)
                    s += rb[0]
                s = s/len(acc[j])
                n = np.linalg.norm(s)
                if n < 1e-12:
                    s = Yf[j]
                    n = np.linalg.norm(s)
                newYf[j] = s/n
        Yf = newYf
        if nfx>0:
            # update fixed points: mean of assigned, projected
            fix_lists = [ [] for _ in range(nfx) ]
            for i in range(B.shape[0]):
                a = int(assign[i])
                if a >= Nf:
                    fix_lists[a-Nf].append(i)
            newYx = Yx.copy()
            for k in range(nfx):
                if len(fix_lists[k])==0:
                    continue
                M = B[fix_lists[k]].mean(axis=0)
                if n_fixed>0:
                    v = M[2*p:]
                    n = np.linalg.norm(v)
                    if n < 1e-12:
                        continue
                    v = v/n
                    newYx[k, :2*p] = 0.0
                    newYx[k, 2*p:] = v
                else:
                    n = np.linalg.norm(M)
                    if n>1e-12:
                        newYx[k]=M/n
            Yx = newYx
    # pack small fixed representation
    if nfx>0 and n_fixed>0:
        Yfix_small = Yx[:, 2*p:].copy()
    else:
        Yfix_small = np.zeros((nfx, 0))
    return Yf, Yfix_small, Yx, (C,S)

def lse_grad(X, alpha):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    E = np.exp(alpha*(G-m))
    np.fill_diagonal(E, 0.0)
    S = E.sum()/2.0 + 1e-300
    L = m + np.log(S)/alpha
    gX = (E @ X)/S
    return L, gX

def run_lbfgs(fun, x0, maxiter):
    try:
        r = minimize(fun, x0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol':1e-20, 'gtol':1e-12, 'maxfun': maxiter*5, 'maxcor': 12})
        return r.x.copy(), float(r.fun)
    except Exception:
        return x0, float('inf')

def make_sym_fun(cfg, C, S, alpha):
    m=cfg['m']; mults=cfg['mults']; p=cfg['p']
    n_fixed=13-2*p; nf=cfg['n_full']; nfx=cfg['n_fix']
    def fun(vf):
        Vf = vf[:nf*13].reshape(nf,13)
        nr = np.linalg.norm(Vf, axis=1, keepdims=True); nr[nr==0]=1.0
        Yf = Vf/nr
        if nfx>0 and n_fixed>0:
            Vs = vf[nf*13:].reshape(nfx, n_fixed)
            ns = np.linalg.norm(Vs, axis=1, keepdims=True); ns[ns==0]=1.0
            Ys = Vs/ns
            Yx = np.zeros((nfx,13))
            Yx[:, 2*p:] = Ys
        elif nfx>0:
            Yx = norm_rows(vf[nf*13:].reshape(nfx,13))
            Ys = None
        else:
            Yx = np.zeros((0,13)); Ys=None
        Xf = gen_full_block(Yf, m, mults, p, n_fixed, C, S)
        X = Xf if nfx==0 else np.vstack([Xf, Yx])
        L, gX = lse_grad(X, alpha)
        Nf = Xf.shape[0]
        gXf = gX[:Nf]; gXx = gX[Nf:] if nfx>0 else None
        gYf = accum_full(gXf, nf, m, mults, p, n_fixed, C, S)
        dot = (gYf*Yf).sum(axis=1, keepdims=True)
        gVf = (gYf - dot*Yf)/nr
        parts = [gVf.ravel()]
        if nfx>0 and n_fixed>0:
            gYx_fix = gXx[:, 2*p:]
            # project to tangent of small sphere
            dot2 = (gYx_fix*Ys).sum(axis=1, keepdims=True)
            ns2 = np.linalg.norm(vf[nf*13:].reshape(nfx,n_fixed), axis=1, keepdims=True); ns2[ns2==0]=1.0
            gVs = (gYx_fix - dot2*Ys)/ns2
            parts.append(gVs.ravel())
        elif nfx>0:
            dot2=(gXx*Yx).sum(axis=1,keepdims=True)
            ns2=np.linalg.norm(vf[nf*13:].reshape(nfx,13),axis=1,keepdims=True); ns2[ns2==0]=1.0
            parts.append(((gXx-dot2*Yx)/ns2).ravel())
        g = np.concatenate(parts).astype(float)
        return float(L), g
    return fun

def make_stage1_fun(cfg, C, S, alpha):
    # Yf symmetric + free fixed points in 13D
    m=cfg['m']; mults=cfg['mults']; p=cfg['p']
    n_fixed=13-2*p; nf=cfg['n_full']; nfx=cfg['n_fix']
    def fun(vf):
        Vf = vf[:nf*13].reshape(nf,13)
        nr = np.linalg.norm(Vf, axis=1, keepdims=True); nr[nr==0]=1.0
        Yf = Vf/nr
        if nfx>0:
            Vz = vf[nf*13:].reshape(nfx,13)
            nz = np.linalg.norm(Vz, axis=1, keepdims=True); nz[nz==0]=1.0
            Yz = Vz/nz
        else:
            Yz = np.zeros((0,13))
        Xf = gen_full_block(Yf, m, mults, p, n_fixed, C, S)
        X = Xf if nfx==0 else np.vstack([Xf, Yz])
        L, gX = lse_grad(X, alpha)
        Nf = Xf.shape[0]
        gXf = gX[:Nf]; gXz = gX[Nf:] if nfx>0 else None
        gYf = accum_full(gXf, nf, m, mults, p, n_fixed, C, S)
        dot=(gYf*Yf).sum(axis=1,keepdims=True)
        gVf=(gYf-dot*Yf)/nr
        if nfx>0:
            nz2=np.linalg.norm(vf[nf*13:].reshape(nfx,13),axis=1,keepdims=True); nz2[nz2==0]=1.0
            Yz2=Vf if False else Yz
            dot2=(gXz*Yz).sum(axis=1,keepdims=True)
            gVz=(gXz-dot2*Yz)/nz2
            g=np.concatenate([gVf.ravel(), gVz.ravel()]).astype(float)
        else:
            g=gVf.ravel().astype(float)
        return float(L), g
    return fun

def make_full_fun(alpha):
    def fun(vf):
        V=vf.reshape(99,13)
        nr=np.linalg.norm(V,axis=1,keepdims=True); nr[nr==0]=1.0
        Y=V/nr
        L,gX=lse_grad(Y,alpha)
        dot=(gX*Y).sum(axis=1,keepdims=True)
        gV=(gX-dot*Y)/nr
        return float(L), gV.ravel().astype(float)
    return fun

def build_X_from_sym(Yf, Yfix_small, Yx_guess, cfg, C, S, free_fix=False, Yz=None):
    m=cfg['m']; mults=cfg['mults']; p=cfg['p']
    n_fixed=13-2*p
    Xf=gen_full_block(Yf,m,mults,p,n_fixed,C,S)
    if cfg['n_fix']==0:
        return norm_rows(Xf)
    if free_fix and Yz is not None:
        return norm_rows(np.vstack([Xf, Yz]))
    # constrained: use Yx_guess embedded
    return norm_rows(np.vstack([Xf, Yx_guess]))

def lp_polish(x, seconds):
    x = norm_rows(np.asarray(x,float).copy())
    n,d = x.shape
    best = max_cos(x); bestx=x.copy()
    radius=5e-4
    start=time.time()
    iu=np.triu_indices(n,1)
    ar=np.arange(d)
    while time.time()-start < seconds and radius>1e-14:
        c=(x@x.T)[iu]
        m=float(c.max())
        thr=max(30.0*radius,3e-7)
        sel=c>m-thr
        if int(sel.sum())<900:
            o=np.argsort(-c); s2=np.zeros_like(sel,bool); s2[o[:900]]=True
            sel=sel|s2
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0: break
        if p>4000:
            o=np.argsort(c[sel])[-4000:]; I,J=I[o],J[o]; p=4000
            cs=c[sel][o]
        else:
            cs=c[sel]
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+ar,J[:,None]*d+ar,np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-cs,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=norm_rows(x+res.x[:-1].reshape(n,d))
        v=max_cos(y)
        if v<best:
            best=v; bestx=y.copy(); x=y
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return bestx

def solve():
    t0=time.time(); BUDGET=172.0
    rng=np.random.default_rng(777)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        try: xb=np.load("record.npy").astype(float)
        except Exception: xb=rng.standard_normal(size=(99,13))
    xb=np.asarray(xb,float)
    if xb.shape!=(99,13):
        tmp=rng.standard_normal(size=(99,13))
        try:
            if xb.ndim==2:
                m0=min(xb.shape[0],99); m1=min(xb.shape[1],13)
                tmp[:m0,:m1]=xb[:m0,:m1]
        except Exception: pass
        xb=tmp
    xb=norm_rows(xb)
    best_x=xb.copy(); best_val=max_cos(best_x)
    try:
        xr=norm_rows(np.load("record.npy").astype(float))
        rv=max_cos(xr)
        if rv<best_val: best_val=rv; best_x=xr.copy()
    except Exception: pass
    base_best=xb.copy()

    groups=[
        dict(name='C11',m=11,mults=[1,2,3,4,5],p=5,n_full=9,n_fix=0),
        dict(name='C9',m=9,mults=[1,2,3,4],p=4,n_full=11,n_fix=0),
        dict(name='C3',m=3,mults=[1,1,1,2,2,1],p=6,n_full=33,n_fix=0),
        dict(name='C2',m=2,mults=[1,1,1,1,1,1],p=6,n_full=49,n_fix=1),
        dict(name='C16',m=16,mults=[1,2,3,4,5],p=5,n_full=6,n_fix=3),
        dict(name='C12',m=12,mults=[1,2,3,4,5],p=5,n_full=8,n_fix=3),
    ]
    sym_cands=[]
    SYM_DEADLINE=t0+88.0
    for gi,cfg in enumerate(groups):
        gstart=time.time()
        if time.time()>SYM_DEADLINE: break
        # symmetrised start by projection
        try:
            Yf0,Ys0,Yx0,(C,S)=symmetrise(base_best,cfg,rng,iters=12)
        except Exception:
            continue
        starts=[]
        nf=cfg['n_full']; nfx=cfg['n_fix']; n_fixed=13-2*cfg['p']
        # pack symmetrised
        if nfx>0 and n_fixed>0:
            v0=np.concatenate([Yf0.ravel(),Ys0.ravel()]).astype(float)
        elif nfx>0:
            v0=np.concatenate([Yf0.ravel(),norm_rows(rng.standard_normal(size=(nfx,13))).ravel()]).astype(float)
        else:
            v0=Yf0.ravel().astype(float)
        starts.append(v0)
        # random start
        try:
            Yr=norm_rows(rng.standard_normal(size=(nf,13)))
            if nfx>0 and n_fixed>0:
                Ysr=norm_rows(rng.standard_normal(size=(nfx,n_fixed)))
                vr=np.concatenate([Yr.ravel(),Ysr.ravel()]).astype(float)
            elif nfx>0:
                Yr2=norm_rows(rng.standard_normal(size=(nfx,13)))
                vr=np.concatenate([Yr.ravel(),Yr2.ravel()]).astype(float)
            else:
                vr=Yr.ravel().astype(float)
            starts.append(vr)
        except Exception: pass
        for si,v0 in enumerate(starts):
            if time.time()-gstart>15.0: break
            if time.time()>SYM_DEADLINE: break
            cur=v0.copy()
            for alpha,mi in [(20.0,120),(80.0,120),(200.0,120)]:
                if time.time()-gstart>15.0: break
                if time.time()>SYM_DEADLINE: break
                fun=make_sym_fun(cfg,C,S,alpha)
                cur,_=run_lbfgs(fun,cur,mi)
            # evaluate
            try:
                Vf=cur[:nf*13].reshape(nf,13); Yf=norm_rows(Vf)
                if nfx>0 and n_fixed>0:
                    Vs=cur[nf*13:].reshape(nfx,n_fixed); Ys=norm_rows(Vs)
                    Yx=np.zeros((nfx,13)); Yx[:,2*cfg['p']:]=Ys
                elif nfx>0:
                    Yx=norm_rows(cur[nf*13:].reshape(nfx,13)); Ys=None
                else:
                    Yx=np.zeros((0,13))
                X=build_X_from_sym(Yf,None,Yx,cfg,C,S)
                sc=max_cos(X)
                sym_cands.append((sc,X.copy(),cfg['name'],cur.copy(),C.copy(),S.copy(),cfg))
                if sc<best_val: best_val=sc; best_x=X.copy()
            except Exception: continue
    sym_cands.sort(key=lambda t:t[0])
    # release symmetry in stages
    seeds=sym_cands[:3]
    for si,(sc,Xs,name,vflat,C,S,cfg) in enumerate(seeds):
        if time.time()-t0>132.0: break
        try:
            curX=Xs.copy()
            nf=cfg['n_full']; nfx=cfg['n_fix']; p=cfg['p']; n_fixed=13-2*p
            # stage 1: allow fixed coords if applicable
            if nfx>0:
                # build stage1 vector: Yf part from symmetric optimum + free fixed from Xs tail
                Yf_part=None
                # recover Yf from vflat
                Vf=vflat[:nf*13].reshape(nf,13); Yf_part=norm_rows(Vf)
                Yz_part=curX[-nfx:].copy()+rng.standard_normal(size=(nfx,13))*0.002
                Yz_part=norm_rows(Yz_part)
                v1=np.concatenate([Yf_part.ravel(),Yz_part.ravel()]).astype(float)
                fun1=make_stage1_fun(cfg,C,S,80.0)
                v1,_=run_lbfgs(fun1,v1,100)
                # rebuild full X from stage1
                Vf2=v1[:nf*13].reshape(nf,13); Yf2=norm_rows(Vf2)
                Yz2=norm_rows(v1[nf*13:].reshape(nfx,13))
                curX=build_X_from_sym(Yf2,None,None,cfg,C,S,free_fix=True,Yz=Yz2)
                sv=max_cos(curX)
                if sv<best_val: best_val=sv; best_x=curX.copy()
                if time.time()-t0>132.0: break
            # stage 2: full release with small shake + anneal
            shake = 0.003 if si<2 else 0.006
            cur = curX + rng.standard_normal(size=curX.shape)*shake
            cur = norm_rows(cur)
            vf = cur.ravel().astype(float)
            for alpha,mi in [(80.0,150),(200.0,120)]:
                if time.time()-t0>134.0: break
                fun=make_full_fun(alpha)
                vf,_=run_lbfgs(fun,vf,mi)
            Xf2=norm_rows(vf.reshape(99,13))
            s2=max_cos(Xf2)
            if s2<best_val: best_val=s2; best_x=Xf2.copy()
            # keep curX updated for possible extra shake
            if time.time()-t0<128.0:
                try:
                    cur2=Xf2+rng.standard_normal(size=Xf2.shape)*0.008
                    cur2=norm_rows(cur2)
                    fun=make_full_fun(150.0)
                    vf2,_=run_lbfgs(fun,cur2.ravel().astype(float),100)
                    X3=norm_rows(vf2.reshape(99,13))
                    s3=max_cos(X3)
                    if s3<best_val: best_val=s3; best_x=X3.copy()
                except Exception: pass
        except Exception: continue
    # also try one direct full anneal from incumbent if time remains
    if time.time()-t0<130.0:
        try:
            cur=best_x+rng.standard_normal(size=best_x.shape)*0.004
            cur=norm_rows(cur)
            vf=cur.ravel().astype(float)
            for alpha,mi in [(100.0,120),(300.0,100)]:
                if time.time()-t0>136.0: break
                fun=make_full_fun(alpha)
                vf,_=run_lbfgs(fun,vf,mi)
            Xf=norm_rows(vf.reshape(99,13))
            sv=max_cos(Xf)
            if sv<best_val: best_val=sv; best_x=Xf.copy()
        except Exception: pass
    rem=BUDGET-(time.time()-t0)
    rem=max(5.0,min(rem,40.0))
    try:
        pol=lp_polish(best_x,rem-0.5)
        pv=max_cos(pol)
        if pv<best_val+1e-9:
            best_x=pol; best_val=min(best_val,pv)
    except Exception: pass
    best_x=norm_rows(best_x)
    return best_x

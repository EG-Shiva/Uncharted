import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def norm_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

_TH11 = 2*np.pi*np.arange(11)[:,None]*np.arange(1,6)[None,:]/11.0
_C11 = np.cos(_TH11); _S11 = np.sin(_TH11)
_TH9 = 2*np.pi*np.arange(9)[:,None]*np.arange(1,5)[None,:]/9.0
_C9 = np.cos(_TH9); _S9 = np.sin(_TH9)

def gen_C11(Y):
    nrep, G = 9, 11
    X = np.zeros((nrep*G, 13))
    for g in range(G):
        idx = np.arange(nrep)*G+g
        for p in range(5):
            c, s = _C11[g,p], _S11[g,p]
            y0 = Y[:,2*p]; y1 = Y[:,2*p+1]
            X[idx,2*p]=c*y0-s*y1
            X[idx,2*p+1]=s*y0+c*y1
        X[idx,10:13]=Y[:,10:13]
    return X

def gen_C9(Y):
    nrep, G = 11, 9
    X = np.zeros((nrep*G, 13))
    for g in range(G):
        idx = np.arange(nrep)*G+g
        for p in range(4):
            c, s = _C9[g,p], _S9[g,p]
            y0 = Y[:,2*p]; y1 = Y[:,2*p+1]
            X[idx,2*p]=c*y0-s*y1
            X[idx,2*p+1]=s*y0+c*y1
        X[idx,8:13]=Y[:,8:13]
    return X

def accum_C11(gX):
    gY = np.zeros((9,13))
    for g in range(11):
        idx = np.arange(9)*11+g
        for p in range(5):
            c, s = _C11[g,p], _S11[g,p]
            gx0=gX[idx,2*p]; gx1=gX[idx,2*p+1]
            gY[:,2*p]+=c*gx0+s*gx1
            gY[:,2*p+1]+=-s*gx0+c*gx1
        gY[:,10:13]+=gX[idx,10:13]
    return gY

def accum_C9(gX):
    gY = np.zeros((11,13))
    for g in range(9):
        idx = np.arange(11)*9+g
        for p in range(4):
            c, s = _C9[g,p], _S9[g,p]
            gx0=gX[idx,2*p]; gx1=gX[idx,2*p+1]
            gY[:,2*p]+=c*gx0+s*gx1
            gY[:,2*p+1]+=-s*gx0+c*gx1
        gY[:,8:13]+=gX[idx,8:13]
    return gY

def lse_grad(X, alpha):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    E = np.exp(alpha*(G-m))
    np.fill_diagonal(E, 0.0)
    S = E.sum()/2.0+1e-300
    L = m+np.log(S)/alpha
    gX = (E @ X)/S
    return L, gX

def make_sym_obj(kind, alpha):
    if kind=='C11':
        nrep,gen,acc=9,gen_C11,accum_C11
    else:
        nrep,gen,acc=11,gen_C9,accum_C9
    def fun(vf):
        V=vf.reshape(nrep,13)
        nr=np.linalg.norm(V,axis=1,keepdims=True); nr[nr==0]=1.0
        Y=V/nr
        X=gen(Y)
        L,gX=lse_grad(X,alpha)
        gY=acc(gX)
        dot=(gY*Y).sum(axis=1,keepdims=True)
        gV=(gY-dot*Y)/nr
        return float(L), gV.ravel().astype(float)
    return fun, nrep

def make_full_obj(alpha):
    def fun(vf):
        V=vf.reshape(99,13)
        nr=np.linalg.norm(V,axis=1,keepdims=True); nr[nr==0]=1.0
        Y=V/nr
        L,gX=lse_grad(Y,alpha)
        dot=(gX*Y).sum(axis=1,keepdims=True)
        gV=(gX-dot*Y)/nr
        return float(L), gV.ravel().astype(float)
    return fun

def run_lbfgs(fun,x0,maxiter):
    try:
        r=minimize(fun,x0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-20,'gtol':1e-9,'maxfun':maxiter*4})
        return r.x,float(r.fun)
    except Exception:
        return x0,float('inf')

def smooth_refine(v,a0,a1,steps,lr,beta=0.9):
    n,d=v.shape
    cur=v.copy(); best=cur.copy(); bs=max_cos(cur)
    mom=np.zeros_like(cur)
    for s in range(steps):
        f=s/max(steps-1,1)
        a=a0*((a1/max(a0,1e-12))**f)
        G=cur@cur.T; np.fill_diagonal(G,-10.0)
        m=float(np.max(G))
        W=np.exp(a*(G-m)); np.fill_diagonal(W,0.0)
        Gr=W@cur
        Gt=Gr-np.sum(Gr*cur,axis=1,keepdims=True)*cur
        rs=W.sum(axis=1,keepdims=True)+1e-12
        Gt=Gt/rs
        mom=beta*mom+(1-beta)*Gt
        st=mom/(1-beta**(s+1))
        cur=cur-lr*(1-0.5*f)*st
        cur=norm_rows(cur)
        if (s%200==199) or s==steps-1:
            sc=max_cos(cur)
            if sc<bs: bs=sc; best=cur.copy()
    return best,bs

def lp_wide(x,seconds,kmin=900,wmin=2e-7,cap=3800):
    x=norm_rows(np.asarray(x,dtype=float))
    n,d=x.shape
    best=max_cos(x); bx=x.copy()
    radius=5e-4; start=time.time()
    iu=np.triu_indices(n,1); ar=np.arange(d)
    while time.time()-start<seconds and radius>1e-14:
        c=(x@x.T)[iu]; m=float(c.max())
        thr=m-max(30.0*radius,wmin,1e-9)
        sel=c>thr
        if int(np.sum(sel))<kmin:
            o=np.argsort(-c); s2=np.zeros_like(sel,dtype=bool); s2[o[:kmin]]=True; sel=sel|s2
        if cap is not None and int(np.sum(sel))>cap:
            idx=np.where(sel)[0]; keep=idx[np.argsort(-c[idx])[:cap]]
            s2=np.zeros_like(sel,dtype=bool); s2[keep]=True; sel=s2
        I,J=iu[0][sel],iu[1][sel]; p=len(I)
        if p==0: break
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+ar,J[:,None]*d+ar,np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            Aub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            Aeq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=Aub,b_ub=-c[sel],A_eq=Aeq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d); y=norm_rows(y)
        v=max_cos(y)
        if v<best:
            best=v; bx=y.copy(); x=y; radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return bx

def far_reinsert(base,idx,rng,nc=1500):
    x0=base.copy()
    for k in idx:
        cand=rng.standard_normal(size=(nc,base.shape[1])); cand=norm_rows(cand)
        oth=np.delete(x0,int(k),axis=0)
        md=(cand@oth.T).max(axis=1)
        x0[int(k)]=cand[np.argmin(md)]
    return x0

def solve():
    t0=time.time(); BUDGET=172.0
    rng=np.random.default_rng(0)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        try: xb=np.load("record.npy").astype(float)
        except Exception: xb=rng.standard_normal(size=(99,13))
    xb=np.asarray(xb,dtype=float)
    if xb.shape!=(99,13):
        tmp=rng.standard_normal(size=(99,13)); xb=tmp
    xb=norm_rows(xb)
    n,d=99,13
    best_val=max_cos(xb); best_x=xb.copy()
    # --- symmetry search ---
    sym_cands=[]
    sym_deadline=t0+42.0
    trial=0
    kinds=['C11','C9','C11','C9']
    while time.time()<sym_deadline:
        kind=kinds[trial%len(kinds)]; trial+=1
        try:
            fun,nrep=make_sym_obj(kind,100.0)
            Y0=norm_rows(rng.standard_normal(size=(nrep,13)))
            # occasionally seed from best by averaging orbits (quick random instead)
            x0=Y0.ravel()
            xf,_=run_lbfgs(fun,x0,maxiter=120)
            V=xf.reshape(nrep,13); Y=norm_rows(V)
            fun2,_=make_sym_obj(kind,400.0)
            xf2,_=run_lbfgs(fun2,Y.ravel(),maxiter=120)
            V2=xf2.reshape(nrep,13); Y2=norm_rows(V2)
            X=gen_C11(Y2) if kind=='C11' else gen_C9(Y2)
            X=norm_rows(X)
            sc=max_cos(X)
            sym_cands.append((sc,X.copy(),kind))
            if sc<best_val-1e-12:
                best_val=sc; best_x=X.copy()
            if time.time()>sym_deadline-2: break
        except Exception:
            continue
    sym_cands.sort(key=lambda t:t[0])
    # --- release symmetry: annealed LSE + wide LP ---
    rel_end=t0+78.0
    for si,(sc,X0,kind) in enumerate(sym_cands[:3]):
        if time.time()>rel_end: break
        if sc>0.30: continue
        try:
            # release: full smooth anneal
            Xr,_=smooth_refine(X0,80.0,350.0,700,0.035)
            Xr2,_=smooth_refine(Xr,250.0,650.0,600,0.015)
            v=max_cos(Xr2)
            # quick LP
            if BUDGET-(time.time()-t0)>10:
                Xp=lp_wide(Xr2,1.2,kmin=1100,wmin=2e-7)
                vp=max_cos(Xp)
                if vp<v: Xr2=Xp; v=vp
            if v<best_val:
                best_val=v; best_x=Xr2.copy()
            # also full LBFGS squeeze if promising
            if v<best_val+0.002 and time.time()<rel_end-4:
                try:
                    fo=make_full_obj(500.0)
                    xf,_=run_lbfgs(fo,Xr2.ravel(),maxiter=80)
                    Xf=norm_rows(xf.reshape(99,13))
                    vf=max_cos(Xf)
                    if vf<v:
                        v=vf; Xr2=Xf
                    if vf<best_val:
                        best_val=vf; best_x=Xf.copy()
                except Exception: pass
        except Exception: continue
    # --- general basin hopping from best ---
    cur_x=best_x.copy(); cur_val=best_val
    EXPLORE_UNTIL=t0+138.0
    it=0
    sigmas=[0.015,0.04,0.09,0.17,0.30]
    while time.time()<EXPLORE_UNTIL and it<1000:
        it+=1
        prog=(time.time()-t0)/max(EXPLORE_UNTIL-t0,1e-9)
        Temp=0.0015*(1-prog)+0.00015
        try:
            Gc=cur_x@cur_x.T; np.fill_diagonal(Gc,-2.0)
            rm=Gc.max(axis=1); worst=np.argsort(rm)
            mode=rng.random()
            if mode<0.35:
                sig=sigmas[int(rng.integers(0,len(sigmas)))]
                x0=norm_rows(cur_x+rng.standard_normal(size=(n,d))*sig)
                a0=float(rng.choice([15.0,25.0,40.0])); a1=a0*float(rng.choice([4.0,6.0,9.0]))
                x1,_=smooth_refine(x0,a0,a1,int(rng.integers(500,1000)),float(rng.uniform(0.02,0.06)))
            elif mode<0.62:
                kk=int(rng.integers(2,7)); idx=worst[-kk:]
                x0=cur_x.copy(); sig=float(rng.uniform(0.05,0.30))
                x0[idx]=norm_rows(x0[idx]+rng.standard_normal(size=(kk,d))*sig)
                if rng.random()<0.4:
                    pick=rng.choice(idx,size=min(2,kk),replace=False)
                    x0=far_reinsert(x0,pick,rng,nc=1500)
                a0=float(rng.choice([20.0,35.0,55.0]))
                x1,_=smooth_refine(x0,a0,a0*5.0+20.0,int(rng.integers(500,1000)),float(rng.uniform(0.02,0.06)))
            elif mode<0.82:
                kk=int(rng.integers(2,6)); idx=worst[-kk:]
                x0=far_reinsert(cur_x,idx,rng,nc=1500)
                x0=norm_rows(x0+rng.standard_normal(size=(n,d))*0.01)
                x1,_=smooth_refine(x0,25.0,200.0,int(rng.integers(600,1000)),float(rng.uniform(0.025,0.06)))
            else:
                msz=int(rng.integers(8,18)); idx=rng.choice(n,size=msz,replace=False)
                x0=cur_x.copy()
                if rng.random()<0.5: x0[idx]=norm_rows(rng.standard_normal(size=(msz,d)))
                else: x0=norm_rows(x0+np.concatenate([rng.standard_normal(size=(msz,d))*0.4 if False else np.zeros((0,d))],axis=0) if False else x0*0+cur_x+rng.standard_normal(size=(n,d))*0.0+ (np.zeros((n,d)) ))  # placeholder
                # simpler large shake:
                x0=cur_x.copy()
                x0[idx]=norm_rows(x0[idx]+rng.standard_normal(size=(msz,d))*0.4)
                x1,_=smooth_refine(x0,20.0,200.0,int(rng.integers(500,900)),float(rng.uniform(0.03,0.07)))
        except Exception: continue
        try: s1=max_cos(x1)
        except Exception: continue
        if s1<best_val-1e-12:
            best_val=s1; best_x=x1.copy(); cur_x=x1.copy(); cur_val=s1
            if time.time()<EXPLORE_UNTIL-6:
                try:
                    sn=lp_wide(best_x,0.8,kmin=900,wmin=3e-7)
                    sv=max_cos(sn)
                    if sv<best_val: best_val=sv; best_x=sn.copy(); cur_x=sn.copy(); cur_val=sv
                except Exception: pass
        else:
            if s1<=cur_val or rng.random()<np.exp(-(s1-cur_val)/max(Temp,1e-12)):
                cur_x=x1.copy(); cur_val=s1
            if rng.random()<0.07: cur_x=best_x.copy(); cur_val=best_val
    # --- final polish ---
    remain=BUDGET-(time.time()-t0)-1.5
    if remain>5:
        bx=best_x.copy(); bv=max_cos(bx)
        try:
            tx,_=smooth_refine(bx,25.0,150.0,700,0.03)
            if max_cos(tx)<bv: bx=tx.copy(); bv=max_cos(bx)
        except Exception: pass
        try:
            tx2,_=smooth_refine(bx,120.0,700.0,700,0.012)
            if max_cos(tx2)<bv: bx=tx2.copy(); bv=max_cos(bx)
        except Exception: pass
        try:
            # full LBFGS high alpha squeeze
            if remain>12:
                try:
                    fo=make_full_obj(600.0)
                    xf,_=run_lbfgs(fo,bx.ravel(),maxiter=100)
                    Xf=norm_rows(xf.reshape(99,13))
                    if max_cos(Xf)<bv: bx=Xf.copy(); bv=max_cos(bx)
                except Exception: pass
        except Exception: pass
        try:
            pol=lp_wide(bx,remain*0.7,kmin=1100,wmin=2e-7)
            if max_cos(pol)<bv: bx=pol.copy(); bv=max_cos(bx)
        except Exception: pass
        if bv<best_val: best_x=bx.copy(); best_val=bv
    best_x=norm_rows(best_x)
    return best_x

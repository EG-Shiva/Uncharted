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

_C11_TH = 2*np.pi*np.arange(11)[:,None]*np.arange(1,6)[None,:]/11.0
_C11_C = np.cos(_C11_TH); _C11_S = np.sin(_C11_TH)
_C9_TH = 2*np.pi*np.arange(9)[:,None]*np.arange(1,5)[None,:]/9.0
_C9_C = np.cos(_C9_TH); _C9_S = np.sin(_C9_TH)

def gen_C11(Y):
    G = 11; nrep = 9
    X = np.zeros((99,13))
    for g in range(G):
        idx = np.arange(nrep)*G+g
        for p in range(5):
            c=_C11_C[g,p]; s=_C11_S[g,p]
            y0=Y[:,2*p]; y1=Y[:,2*p+1]
            X[idx,2*p]=c*y0-s*y1
            X[idx,2*p+1]=s*y0+c*y1
        X[idx,10:13]=Y[:,10:13]
    return X

def gen_C9(Y):
    G=9; nrep=11
    X=np.zeros((99,13))
    for g in range(G):
        idx=np.arange(nrep)*G+g
        for p in range(4):
            c=_C9_C[g,p]; s=_C9_S[g,p]
            y0=Y[:,2*p]; y1=Y[:,2*p+1]
            X[idx,2*p]=c*y0-s*y1
            X[idx,2*p+1]=s*y0+c*y1
        X[idx,8:13]=Y[:,8:13]
    return X

def accum_C11(gX):
    gY=np.zeros((9,13))
    for g in range(11):
        idx=np.arange(9)*11+g
        for p in range(5):
            c=_C11_C[g,p]; s=_C11_S[g,p]
            a0=gX[idx,2*p]; a1=gX[idx,2*p+1]
            gY[:,2*p]+=c*a0+s*a1
            gY[:,2*p+1]+=-s*a0+c*a1
        gY[:,10:13]+=gX[idx,10:13]
    return gY

def accum_C9(gX):
    gY=np.zeros((11,13))
    for g in range(9):
        idx=np.arange(11)*9+g
        for p in range(4):
            c=_C9_C[g,p]; s=_C9_S[g,p]
            a0=gX[idx,2*p]; a1=gX[idx,2*p+1]
            gY[:,2*p]+=c*a0+s*a1
            gY[:,2*p+1]+=-s*a0+c*a1
        gY[:,8:13]+=gX[idx,8:13]
    return gY

def lse_full_grad(X, alpha):
    G=X@X.T
    np.fill_diagonal(G,-10.0)
    m=float(G.max())
    E=np.exp(alpha*(G-m))
    np.fill_diagonal(E,0.0)
    S=E.sum()/2.0+1e-300
    L=m+np.log(S)/alpha
    gX=(E@X)/S
    return L,gX,m

def make_sym_obj(kind, alpha):
    if kind=='C11':
        nrep=9; gen=gen_C11; acc=accum_C11
    else:
        nrep=11; gen=gen_C9; acc=accum_C9
    def fun(vf):
        V=vf.reshape(nrep,13)
        nr=np.linalg.norm(V,axis=1,keepdims=True); nr[nr==0]=1.0
        Y=V/nr
        X=gen(Y)
        L,gX,_=lse_full_grad(X,alpha)
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
        L,gX,_=lse_full_grad(Y,alpha)
        dot=(gX*Y).sum(axis=1,keepdims=True)
        gV=(gX-dot*Y)/nr
        return float(L), gV.ravel().astype(float)
    return fun

def run_lbfgs(fun,x0,maxiter):
    try:
        r=minimize(fun,x0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-20,'gtol':1e-12,'maxfun':maxiter*5})
        return r.x.copy(),float(r.fun)
    except Exception:
        return x0,float('inf')

def smooth_refine(v,a0,a1,steps,lr,beta=0.9):
    n,d=v.shape
    cur=v.copy(); best=cur.copy(); bs=max_cos(cur)
    mom=np.zeros_like(cur)
    for s in range(steps):
        frac=s/max(steps-1,1)
        alpha=a0*((a1/max(a0,1e-12))**frac)
        G=cur@cur.T; np.fill_diagonal(G,-10.0)
        m=float(np.max(G))
        W=np.exp(alpha*(G-m)); np.fill_diagonal(W,0.0)
        Grad=W@cur
        proj=np.sum(Grad*cur,axis=1,keepdims=True)
        Gt=Grad-proj*cur
        rs=W.sum(axis=1,keepdims=True)+1e-12
        Gt=Gt/rs
        mom=beta*mom+(1-beta)*Gt
        bc=1-beta**(s+1)
        cur=cur-lr*(1-0.5*frac)*(mom/bc)
        cur/=np.linalg.norm(cur,axis=1,keepdims=True)
        if (s%200==199) or s==steps-1:
            sc=max_cos(cur)
            if sc<bs: bs=sc; best=cur.copy()
    return best,bs

def balanced_greedy(v,steps=250,lr0=0.015,delta=7e-05,topk=7,w_top=1.0,w_bal=0.4):
    cur=v.copy(); best=cur.copy(); bs=max_cos(cur)
    n,d=v.shape
    for s in range(steps):
        G=cur@cur.T; np.fill_diagonal(G,-2.0)
        m=float(np.max(G))
        Mask=(G>m-delta).astype(float); np.fill_diagonal(Mask,0.0)
        idx=np.argsort(-G,axis=1)[:,:topk]
        Bal=np.zeros_like(Mask)
        for i in range(n):
            Bal[i,idx[i]]=1.0
        np.fill_diagonal(Bal,0.0)
        Bal=np.maximum(Bal,Bal.T)
        Comb=w_top*Mask+w_bal*Bal
        cnt=Comb.sum(axis=1,keepdims=True)+1e-12
        Grad=Comb@cur
        proj=np.sum(Grad*cur,axis=1,keepdims=True)
        Gt=(Grad-proj*cur)/cnt
        cur=cur-lr0*(1-0.6*s/max(steps,1))*Gt
        cur/=np.linalg.norm(cur,axis=1,keepdims=True)
        if s%100==99 or s==steps-1:
            sc=max_cos(cur)
            if sc<bs: bs=sc; best=cur.copy()
        if s==steps//2: delta=max(delta*0.25,1e-06)
    return best,bs

def lp_wide(x,seconds,kmin=900,wmin=3e-07):
    x=norm_rows(np.asarray(x,float)); n,d=x.shape
    best=max_cos(x); bx=x.copy()
    radius=6e-4; start=time.time()
    iu=np.triu_indices(n,1); ar=np.arange(d)
    while time.time()-start<seconds and radius>1e-14:
        c=(x@x.T)[iu]; m=float(c.max())
        thr=m-max(30*radius,wmin,1e-9)
        sel=c>thr
        if np.sum(sel)<kmin:
            o=np.argsort(-c); s2=np.zeros_like(sel,bool); s2[o[:kmin]]=True; sel=sel|s2
        I,J=iu[0][sel],iu[1][sel]; p=len(I)
        if p==0: break
        if p>3800:
            keep=np.argsort(-c[sel])[:3800]
            I,J=I[keep],J[keep]; csel=c[sel][keep]
        else:
            csel=c[sel]
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+ar,J[:,None]*d+ar,np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            Aub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            Aeq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=Aub,b_ub=-csel,A_eq=Aeq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y=norm_rows(y); vv=max_cos(y)
        if vv<best:
            best=vv; bx=y.copy(); x=y; radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
        if best<max_cos(x)-1e-12: x=bx.copy()
    return bx

def far_reinsert(base,idx,rng,n_cand=2000):
    x0=base.copy()
    for k in idx:
        cand=rng.standard_normal(size=(n_cand,base.shape[1])); cand=norm_rows(cand)
        others=np.delete(x0,int(k),axis=0)
        md=(cand@others.T).max(axis=1)
        x0[int(k)]=cand[np.argmin(md)]
    return x0

def solve():
    t0=time.time(); BUDGET=172.0; EXPLORE_UNTIL=128.0
    rng=np.random.default_rng(20261103)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        try: xb=np.load("record.npy").astype(float)
        except Exception: xb=rng.standard_normal(size=(99,13))
    xb=np.asarray(xb,float)
    if xb.shape!=(99,13):
        tmp=rng.standard_normal(size=(99,13)); xb=tmp
    xb=norm_rows(xb)
    n,d=99,13
    best_val=max_cos(xb); best_x=xb.copy()
    cur_x=xb.copy(); cur_val=best_val
    pool=[xb.copy()]
    # ---- symmetric search from scratch ----
    try:
        sym_cands=[]
        configs=[('C11',100.0,400.0,3),('C9',100.0,400.0,2)]
        for kind,a0,a1,ntry in configs:
            fun0,_n=make_sym_obj(kind,a0)
            fun1,_n2=make_sym_obj(kind,a1)
            for t in range(ntry):
                if time.time()-t0>48: break
                rs=np.random.default_rng(1000+hash(kind)%500+t*17)
                Y0=norm_rows(rs.standard_normal(size=(_n,13)))
                v0=Y0.ravel()
                v1,_=run_lbfgs(fun0,v0,120)
                v2,_=run_lbfgs(fun1,v1,120)
                V=v2.reshape(_n,13); Y=norm_rows(V)
                X=gen_C11(Y) if kind=='C11' else gen_C9(Y)
                X=norm_rows(X)
                sc=max_cos(X)
                sym_cands.append((sc,X.copy()))
                if sc<best_val-1e-12:
                    best_val=sc; best_x=X.copy(); cur_x=X.copy(); cur_val=sc
        sym_cands.sort(key=lambda z:z[0])
        # release symmetry: annealed full LBFGS
        for k in range(min(3,len(sym_cands))):
            if time.time()-t0>78: break
            Xs=sym_cands[k][1]
            sv=max_cos(Xs)
            # only release promising or diverse: release even if 1e-4 worse
            v0=(Xs+ rng.standard_normal(size=(99,13))*0.002)
            v0=norm_rows(v0).ravel()
            for alpha,mi in [(120.0,150),(350.0,150),(650.0,120)]:
                if time.time()-t0>85: break
                fo=make_full_obj(alpha)
                v0,_=run_lbfgs(fo,v0,mi)
            V=v0.reshape(99,13); Xr=norm_rows(V)
            sc=max_cos(Xr)
            pool.append(Xr.copy())
            if sc<best_val:
                best_val=sc; best_x=Xr.copy(); cur_x=Xr.copy(); cur_val=sc
            # also smooth+balance variant
            try:
                q,_=smooth_refine(Xr,25.0,180.0,700,0.035)
                q,_=balanced_greedy(q,steps=200,lr0=0.014,delta=6e-05,topk=7,w_top=1.0,w_bal=0.45)
                sq=max_cos(q)
                pool.append(q.copy())
                if sq<best_val:
                    best_val=sq; best_x=q.copy(); cur_x=q.copy(); cur_val=sq
            except Exception:
                pass
    except Exception:
        pass
    # ensure pool has best
    pool.append(best_x.copy())
    sigmas=[0.015,0.04,0.09,0.17,0.30]
    it=0
    while time.time()-t0<EXPLORE_UNTIL and it<1000:
        it+=1
        progress=(time.time()-t0)/max(EXPLORE_UNTIL,1e-9)
        Temp=0.0015*(1-progress)+0.00015
        try:
            Gc=cur_x@cur_x.T; np.fill_diagonal(Gc,-2.0)
            rm=Gc.max(axis=1); order=np.argsort(rm)
            mode=rng.random()
            if mode<0.30:
                sigma=sigmas[int(rng.integers(0,len(sigmas)))]
                # occasionally start from symmetric pool member
                base=cur_x
                if rng.random()<0.18 and len(pool)>1:
                    base=pool[int(rng.integers(0,len(pool)))]
                x0=base+rng.standard_normal(size=(n,d))*sigma
                x0=norm_rows(x0)
                a0=float(rng.choice([15.0,25.0,40.0])); a1=a0*float(rng.choice([4.0,6.0,9.0]))
                steps=int(rng.integers(500,1100)); lr=float(rng.uniform(0.02,0.06))
                x1,_=smooth_refine(x0,a0,a1,steps,lr)
                x1,_=balanced_greedy(x1,steps=220,lr0=0.016,delta=8e-05,topk=7,w_top=1.0,w_bal=0.4)
            elif mode<0.55:
                kk=int(rng.integers(2,8)); idx=order[-kk:]
                x0=cur_x.copy(); sigma=float(rng.uniform(0.05,0.30))
                x0[idx]=x0[idx]+rng.standard_normal(size=(kk,d))*sigma; x0=norm_rows(x0)
                if rng.random()<0.5:
                    pick=rng.choice(idx,size=min(2,kk),replace=False)
                    x0=far_reinsert(x0,pick,rng,2000)
                a0=float(rng.choice([20.0,35.0,55.0])); steps=int(rng.integers(500,1100)); lr=float(rng.uniform(0.02,0.06))
                x1,_=smooth_refine(x0,a0,a0*5.0+20.0,steps,lr)
                x1,_=balanced_greedy(x1,steps=220,lr0=0.014,delta=6e-05,topk=8,w_top=1.0,w_bal=0.5)
            elif mode<0.78:
                kk=int(rng.integers(2,6)); idx=order[-kk:]
                x0=far_reinsert(cur_x,idx,rng,2500)
                x0=x0+rng.standard_normal(size=(n,d))*0.01; x0=norm_rows(x0)
                a0=float(rng.choice([25.0,40.0]))
                x1,_=smooth_refine(x0,a0,a0*6.0,int(rng.integers(600,1200)),float(rng.uniform(0.025,0.06)))
                x1,_=balanced_greedy(x1,steps=250,lr0=0.012,delta=1e-04,topk=6,w_top=1.0,w_bal=0.3)
            else:
                msz=int(rng.integers(8,18)); idx=rng.choice(n,size=msz,replace=False)
                x0=cur_x.copy()
                if rng.random()<0.5:
                    x0[idx]=norm_rows(rng.standard_normal(size=(msz,d)))
                else:
                    x0[idx]=x0[idx]+rng.standard_normal(size=(msz,d))*0.4; x0=norm_rows(x0)
                x1,_=smooth_refine(x0,20.0,200.0,int(rng.integers(500,1000)),float(rng.uniform(0.03,0.07)))
                x1,_=balanced_greedy(x1,steps=220,lr0=0.014,delta=8e-05,topk=7,w_top=1.0,w_bal=0.4)
        except Exception:
            continue
        s1=max_cos(x1)
        if s1<best_val-1e-12:
            best_val=s1; best_x=x1.copy(); cur_x=x1.copy(); cur_val=s1
            pool.append(x1.copy())
            if len(pool)>8: pool=pool[-8:]
            if time.time()-t0<EXPLORE_UNTIL-6:
                try:
                    sn=lp_wide(best_x,1.0,kmin=900,wmin=3e-07)
                    sv=max_cos(sn)
                    if sv<best_val:
                        best_val=sv; best_x=sn.copy(); cur_x=sn.copy(); cur_val=sv
                except Exception: pass
        else:
            if s1<=cur_val or rng.random()<np.exp(-(s1-cur_val)/max(Temp,1e-12)):
                cur_x=x1.copy(); cur_val=s1
            if rng.random()<0.07:
                cur_x=best_x.copy(); cur_val=best_val
    remain=BUDGET-(time.time()-t0)-1.5
    if remain>5:
        bx=best_x.copy(); bv=max_cos(bx)
        try:
            tx,_=smooth_refine(bx,25.0,150.0,800,0.03)
            tx,_=balanced_greedy(tx,steps=350,lr0=0.010,delta=5e-05,topk=8,w_top=1.0,w_bal=0.5)
            if max_cos(tx)<bv: bx=tx.copy(); bv=max_cos(bx)
        except Exception: pass
        try:
            tx2,_=smooth_refine(bx,120.0,700.0,800,0.012)
            if max_cos(tx2)<bv: bx=tx2.copy(); bv=max_cos(bx)
        except Exception: pass
        try:
            pol=lp_wide(bx,remain*0.85,kmin=1100,wmin=2e-07)
            pv=max_cos(pol)
            if pv<bv: bx=pol.copy(); bv=pv
            if BUDGET-(time.time()-t0)>6:
                pol2=lp_wide(bx,4.0,kmin=1400,wmin=5e-07)
                pv2=max_cos(pol2)
                if pv2<bv: bx=pol2.copy(); bv=pv2
        except Exception: pass
        if bv<best_val: best_x=bx.copy(); best_val=bv
    best_x=norm_rows(best_x)
    return best_x

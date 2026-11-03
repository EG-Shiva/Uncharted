import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(X):
    n = np.sqrt((X*X).sum(axis=1, keepdims=True))
    n[n==0]=1.0
    return X/n

def _make_G(K, mults):
    G = np.zeros((K,8,8))
    for k in range(K):
        M = np.eye(8)
        for b,mm in enumerate(mults):
            ang = 2*np.pi*k*mm/K
            c=np.cos(ang); s=np.sin(ang)
            M[2*b,2*b]=c; M[2*b,2*b+1]=-s
            M[2*b+1,2*b]=s; M[2*b+1,2*b+1]=c
        G[k]=M
    return G

def _expand(S, Gs):
    K = Gs.shape[0]
    ns = S.shape[0]
    X = np.empty((K*ns, 8))
    for k in range(K):
        X[k*ns:(k+1)*ns] = S @ Gs[k].T
    return _norm(X)

def _adam_free(X, steps, beta0, beta1, lr0, lr1, rownorm):
    n,d = X.shape
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(1, steps+1):
        f=t/steps
        beta=np.exp(lb0+(lb1-lb0)*f)
        lr=np.exp(ll0+(ll1-ll0)*f)
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        gm=G.max()
        W=np.exp(beta*(G-gm))
        np.fill_diagonal(W,0.0)
        if rownorm:
            S=W.sum(axis=1,keepdims=True)+1e-12
            Grad=(W@X)/S
        else:
            Z=W.sum()+1e-12
            Grad=(W@X)/Z
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        m*=b1; m+=(1-b1)*Gt
        v*=b2; v+=(1-b2)*(Gt*Gt)
        mh=m/(1-b1**t); vh=v/(1-b2**t)
        X=X-lr*mh/(np.sqrt(vh)+eps)
        X=_norm(X)
    return X

def _hard(X, steps=250, lr0=0.003, lr1=0.00015, tol0=3e-4, tol1=2e-6):
    n,d=X.shape
    for s in range(steps):
        f=s/max(steps-1,1)
        tol=tol0+(tol1-tol0)*f
        lr=lr0*(lr1/lr0)**f if lr0>0 and lr1>0 else lr0
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        mx=G.max()
        M=(G>(mx-tol)).astype(np.float64)
        cnt=M.sum(axis=1,keepdims=True)+1e-12
        Grad=(M@X)/cnt
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        nn=np.sqrt((Gt*Gt).sum(axis=1,keepdims=True))+1e-12
        act=(cnt[:,0]>0.5)
        st=np.zeros_like(Gt)
        st[act]=Gt[act]/nn[act]
        X=X-lr*st
        X=_norm(X)
    return X

def _sym_quench(S0, Gs, steps, beta0, beta1, lr0, lr1):
    S=_norm(S0.copy())
    ns,d=S.shape
    K=Gs.shape[0]
    m=np.zeros_like(S); v=np.zeros_like(S)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(1,steps+1):
        f=t/steps
        beta=np.exp(lb0+(lb1-lb0)*f)
        lr=np.exp(ll0+(ll1-ll0)*f)
        X=_expand(S,Gs)
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        gm=G.max()
        W=np.exp(beta*(G-gm))
        np.fill_diagonal(W,0.0)
        St=W.sum(axis=1,keepdims=True)+1e-12
        Grad=(W@X)/St
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        Gt3=Gt.reshape(K,ns,d)
        GS=np.zeros_like(S)
        for k in range(K):
            GS+=Gt3[k]@Gs[k]
        dot2=(GS*S).sum(axis=1,keepdims=True)
        GtS=GS-dot2*S
        m*=b1; m+=(1-b1)*GtS
        v*=b2; v+=(1-b2)*(GtS*GtS)
        mh=m/(1-b1**t); vh=v/(1-b2**t)
        S=S-lr*mh/(np.sqrt(vh)+eps)
        S=_norm(S)
    return S

def _refine_hole(Y,p,iters=160,beta=90.0,lr0=0.12,lr1=0.004):
    p=p/np.linalg.norm(p)
    for it in range(iters):
        f=it/max(iters-1,1)
        lr=lr0*(lr1/lr0)**f
        g=Y@p
        mx=g.max()
        w=np.exp(beta*(g-mx))
        A=(w@Y)/(w.sum()+1e-12)
        At=A-float(A@p)*p
        p=p-lr*At
        nr=np.linalg.norm(p)
        if nr==0: break
        p/=nr
    return p

def _find_void(Y,rng,n_probe=8000):
    d=Y.shape[1]
    P=rng.randn(n_probe,d); P=_norm(P)
    bs=np.empty(n_probe)
    CH=4000
    for a in range(0,n_probe,CH):
        b=min(a+CH,n_probe)
        bs[a:b]=(P[a:b]@Y.T).max(axis=1)
    idx=np.argmin(bs)
    p=P[idx].copy()
    q=_refine_hole(Y,p)
    return q

def _e8_roots():
    pts=[]
    for i in range(8):
        for j in range(i+1,8):
            for s1 in (-1.0,1.0):
                for s2 in (-1.0,1.0):
                    v=np.zeros(8); v[i]=s1; v[j]=s2
                    pts.append(v)
    for mask in range(256):
        v=np.ones(8)*0.5
        nm=0
        for i in range(8):
            if (mask>>i)&1:
                v[i]=-0.5; nm+=1
        if nm%2==0:
            pts.append(v.copy())
    A=np.array(pts)/np.sqrt(2.0)
    A=_norm(A)
    return A

def _e8_delete(rng):
    A=_e8_roots()
    keep=np.arange(A.shape[0])
    cur=A.copy()
    while len(keep)>168:
        G=cur@cur.T
        np.fill_diagonal(G,-2)
        cnt=(G>0.49).sum(axis=1)
        j=int(np.argmax(cnt))
        # tie break random among max
        mx=cnt.max()
        ties=np.where(cnt==mx)[0]
        if len(ties)>1:
            j=int(ties[rng.randint(len(ties))])
        cur=np.delete(cur,j,axis=0)
        keep=np.delete(keep,j)
    return _norm(cur)

def _lp_polish(x, seconds, r0=1e-3):
    x=_norm(x.copy())
    n,d=x.shape
    best=_max_cos(x)
    radius=r0
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        c=(x@x.T)[iu]
        m=c.max()
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0: break
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y=_norm(y)
        vv=_max_cos(y)
        if vv<best:
            x,best,radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    BUDGET=175.0
    rng=np.random.RandomState(12345)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape!=(168,8): continue
            a=_norm(a)
            if not np.all(np.isfinite(a)): continue
            va=_max_cos(a)
            if va<best_val:
                best_val=va; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=_norm(rng.randn(168,8)); best_val=_max_cos(best)
    # ---- symmetry search ----
    configs=[(7,[1,2,3],24),(14,[1,3,5],12),(21,[1,2,4],8),(28,[1,3,5],6)]
    sym_best=None; sym_val=1e9; sym_Gs=None; sym_S=None
    sdead=t0+28.0
    for (K,mults,ns) in configs:
        if time.time()>sdead: break
        Gs=_make_G(K,mults)
        for trial in range(2):
            if time.time()>sdead: break
            S0=rng.randn(ns,8); S0=_norm(S0)
            try:
                S=_sym_quench(S0,Gs,steps=650,beta0=25.0,beta1=450.0,lr0=0.02,lr1=0.002)
            except Exception:
                continue
            X=_expand(S,Gs)
            v=_max_cos(X)
            if v<sym_val:
                sym_val=v; sym_best=X.copy(); sym_S=S.copy(); sym_Gs=Gs
    # E8 start
    try:
        Xe8=_e8_delete(np.random.RandomState(999))
        ve8=_max_cos(Xe8)
        if ve8<sym_val and sym_best is None:
            sym_best=Xe8.copy(); sym_val=ve8
    except Exception:
        Xe8=None
    # ---- staged release ----
    cand_list=[]
    if sym_best is not None:
        cand_list.append(sym_best)
    if Xe8 is not None:
        cand_list.append(Xe8)
    # release best symmetric through stages
    if sym_best is not None:
        Xcur=sym_best.copy()
        amps=[0.0,0.003,0.012,0.035]
        for ai,amp in enumerate(amps):
            if time.time()>t0+BUDGET-95: break
            if amp>0:
                Xp=Xcur+rng.randn(*Xcur.shape)*amp
                Xp=_norm(Xp)
            else:
                Xp=Xcur.copy()
            # quench
            b0=35+rng.rand()*20; b1=350+rng.rand()*450
            lr0=0.004+rng.rand()*0.004
            rn=bool(rng.rand()<0.5)
            try:
                Xq=_adam_free(Xp,steps=750,beta0=b0,beta1=b1,lr0=lr0,lr1=lr0*0.1,rownorm=rn)
                Xq=_hard(Xq,steps=220,lr0=0.003,lr1=0.00015)
            except Exception:
                continue
            vq=_max_cos(Xq)
            if vq<best_val:
                best_val=vq; best=Xq.copy()
            # chain from better of Xq / Xcur
            if vq<_max_cos(Xcur)+0.003:
                Xcur=Xq.copy()
            cand_list.append(Xcur.copy())
    # quick E8 quench
    if Xe8 is not None and time.time()<t0+BUDGET-90:
        try:
            Xp=_norm(Xe8+rng.randn(*Xe8.shape)*0.01)
            Xq=_adam_free(Xp,steps=800,beta0=40,beta1=500,lr0=0.005,lr1=0.0005,rownorm=True)
            Xq=_hard(Xq,steps=220,lr0=0.003,lr1=0.00015)
            vq=_max_cos(Xq)
            if vq<best_val:
                best_val=vq; best=Xq.copy()
            cand_list.append(Xq.copy())
        except Exception:
            pass
    # ---- free basin hopping mixing record + symmetric basins ----
    elite=[best.copy()]; elite_v=[best_val]
    for c in cand_list:
        try:
            vv=_max_cos(c)
            if vv<best_val+0.004:
                dup=False
                for ev in elite_v:
                    if abs(ev-vv)<1e-12: dup=True
                if not dup and len(elite)<4:
                    elite.append(c.copy()); elite_v.append(vv)
        except Exception:
            pass
    it=0
    while time.time()<t0+BUDGET-22:
        if time.time()+3.0>t0+BUDGET-20: break
        it+=1
        # pick start
        if rng.rand()<0.7:
            X=best.copy()
        else:
            X=elite[rng.randint(len(elite))].copy()
        n,d=X.shape
        r=rng.rand()
        if r<0.4:
            sig=rng.choice(np.array([0.006,0.012,0.025,0.05]))
            X=_norm(X+rng.randn(n,d)*sig)
            Xq=_adam_free(X,steps=800+int(rng.randint(600)),beta0=30+rng.rand()*30,beta1=300+rng.rand()*500,lr0=0.003+rng.rand()*0.005,lr1=0.0004,rownorm=bool(rng.rand()<0.5))
            Xq=_hard(Xq,steps=220,lr0=0.003,lr1=0.00015)
        elif r<0.75:
            G=X@X.T; np.fill_diagonal(G,-2.0)
            rm=G.max(axis=1)
            k=1+int(rng.randint(3))
            # bias to worst or rattlers
            u=rng.rand()
            if u<0.5:
                idx=np.argsort(rm)[-k:]
            elif u<0.8:
                idx=np.argsort(rm)[:k]
            else:
                idx=rng.choice(n,k,replace=False)
            for j in idx:
                others=np.delete(X,j,axis=0)
                try:
                    y=_find_void(others,rng,n_probe=6000+int(rng.randint(6000)))
                except Exception:
                    y=rng.randn(d); y/=np.linalg.norm(y)
                X[j]=y
            X=_norm(X)
            Xq=_adam_free(X,steps=900+int(rng.randint(600)),beta0=40,beta1=350+rng.rand()*450,lr0=0.004+rng.rand()*0.004,lr1=0.0004,rownorm=bool(rng.rand()<0.6))
            Xq=_hard(Xq,steps=250,lr0=0.003,lr1=0.00012)
        else:
            G=X@X.T; np.fill_diagonal(G,-2.0)
            m0=G.max()
            thr=m0-0.002
            deg=(G>thr).sum(axis=1)
            hub=int(np.argmax(deg))
            nb=int(min(n-1,max(4,deg[hub]+2)))
            order=np.argsort(-G[hub])
            idx=np.unique(np.concatenate([[hub],order[:nb]]))
            sig=0.03+rng.rand()*0.05
            X[idx]=_norm(X[idx]+rng.randn(len(idx),d)*sig)
            X=_norm(X)
            Xq=_adam_free(X,steps=900+int(rng.randint(500)),beta0=35+rng.rand()*25,beta1=300+rng.rand()*500,lr0=0.004+rng.rand()*0.004,lr1=0.0004,rownorm=bool(rng.rand()<0.5))
            Xq=_hard(Xq,steps=220,lr0=0.003,lr1=0.00015)
        if not np.all(np.isfinite(Xq)):
            continue
        vq=_max_cos(Xq)
        if vq<best_val:
            best_val=vq; best=Xq.copy()
            # update elite
            dup=False
            for ev in elite_v:
                if abs(ev-vq)<1e-12: dup=True
            if not dup:
                if len(elite)<4:
                    elite.append(Xq.copy()); elite_v.append(vq)
                else:
                    k=int(np.argmax(elite_v))
                    if vq<elite_v[k]:
                        elite_v[k]=vq; elite[k]=Xq.copy()
        elif vq<best_val+2e-4 and len(elite)<4:
            dup=False
            for ev in elite_v:
                if abs(ev-vq)<1e-12: dup=True
            if not dup:
                elite.append(Xq.copy()); elite_v.append(vq)
    # ---- final LP polish ----
    remain=(t0+BUDGET-1.5)-time.time()
    if remain>3:
        try:
            # polish best and elite, keep best
            cands=[best]+elite[:2]
            t_each=remain/len(cands)
            for c in cands:
                if time.time()+2>t0+BUDGET-1: break
                try:
                    Xp=_lp_polish(c,min(t_each,12.0),r0=8e-4)
                    vp=_max_cos(Xp)
                    if vp<best_val and np.all(np.isfinite(Xp)):
                        best_val=vp; best=Xp.copy()
                except Exception:
                    continue
        except Exception:
            pass
    best=_norm(best)
    return best

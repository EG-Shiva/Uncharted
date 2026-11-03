import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _build_pool(rng):
    P = []
    for i in range(10):
        for s in (1.0, -1.0):
            v = np.zeros(10); v[i] = s; P.append(v)
    for i in range(10):
        for j in range(i+1, 10):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(10); v[i]=si/np.sqrt(2.0); v[j]=sj/np.sqrt(2.0); P.append(v)
    # four-support random sample
    for _ in range(1200):
        idx = rng.choice(10, size=4, replace=False)
        s = rng.choice(np.array([-1.0,1.0]), size=4)
        v = np.zeros(10); v[idx]=s/2.0; P.append(v)
    # three-support all combos with signs (480*? subsample)
    import itertools
    for combo in itertools.combinations(range(10), 3):
        # limit: only some sign patterns to keep size ok
        for mask in range(8):
            v = np.zeros(10)
            for t,ix in enumerate(combo):
                v[ix]= 1.0 if ((mask>>t)&1)==0 else -1.0
            v/=np.sqrt(3.0); P.append(v)
    # random vectors
    R = rng.standard_normal(size=(1800,10))
    R = _norm(R)
    for v in R:
        P.append(v)
    # low-dim lattice embeddings
    def add_embed(k, Bset, nplane):
        # Bset (b, k) rows unit
        for _ in range(nplane):
            A = rng.standard_normal(size=(10,k))
            Q,_r = np.linalg.qr(A)
            for b in Bset:
                v = Q @ b
                P.append(v)
    # 2D circles
    for m in [6,8,10]:
        B = np.array([[np.cos(2*np.pi*i/m), np.sin(2*np.pi*i/m)] for i in range(m)])
        add_embed(2, B, 25)
    # 3D cube +/-1
    B3 = np.array([[a,b,c] for a in (-1.,1.) for b in (-1.,1.) for c in (-1.,1.)],float)
    B3 = _norm(B3)
    add_embed(3, B3, 20)
    # 4D cross + D4-ish
    B4 = []
    for i in range(4):
        for s in (-1.,1.):
            v=np.zeros(4); v[i]=s; B4.append(v)
    for i in range(4):
        for j in range(i+1,4):
            for si in (-1.,1.):
                for sj in (-1.,1.):
                    v=np.zeros(4); v[i]=si/np.sqrt(2); v[j]=sj/np.sqrt(2); B4.append(v)
    B4=np.array(B4)
    add_embed(4, B4, 15)
    # 5D patterns
    for th in [0.5,1.0]:
        a=np.cos(th)/np.sqrt(3.0); b=np.sin(th)/np.sqrt(2.0)
        # vector in 5d: (a,a,a,b,b)
        B5=np.zeros((1,5)); B5[0]=np.array([a,a,a,b,b])
        nrm=np.linalg.norm(B5[0]); B5[0]/=nrm
        add_embed(5, B5, 60)
    P=np.array(P,float)
    P=_norm(P)
    # cap size deterministically
    if len(P)>6500:
        idx=rng.choice(len(P),size=6500,replace=False)
        P=P[idx]
    return P

def _greedy_select_abs(P, k):
    M=P.shape[0]
    cur=np.full(M, np.inf)
    sel=[]
    s0=0
    sel.append(s0)
    d=np.abs(P@P[s0]); d[s0]=np.inf
    cur=np.minimum(cur,d)
    for _ in range(1,k):
        nxt=int(np.argmin(cur))
        sel.append(nxt)
        d=np.abs(P@P[nxt]); d[nxt]=np.inf
        # mask already selected
        for s in sel:
            d[s]=np.inf
        cur=np.minimum(cur,d)
    return np.array(sel,dtype=int)

def _adam_axes_target(Y, target, steps, lr, T0, T1, deadline=1e18):
    Y=_norm(Y.copy())
    m=np.zeros_like(Y); v=np.zeros_like(Y)
    b1,b2,eps=0.9,0.999,1e-8
    b1p,b2p=1.0,1.0
    if steps>1:
        logr=np.log(max(T1,1e-12)/max(T0,1e-12))/(steps-1)
    else:
        logr=0.0
    n77=Y.shape[0]
    for t in range(1,steps+1):
        if (t%150)==0 and time.time()>deadline:
            break
        T=T0*np.exp(logr*(t-1))
        if T<5e-4: T=5e-4
        X=np.vstack([Y,-Y])
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        # hinge: only near-violators
        thr=target-5.0*T
        mask=G>thr
        if not np.any(mask):
            # nothing to do, jitter slightly
            continue
        mv=float(G[mask].max())
        E=np.exp((G-mv)/T)*mask.astype(float)
        np.fill_diagonal(E,0.0)
        S=float(E.sum()*0.5)
        if not np.isfinite(S) or S<=0:
            continue
        GradX=(E@X)/S
        Gt_top=GradX[:n77]; Gt_bot=GradX[n77:]
        GradY=Gt_top-Gt_bot
        dot=(GradY*Y).sum(axis=1,keepdims=True)
        Gt=GradY-dot*Y
        b1p*=b1; b2p*=b2
        m=b1*m+(1-b1)*Gt
        vv=b2*v+(1-b2)*(Gt*Gt)
        v=vv
        mhat=m/(1-b1p); vhat=v/(1-b2p)
        step=mhat/(np.sqrt(vhat)+eps)
        Y=Y-lr*step
        Y=_norm(Y)
    return Y

def _swap_axes(Y, P, target, rng):
    Y=Y.copy()
    k=Y.shape[0]
    G=np.abs(Y@Y.T)
    np.fill_diagonal(G,0.0)
    viol=(G>target).sum(axis=1)
    maxa=G.max(axis=1)
    # worst order: most violations then largest max
    order=np.lexsort((-maxa,-viol))
    order=order[::-1]
    # try worst 4
    for wpos in range(min(4,k)):
        i=int(order[wpos])
        if viol[i]==0 and maxa[i]<target-0.005:
            continue
        remain=np.delete(Y,i,axis=0)
        D=P@remain.T
        ma=np.abs(D).max(axis=1)
        # exclude pool vectors already present (close to any Y)
        Dall=np.abs(P@Y.T).max(axis=1)
        # allow if not duplicate
        ma2=ma.copy()
        ma2[Dall>0.999]=np.inf
        j=int(np.argmin(ma2))
        best=float(ma2[j])
        cur=float(maxa[i])
        if best<cur-1e-4:
            Y[i]=P[j]/np.linalg.norm(P[j])
    return _norm(Y)

def _adam_full(X, steps, lr, T0, T1, deadline=1e18):
    X=_norm(X.copy())
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1,b2,eps=0.9,0.999,1e-8
    b1p,b2p=1.0,1.0
    if steps>1:
        logr=np.log(max(T1,1e-12)/max(T0,1e-12))/(steps-1)
    else:
        logr=0.0
    for t in range(1,steps+1):
        if (t%250)==0 and time.time()>deadline:
            break
        T=T0*np.exp(logr*(t-1))
        if T<8e-4: T=8e-4
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        mv=float(G.max())
        E=np.exp((G-mv)/T)
        np.fill_diagonal(E,0.0)
        S=float(E.sum()*0.5)
        if not np.isfinite(S) or S<=0:
            break
        Grad=(E@X)/S
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        b1p*=b1; b2p*=b2
        m=b1*m+(1-b1)*Gt
        v=b2*v+(1-b2)*(Gt*Gt)
        mhat=m/(1-b1p); vhat=v/(1-b2p)
        step=mhat/(np.sqrt(vhat)+eps)
        X=X-lr*step
        X=_norm(X)
    return X

def _lp_polish(x, seconds, deadline=1e18):
    x=_norm(x.copy())
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    itn=0
    while time.time()-start<seconds and time.time()<deadline:
        if radius<1e-14 or itn>=30:
            break
        itn+=1
        c=(x@x.T)[iu]
        m=float(c.max())
        thr=max(20*radius,1e-9)
        sel=c>m-thr
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            break
        if p>2500:
            order=np.argsort(-c[sel])[:2500]
            I,J=I[order],J[order]
            cvals=c[sel][order]
        else:
            cvals=c[sel]
            p=len(I)
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-cvals,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
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
    DEAD=t0+172.0
    rng=np.random.default_rng(12345)
    n,d=154,10
    # load incumbents for fallback comparison
    inc=None; inc_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            if a.shape==(n,d):
                a=_norm(a.astype(float))
                v=_max_cos(a)
                if v<inc_val:
                    inc_val=v; inc=a.copy()
        except Exception:
            pass
    # structured pool + greedy 77 axes
    P=_build_pool(rng)
    sel=_greedy_select_abs(P,77)
    Y=_norm(P[sel].copy())
    targets=[0.42,0.415,0.41,0.405,0.40,0.397,0.395]
    # symmetric stage
    SYM_END=t0+80.0
    for ti,tgt in enumerate(targets):
        if time.time()>SYM_END or time.time()>DEAD-60:
            break
        # anneal this target
        Y=_adam_axes_target(Y,tgt,steps=550,lr=0.02,T0=0.010,T1=0.0025,deadline=SYM_END)
        Y=_swap_axes(Y,P,tgt,rng)
        # second quick refine at same target with lower T
        Y=_adam_axes_target(Y,tgt,steps=250,lr=0.012,T0=0.004,T1=0.0015,deadline=SYM_END)
        Y=_swap_axes(Y,P,tgt,rng)
    Xsym=np.vstack([Y,-Y])
    Xsym=_norm(Xsym)
    # release symmetry with jitter
    Xrel=Xsym+rng.standard_normal(size=Xsym.shape)*0.002
    Xrel=_norm(Xrel)
    if time.time()<DEAD-45:
        Xrel=_adam_full(Xrel,steps=1200,lr=0.02,T0=0.015,T1=0.0025,deadline=DEAD-30)
    if time.time()<DEAD-30:
        Xrel=_adam_full(Xrel,steps=600,lr=0.012,T0=0.003,T1=0.0009,deadline=DEAD-22)
    # LP polish released
    if time.time()<DEAD-20:
        Xrel=_lp_polish(Xrel,seconds=min(18.0,DEAD-5-time.time()),deadline=DEAD-3)
    vrel=_max_cos(Xrel)
    # also lightly polish incumbent if time allows to have fair comparison (short)
    best_x=Xrel; best_v=vrel
    if inc is not None:
        if inc_val<best_v:
            # try to improve incumbent a bit with short polish only if time left
            if time.time()<DEAD-8:
                inc2=_lp_polish(inc,seconds=min(6.0,DEAD-3-time.time()),deadline=DEAD-2)
                v2=_max_cos(inc2)
                if v2<inc_val:
                    inc_val=v2; inc=inc2
            if inc_val<best_v:
                best_x=inc; best_v=inc_val
    # final safety: ensure normalized, no nan
    best_x=_norm(best_x)
    # tiny shrink not needed; return
    return best_x

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_Qs(kind):
    Qs = []
    if kind == 7:
        for k in range(7):
            Q = np.eye(8)
            for a, b, f in [(0,1,1),(2,3,2),(4,5,3)]:
                ang = 2*np.pi*f*k/7.0
                c, s = np.cos(ang), np.sin(ang)
                Q[a,a]=c; Q[a,b]=-s; Q[b,a]=s; Q[b,b]=c
            Qs.append(Q)
        return Qs
    if kind == 14:
        base = _build_Qs(7)
        return base + [-Q for Q in base]
    if kind == 21:
        for i in range(7):
            for j in range(3):
                Q = np.eye(8)
                for a,b,f in [(0,1,1),(2,3,2),(4,5,3)]:
                    ang = 2*np.pi*f*i/7.0
                    c,s = np.cos(ang), np.sin(ang)
                    Q[a,a]=c; Q[a,b]=-s; Q[b,a]=s; Q[b,b]=c
                ang2 = 2*np.pi*j/3.0
                c2,s2 = np.cos(ang2), np.sin(ang2)
                Q[6,6]=c2; Q[6,7]=-s2; Q[7,6]=s2; Q[7,7]=c2
                Qs.append(Q)
        return Qs
    if kind == 28:
        for i in range(7):
            for j in range(4):
                Q = np.eye(8)
                for a,b,f in [(0,1,1),(2,3,2),(4,5,3)]:
                    ang = 2*np.pi*f*i/7.0
                    c,s = np.cos(ang), np.sin(ang)
                    Q[a,a]=c; Q[a,b]=-s; Q[b,a]=s; Q[b,b]=c
                ang2 = 2*np.pi*j/4.0
                c2,s2 = np.cos(ang2), np.sin(ang2)
                Q[6,6]=c2; Q[6,7]=-s2; Q[7,6]=s2; Q[7,7]=c2
                Qs.append(Q)
        return Qs
    raise ValueError

def _expand(S, Qs):
    return np.vstack([S @ Q for Q in Qs])

def _sym_quench(S, Qs, steps, beta0, beta1, lr0, lr1, t_lim=None):
    ns, d = S.shape
    G = len(Qs)
    Qa = np.stack(Qs)
    QaT = Qa.transpose(0,2,1)
    m = np.zeros_like(S); v = np.zeros_like(S)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(1, steps+1):
        if t_lim is not None and (t & 127)==0 and time.time()>t_lim:
            break
        frac = t/steps
        beta = np.exp(lb0+(lb1-lb0)*frac)
        lr = np.exp(ll0+(ll1-ll0)*frac)
        X = np.vstack([S @ Q for Q in Qa])
        Gr = X @ X.T
        np.fill_diagonal(Gr, -10.0)
        gmax = Gr.max()
        W = np.exp(beta*(Gr-gmax))
        np.fill_diagonal(W, 0.0)
        Sr = W.sum(axis=1, keepdims=True)+1e-18
        Grad = (W @ X)/Sr
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        Gs = np.zeros_like(S)
        for gi in range(G):
            blk = Gt[gi*ns:(gi+1)*ns]
            Gs += blk @ QaT[gi]
        Gs /= float(G)
        dot2 = (Gs*S).sum(axis=1, keepdims=True)
        Gs = Gs - dot2*S
        m *= b1; m += (1-b1)*Gs
        v *= b2; v += (1-b2)*(Gs*Gs)
        bc1=1-b1**t; bc2=1-b2**t
        step = (m/bc1)/(np.sqrt(v/bc2)+eps)
        S = S - lr*step
        S /= np.sqrt((S*S).sum(axis=1,keepdims=True))
    return S

def _free_quench(X, steps, beta0, beta1, lr0, lr1, t_lim=None):
    n,d = X.shape
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(1, steps+1):
        if t_lim is not None and (t & 127)==0 and time.time()>t_lim:
            break
        frac=t/steps
        beta=np.exp(lb0+(lb1-lb0)*frac)
        lr=np.exp(ll0+(ll1-ll0)*frac)
        Gr=X@X.T
        np.fill_diagonal(Gr,-10.0)
        gmax=Gr.max()
        W=np.exp(beta*(Gr-gmax))
        np.fill_diagonal(W,0.0)
        Sr=W.sum(axis=1,keepdims=True)+1e-18
        Grad=(W@X)/Sr
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        m*=b1; m+=(1-b1)*Gt
        v*=b2; v+=(1-b2)*(Gt*Gt)
        bc1=1-b1**t; bc2=1-b2**t
        step=(m/bc1)/(np.sqrt(v/bc2)+eps)
        X=X-lr*step
        X/=np.sqrt((X*X).sum(axis=1,keepdims=True))
    return X

def _hard(X, steps=250, lr0=0.003, lr1=0.00015, tol0=3e-4, tol1=2e-6):
    n,d=X.shape
    for s in range(steps):
        frac=s/max(steps-1,1)
        tol=tol0+(tol1-tol0)*frac
        lr=lr0*(lr1/lr0)**frac
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        mx=float(G.max())
        M=(G>(mx-tol)).astype(np.float64)
        np.fill_diagonal(M,0.0)
        cnt=M.sum(axis=1,keepdims=True)+1e-12
        act=(cnt[:,0]>0.5)
        if not np.any(act):
            continue
        Grad=(M@X)/cnt
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        nn=np.sqrt((Gt*Gt).sum(axis=1,keepdims=True))+1e-12
        st=np.zeros_like(Gt)
        st[act]=Gt[act]/nn[act]
        X=X-lr*st
        X/=np.linalg.norm(X,axis=1,keepdims=True)
    return X

def _refine_hole(Y,p,iters=160,beta=90.0,lr0=0.1,lr1=0.003):
    p=p/np.linalg.norm(p)
    for it in range(iters):
        frac=it/max(iters-1,1)
        lr=lr0*(lr1/lr0)**frac
        g=Y@p
        gm=g.max()
        w=np.exp(beta*(g-gm))
        s=w.sum()+1e-18
        A=(w@Y)/s
        At=A-float(A@p)*p
        p=p-lr*At
        nr=np.linalg.norm(p)
        if nr==0: break
        p/=nr
    return p

def _find_void(Y,rng,n_probe=8000):
    d=Y.shape[1]
    P=rng.randn(n_probe,d)
    P/=np.linalg.norm(P,axis=1,keepdims=True)
    CH=4000
    bs=np.empty(n_probe)
    for a in range(0,n_probe,CH):
        b=min(a+CH,n_probe)
        bs[a:b]=(P[a:b]@Y.T).max(axis=1)
    order=np.argsort(bs)
    cands=[]
    for idx in order[:80]:
        p=P[idx]
        ok=True
        for q in cands:
            if abs(float(p@q))>0.9:
                ok=False; break
        if ok:
            cands.append(p)
        if len(cands)>=6: break
    best_p=cands[0]; best_v=1e9
    for p in cands:
        q=_refine_hole(Y,p,iters=160,beta=80.0+rng.rand()*40.0)
        vv=float((Y@q).max())
        if vv<best_v:
            best_v=vv; best_p=q
    return best_p

def _lp(x, seconds, r0=1e-3):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    best=_max_cos(x)
    radius=r0
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        c=(x@x.T)[iu]
        m=float(c.max())
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0: break
        if p>2500:
            idx=np.argsort(c[sel])[-2500:]
            I,J=I[idx],J[idx]
            cc=c[sel][idx]
        else:
            cc=c[sel]
            p=len(I)
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cc,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
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

def _e8():
    pts=[]
    for i in range(8):
        for j in range(i+1,8):
            for si in (1.0,-1.0):
                for sj in (1.0,-1.0):
                    v=np.zeros(8); v[i]=si; v[j]=sj; pts.append(v)
    for mask in range(256):
        v=np.ones(8)*0.5; nm=0
        for k in range(8):
            if (mask>>k)&1:
                v[k]=-0.5; nm+=1
        if nm%2==0:
            pts.append(v.copy())
    A=np.array(pts,float)
    A/=np.linalg.norm(A,axis=1,keepdims=True)
    return A

def _e8_subset():
    E=_e8()
    G=E@E.T
    act=np.ones(len(E),bool)
    for _ in range(len(E)-168):
        idx=np.where(act)[0]
        sub=G[np.ix_(idx,idx)]
        tight=(sub>0.499).astype(np.int32)
        np.fill_diagonal(tight,0)
        deg=tight.sum(axis=1)
        kill=idx[int(np.argmax(deg))]
        act[kill]=False
    return E[act]

def solve():
    t0=time.time(); BUDGET=175.0
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape!=(168,8): continue
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if not np.all(np.isfinite(a)): continue
            va=_max_cos(a)
            if va<best_val: best_val=va; best=a.copy()
        except Exception: pass
    if best is None:
        r0=np.random.RandomState(7)
        best=r0.randn(168,8); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    rng=np.random.RandomState(999)
    # symmetric search
    sym_cands=[]
    kinds=[(7,24),(14,12),(21,8),(28,6)]
    t_sym_lim=t0+42.0
    for kind,ns in kinds:
        Qs=_build_Qs(kind)
        for rep in range(2):
            if time.time()>t_sym_lim: break
            S=rng.randn(ns,8); S/=np.linalg.norm(S,axis=1,keepdims=True)
            try:
                S=_sym_quench(S,Qs,steps=650,beta0=18.0,beta1=320.0,lr0=0.012,lr1=0.0012,t_lim=t_sym_lim)
                S=_sym_quench(S,Qs,steps=550,beta0=120.0,beta1=700.0,lr0=0.004,lr1=0.0006,t_lim=t_sym_lim)
            except Exception: continue
            try:
                X=_expand(S,Qs)
                X/=np.linalg.norm(X,axis=1,keepdims=True)
                v=_max_cos(X)
                sym_cands.append((v,X.copy(),kind))
            except Exception: pass
            if time.time()>t_sym_lim: break
        if time.time()>t_sym_lim: break
    # E8 subset
    try:
        Xe=_e8_subset()
        ve=_max_cos(Xe)
        sym_cands.append((ve,Xe.copy(),-1))
    except Exception: pass
    sym_cands.sort(key=lambda t:t[0])
    top=[c[1] for c in sym_cands[:3]]
    # quick release of symmetric tops
    for X0 in list(top):
        if time.time()>t0+BUDGET-95: break
        try:
            X=X0.copy()
            X=_free_quench(X,steps=1100,beta0=30.0,beta1=450.0,lr0=0.005,lr1=0.0005,t_lim=t0+BUDGET-20)
            X=_hard(X,steps=220,lr0=0.003,lr1=0.00015)
            v=_max_cos(X)
            if v<best_val and np.all(np.isfinite(X)):
                best_val=v; best=X.copy()
            # staged larger kick
            for sig in (1e-3,1e-2,5e-2):
                if time.time()>t0+BUDGET-90: break
                X2=X0+rng.randn(168,8)*sig
                X2/=np.linalg.norm(X2,axis=1,keepdims=True)
                X2=_free_quench(X2,steps=700,beta0=30.0,beta1=400.0,lr0=0.005,lr1=0.0005,t_lim=t0+BUDGET-20)
                X2=_hard(X2,steps=150,lr0=0.0025,lr1=0.00012)
                v2=_max_cos(X2)
                if v2<best_val:
                    best_val=v2; best=X2.copy()
        except Exception: pass
    # main hopping mixing best basin and symmetric basins
    it=0
    while time.time()<t0+BUDGET-22:
        it+=1
        if time.time()+3.0>t0+BUDGET-20: break
        u=rng.rand()
        try:
            if u<0.5:
                X=best.copy()
                r=rng.rand()
                if r<0.35:
                    sig=rng.choice(np.array([0.006,0.012,0.025,0.05]))
                    X=X+rng.randn(168,8)*float(sig)
                    X/=np.linalg.norm(X,axis=1,keepdims=True)
                elif r<0.75:
                    Gc=X@X.T; np.fill_diagonal(Gc,-2.0)
                    rm=Gc.max(axis=1)
                    k=1+int(rng.randint(3))
                    idx=np.argsort(rm)[-k:]
                    for j in idx:
                        oth=np.delete(X,j,axis=0)
                        X[j]=_find_void(oth,rng,n_probe=6000+int(rng.randint(6000)))
                    X/=np.linalg.norm(X,axis=1,keepdims=True)
                else:
                    Gc=X@X.T; np.fill_diagonal(Gc,-2.0)
                    m=Gc.max(); deg=(Gc>m-0.002).sum(axis=1)
                    hub=int(np.argmax(deg))
                    order=np.argsort(-Gc[hub])
                    nb=int(min(167,max(5,int(deg[hub])+2)))
                    idx=np.unique(np.concatenate([[hub],order[:nb]]))
                    X[idx]+=rng.randn(len(idx),8)*(0.03+rng.rand()*0.04)
                    X/=np.linalg.norm(X,axis=1,keepdims=True)
                X=_free_quench(X,steps=900+int(rng.randint(700)),beta0=30.0+rng.rand()*20.0,beta1=350.0+rng.rand()*400.0,lr0=0.004+rng.rand()*0.003,lr1=0.0005,t_lim=t0+BUDGET-18)
                X=_hard(X,steps=220,lr0=0.003,lr1=0.00013)
            else:
                if len(top)==0:
                    X=best.copy()+rng.randn(168,8)*0.02
                    X/=np.linalg.norm(X,axis=1,keepdims=True)
                else:
                    src=top[rng.randint(len(top))].copy()
                    sig=rng.choice(np.array([0.001,0.01,0.05,0.02]))
                    src=src+rng.randn(168,8)*float(sig)
                    src/=np.linalg.norm(src,axis=1,keepdims=True)
                    # occasionally void-relocate worst of src
                    if rng.rand()<0.4:
                        Gc=src@src.T; np.fill_diagonal(Gc,-2.0)
                        rm=Gc.max(axis=1); j=int(np.argmax(rm))
                        oth=np.delete(src,j,axis=0)
                        try: src[j]=_find_void(oth,rng,n_probe=6000)
                        except Exception: pass
                    X=_free_quench(src,steps=900+int(rng.randint(700)),beta0=30.0,beta1=400.0+rng.rand()*350.0,lr0=0.004+rng.rand()*0.003,lr1=0.0005,t_lim=t0+BUDGET-18)
                    X=_hard(X,steps=220,lr0=0.003,lr1=0.00013)
            if not np.all(np.isfinite(X)): continue
            v=_max_cos(X)
            if v<best_val:
                best_val=v; best=X.copy()
        except Exception:
            continue
    # final LP polish
    try:
        remain=(t0+BUDGET-1.5)-time.time()
        if remain>3:
            Xp=_lp(best,min(remain,14.0),r0=8e-4)
            vp=_max_cos(Xp)
            if np.all(np.isfinite(Xp)) and vp<best_val:
                best=Xp; best_val=vp
    except Exception: pass
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best

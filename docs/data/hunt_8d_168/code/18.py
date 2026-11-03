import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_opt(X, steps, a0, a1, lr0, lr1, mu=0.85):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best = X.copy()
    V = np.zeros_like(X)
    om = 1.0 - mu
    for s in range(steps):
        frac = s / max(steps-1, 1)
        alpha = a0 * (a1/a0)**frac
        lr = lr0 * (lr1/lr0)**frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        E = np.exp(alpha*(G-m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()*0.5
        if S == 0 or not np.isfinite(S):
            continue
        Grad = (E @ X)/S
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        V = mu*V + om*Gt
        X -= lr*V
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn==0]=1.0
        X /= nn
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_m:
        best = X.copy()
    return best

def _lp_polish(x, seconds):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start < seconds and radius>1e-14:
        c=(x@x.T)[iu]
        m=float(c.max())
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            radius*=0.5
            continue
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv);obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best:
            x,best=y,v
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _build_E8():
    vecs=[]
    for i in range(8):
        for j in range(i+1,8):
            for si in (1.0,-1.0):
                for sj in (1.0,-1.0):
                    v=np.zeros(8);v[i]=si;v[j]=sj;vecs.append(v)
    for mask in range(256):
        cnt=0;s=np.ones(8)
        for k in range(8):
            if (mask>>k)&1:
                s[k]=-1.0;cnt+=1
        if cnt%2==0:
            vecs.append(0.5*s)
    E=np.array(vecs,float);E/=np.sqrt(2.0)
    return E

def _deep_hole(keep, n_try=5000, top=10, steps=220, b0=25.0, b1=350.0, l0=0.06, l1=0.004):
    m=keep.shape[0]
    C=np.random.randn(n_try,8);C/=np.linalg.norm(C,axis=1,keepdims=True)
    scores=(C@keep.T).max(axis=1)
    order=np.argsort(scores)[:top]
    best_s=np.inf;best_y=None
    lb0=np.log(b0);lb1=np.log(b1);ll0=np.log(l0);ll1=np.log(l1)
    for ci in order:
        y=C[ci].copy()
        for s in range(steps):
            frac=s/max(steps-1,1)
            beta=np.exp(lb0+(lb1-lb0)*frac)
            lr=np.exp(ll0+(ll1-ll0)*frac)
            d=keep@y
            dm=float(d.max())
            w=np.exp(beta*(d-dm));S=w.sum()
            if S==0 or not np.isfinite(S):
                break
            w/=S
            g=w@keep
            gt=g-np.dot(g,y)*y
            y-=lr*gt
            nrm=np.linalg.norm(y)
            if nrm==0:
                break
            y/=nrm
        sf=float((keep@y).max())
        if sf<best_s:
            best_s=sf;best_y=y.copy()
    return best_y

def _farthest_reinsert(keep,k):
    cur=keep.copy()
    outs=[]
    for _ in range(k):
        y=_deep_hole(cur,n_try=5000,top=8,steps=200)
        if y is None:
            y=np.random.randn(8);y/=np.linalg.norm(y)
        outs.append(y)
        cur=np.vstack([cur,y.reshape(1,8)])
    return np.array(outs)

def _e8_greedy_subset(E8):
    rem=list(range(E8.shape[0]))
    # remove 72 points greedily, randomised among top-5
    for _ in range(72):
        sub=E8[rem]
        G=sub@sub.T
        np.fill_diagonal(G,-2.0)
        cnt=(G>0.49).sum(axis=1)
        mx=G.max(axis=1)
        score=cnt.astype(float)+0.2*mx
        order=np.argsort(score)  # ascending, worst at end
        # pick among 5 worst at random
        cand=order[-5:]
        pick=cand[np.random.randint(len(cand))]
        # map back to rem index
        del rem[int(pick)]
    return E8[rem].copy()

def solve():
    np.random.seed(7)
    t0=time.time()
    TOTAL=172.0
    try:
        Xb=np.load("best.npy").astype(float).reshape(168,8)
    except Exception:
        try:
            Xb=np.load("record.npy").astype(float).reshape(168,8)
        except Exception:
            Xb=np.random.randn(168,8)
    Xb/=np.linalg.norm(Xb,axis=1,keepdims=True)
    try:
        Xr=np.load("record.npy").astype(float).reshape(168,8)
        Xr/=np.linalg.norm(Xr,axis=1,keepdims=True)
        if _max_cos(Xr)<_max_cos(Xb):
            Xb=Xr.copy()
    except Exception:
        pass
    best_val=_max_cos(Xb)
    Xcur=Xb.copy();cur_val=best_val
    E8=_build_E8()
    T=0.0025
    it=0
    while time.time()-t0 < TOTAL-34.0:
        it+=1
        r=np.random.rand()
        if r<0.28:
            # E8 greedy decimation topology
            try:
                Xs=_e8_greedy_subset(E8)
            except Exception:
                idx=np.random.choice(240,168,replace=False)
                Xs=E8[idx].copy()
            Xs=Xs+0.012*np.random.randn(168,8)
            if np.random.rand()<0.25:
                Q,_=np.linalg.qr(np.random.randn(8,8))
                if np.linalg.det(Q)<0: Q[:,0]=-Q[:,0]
                Xs=Xs@Q.T
            Xs/=np.linalg.norm(Xs,axis=1,keepdims=True)
            steps,a0,a1,lr0,lr1,mu=1700,12.0,450.0,0.05,0.0022,0.86
            Xinit=Xs
        elif r<0.50:
            # hub ejection + neighbour shake + deep hole
            base=Xcur if np.random.rand()<0.6 else Xb
            G=base@base.T;np.fill_diagonal(G,-2.0)
            m=float(G.max())
            mx=G.max(axis=1)
            cnt=(G>m-1e-7).sum(axis=1)
            score=mx+0.0018*cnt
            k=int(np.random.choice([1,2,3,4],p=[0.35,0.30,0.20,0.15]))
            worst=np.argsort(score)[-k:]
            mask=np.ones(168,bool);mask[worst]=False
            keep=base[mask]
            # neighbours of ejected: points close to any ejected
            try:
                Ge=base[worst]@base.T
                neigh=np.unique(np.argsort(Ge,axis=1)[:,-12:].ravel())
                neigh=[i for i in neigh if i not in set(worst.tolist())]
            except Exception:
                neigh=[]
            try:
                newp=_farthest_reinsert(keep,k)
            except Exception:
                newp=np.random.randn(k,8);newp/=np.linalg.norm(newp,axis=1,keepdims=True)
            Xinit=np.vstack([keep,newp])
            # shake: global small + neighbour larger
            Xinit+=0.003*np.random.randn(168,8)
            if len(neigh)>0:
                # map neigh indices (old) to new positions: keep order preserved except ejected removed
                # simplest: add extra noise to random subset of Xinit
                nn=min(len(neigh),14)
                sel=np.random.choice(168,nn,replace=False)
                Xinit[sel]+=0.035*np.random.randn(nn,8)
            Xinit/=np.linalg.norm(Xinit,axis=1,keepdims=True)
            steps,a0,a1,lr0,lr1,mu=1750,12.0,500.0,0.05,0.002,0.86
        elif r<0.56 and it>2:
            # pure random fresh basin (rare)
            Xinit=np.random.randn(168,8);Xinit/=np.linalg.norm(Xinit,axis=1,keepdims=True)
            steps,a0,a1,lr0,lr1,mu=2000,10.0,350.0,0.06,0.003,0.85
        else:
            # gaussian kick of best/cur
            base=Xcur if np.random.rand()<0.65 else Xb
            sigma=float(np.random.choice([0.004,0.008,0.015,0.03,0.06,0.12]))
            Xinit=base+sigma*np.random.randn(168,8)
            if np.random.rand()<0.55:
                G=base@base.T;np.fill_diagonal(G,-2.0)
                mx=G.max(axis=1);cnt=(G>float(G.max())-1e-9).sum(axis=1)
                score=mx+0.002*cnt
                worst=np.argsort(score)[-6:]
                Xinit[worst]+=np.random.randn(6,8)*sigma*2.5
            Xinit/=np.linalg.norm(Xinit,axis=1,keepdims=True)
            if sigma>=0.06:
                steps,a0,a1,lr0,lr1,mu=2000,10.0,450.0,0.05,0.002,0.88
            else:
                steps,a0,a1,lr0,lr1,mu=1400,18.0,520.0,0.045,0.002,0.85
        try:
            Xopt=_smooth_opt(Xinit,steps,a0,a1,lr0,lr1,mu=mu)
        except Exception:
            continue
        v=_max_cos(Xopt)
        if v<best_val:
            best_val=v;Xb=Xopt.copy();Xcur=Xopt.copy();cur_val=v
        else:
            if v<cur_val or np.random.rand()<np.exp(-(v-cur_val)/max(T,1e-7)):
                Xcur=Xopt.copy();cur_val=v
        T=max(T*0.997,0.0002)
        if time.time()-t0>TOTAL-36:
            break
    try:
        if time.time()-t0<TOTAL-30:
            Xb=_smooth_opt(Xb,1000,80.0,900.0,0.012,0.001,mu=0.7)
    except Exception:
        pass
    remain=TOTAL-(time.time()-t0)-1.0
    if remain>4:
        try:
            Xb=_lp_polish(Xb,seconds=min(remain,28.0))
        except Exception:
            pass
    Xb/=np.linalg.norm(Xb,axis=1,keepdims=True)
    return Xb

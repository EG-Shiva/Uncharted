import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

# single-point hole refinement (softmax)
def _refine_hole(Xf, y, steps=150, a0=60.0, a1=350.0, lr0=0.06, lr1=0.005):
    la0=np.log(a0); la1=np.log(a1); ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(steps):
        f=t/max(steps-1,1)
        a=np.exp(la0+(la1-la0)*f); lr=np.exp(ll0+(ll1-ll0)*f)
        s=Xf @ y
        m=np.max(s)
        w=np.exp(a*(s-m))
        sw=np.sum(w)+1e-300
        g=w @ Xf
        gt=g-np.dot(g,y)*y
        y=y-lr*gt/sw
        n=np.linalg.norm(y)
        if not np.isfinite(n) or n==0:
            break
        y=y/n
    return y

def _best_hole(Xf, rng, n_sample=50000):
    d=Xf.shape[1]
    # split to bound memory
    best_y=None; best_m=np.inf
    done=0
    while done<n_sample:
        bs=min(10000, n_sample-done)
        Y=rng.standard_normal(size=(bs,d))
        Y/=np.linalg.norm(Y,axis=1,keepdims=True)+1e-300
        S=Y @ Xf.T
        m=S.max(axis=1)
        i=int(np.argmin(m))
        if float(m[i])<best_m:
            best_m=float(m[i]); best_y=Y[i].copy()
        done+=bs
    return best_y, best_m

# subset refine: only active move, repulsion from all
def _subset_refine(X, active, steps=110, a0=50.0, a1=320.0, lr0=0.03, lr1=0.003):
    X=X.copy()
    la0=np.log(a0); la1=np.log(a1); ll0=np.log(lr0); ll1=np.log(lr1)
    act=np.asarray(list(active),dtype=np.int64)
    if len(act)==0:
        return X
    for t in range(steps):
        f=t/max(steps-1,1)
        a=np.exp(la0+(la1-la0)*f); lr=np.exp(ll0+(ll1-ll0)*f)
        Xa=X[act]  # A x d
        G=Xa @ X.T  # A x N
        # ignore self: set self columns to -inf via mask
        # for active rows, self-similarity=1; exclude it
        for ii,gi in enumerate(act):
            G[ii,gi]=-np.inf
        mx=np.max(G,axis=1,keepdims=True)
        W=np.exp(a*(G-mx))
        S=W.sum(axis=1,keepdims=True)+1e-300
        grad=W @ X  # A x d
        dot=np.sum(grad*Xa,axis=1,keepdims=True)
        gt=grad-dot*Xa
        direc=gt/S
        Xa2=Xa-lr*direc
        Xa2/=np.linalg.norm(Xa2,axis=1,keepdims=True)+1e-300
        X[act]=Xa2
    return X

def _global_anneal(X, steps=250, a0=45.0, a1=320.0, lr0=0.028, lr1=0.0025):
    X=X.copy()
    la0=np.log(a0); la1=np.log(a1); ll0=np.log(lr0); ll1=np.log(lr1)
    n=X.shape[0]
    for t in range(steps):
        f=t/max(steps-1,1)
        a=np.exp(la0+(la1-la0)*f); lr=np.exp(ll0+(ll1-ll0)*f)
        G=X @ X.T
        np.fill_diagonal(G,-np.inf)
        mx=np.max(G)
        W=np.exp(a*(G-mx))
        np.fill_diagonal(W,0.0)
        S=W.sum(axis=1,keepdims=True)+1e-300
        grad=W @ X
        dot=np.sum(grad*X,axis=1,keepdims=True)
        gt=grad-dot*X
        direc=gt/S
        X=X-lr*direc
        X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-300
    return X

def _lp_polish(x, seconds, deadline):
    x=_norm(x.copy())
    n,d=x.shape
    best=_max_cos(x)
    xb=x.copy()
    radius=1e-3
    t0=time.time()
    iu=np.triu_indices(n,1)
    ar=np.arange(d)
    while time.time()-t0<seconds and time.time()<deadline-1.0 and radius>1e-13:
        G=x @ x.T
        cf=G[iu]
        m=float(cf.max())
        thr=m-max(20.0*radius,1e-9)
        sel=cf>thr
        I=iu[0][sel]; J=iu[1][sel]; cs=cf[sel]
        p=len(I)
        if p==0:
            radius*=0.5; continue
        if p>2000:
            idx=np.argpartition(-cs,2000)[:2000]
            I,J,cs=I[idx],J[idx],cs[idx]; p=2000
        nv=n*d+1
        rows=np.concatenate([np.repeat(np.arange(p),d),np.repeat(np.arange(p),d),np.arange(p)])
        cols=np.concatenate([np.repeat(I*d,d)+np.tile(ar,p),np.repeat(J*d,d)+np.tile(ar,p),np.full(p,n*d)])
        vals=np.concatenate([x[J].ravel(),x[I].ravel(),-np.ones(p)])
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cs,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=_norm(x+res.x[:-1].reshape(n,d))
        v=_max_cos(y)
        if v<best:
            best=v; xb=y.copy(); x=y
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return xb, best

def solve():
    t0=time.time()
    DEADLINE=t0+170.0
    rng=np.random.default_rng(0)
    n,d=184,8
    cands=[]
    for f in ("best.npy","record.npy"):
        try:
            c=np.asarray(np.load(f),dtype=np.float64)
            if c.shape==(n,d):
                c=_norm(c); cands.append((_max_cos(c),c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t:t[0])
        base_val, base = cands[0][0], cands[0][1].copy()
    else:
        base=_norm(rng.standard_normal(size=(n,d))); base_val=_max_cos(base)
    gbest=base.copy(); gval=float(base_val)

    # contact graph analysis
    G=gbest @ gbest.T
    np.fill_diagonal(G,-2.0)
    m=float(np.max(G))
    tol=1e-7
    Adj=(G>m-tol)
    np.fill_diagonal(Adj,False)
    deg=Adj.sum(axis=1).astype(int)
    pm=G.max(axis=1)
    rattler=int(np.argmin(pm))
    low_idx=np.where(deg<=4)[0].tolist()
    high_pool=np.where(deg>=22)[0].tolist()
    if len(high_pool)<12:
        high_pool=np.argsort(-deg,kind="stable")[:12].tolist()
    # ensure rattler not in high pool
    high_pool=[h for h in high_pool if h!=rattler]

    # contact adjacency for greedy max-elimination
    # greedy builder for each k
    def greedy_set(k):
        # want k indices including rattler + (k-1) highs maximizing eliminated contacts
        chosen=[rattler]
        candidates=[h for h in high_pool]
        # also consider low points as tie candidates? keep highs primary
        while len(chosen)<k and candidates:
            best_gain=-1; best_c=None
            cur_set=set(chosen)
            for c in candidates:
                if c in cur_set: continue
                trial=cur_set|{c}
                # eliminated contacts = pairs (i<j) with Adj and (i in trial or j in trial)
                # count quickly
                mask=np.zeros(n,dtype=bool); mask[list(trial)]=True
                # count: sum over i in trial of deg[i] minus internal double count
                # eliminated = sum_{i in trial} deg[i] - internal_edges (counted twice -> subtract once)
                s=sum(int(deg[i]) for i in trial)
                # internal edges
                sub=Adj[np.ix_(list(trial),list(trial))].sum()//2
                elim=s-sub  # each internal counted twice in s, want once => s-sub
                gain=elim
                if gain>best_gain:
                    best_gain=gain; best_c=c
            if best_c is None: break
            chosen.append(best_c); candidates.remove(best_c)
        # if still short (k larger than pool), fill with top-degree remaining
        if len(chosen)<k:
            order=np.argsort(-deg,kind="stable")
            for o in order:
                if int(o) not in chosen:
                    chosen.append(int(o))
                    if len(chosen)>=k: break
        return chosen[:k]

    ks=[1,2,3,4,6,8]
    tasks=[greedy_set(k) for k in ks]
    # add one extra variant for k=6 using next-best highs for diversity if time allows
    # variant: rattler + highs ranked 3..8
    order_hi=np.argsort(-deg,kind="stable")
    order_hi=[int(o) for o in order_hi if int(o)!=rattler]
    tasks.append([rattler]+order_hi[2:8])  # k=7 variant

    results=[]  # (val, x)
    for ti,R in enumerate(tasks):
        if time.time()>DEADLINE-45:
            break
        k=len(R)
        Rset=set(int(r) for r in R)
        keep=[i for i in range(n) if i not in Rset]
        Xcur=gbest[keep].copy()  # use original jammed base for surgery (stable)
        # reinsert one at a time
        for s in range(k):
            if time.time()>DEADLINE-42:
                break
            y0,_=_best_hole(Xcur,rng,n_sample=50000)
            y0=_refine_hole(Xcur,y0,steps=150)
            Xcur=np.vstack([Xcur,y0[None,:]])
            # local: reinserted + neighbours with cos>0.35
            Nnow=Xcur.shape[0]
            new_idx=list(range(Nnow-(s+1),Nnow))
            # neighbours among old points
            if Nnow>len(new_idx):
                sim=Xcur[new_idx] @ Xcur.T  # (s+1) x Nnow
                # max over new rows
                mx=np.max(sim,axis=0)
                neigh=[i for i in range(Nnow) if (i not in new_idx and mx[i]>0.35)]
            else:
                neigh=[]
            active=new_idx+neigh
            # cap active size for speed
            Xcur=_subset_refine(Xcur,active,steps=100)
        # short global Adam/softmax anneal
        Xcur=_global_anneal(Xcur,steps=240)
        v=_max_cos(Xcur)
        results.append((float(v),Xcur.copy()))
        if v<gval:
            gval=float(v); gbest=Xcur.copy()

    # reserve LP for best two surgical candidates
    if results:
        results.sort(key=lambda t:t[0])
        # include current gbest as candidate implicitly; take top2 distinct from results
        top2=results[:2]
        for vi,xi in top2:
            remain=DEADLINE-time.time()-2.0
            if remain<6:
                break
            budget=min(18.0,remain-1.0)
            # only polish if promising (within 0.002 of best or better)
            if vi>gval+0.002 and vi>=base_val:
                continue
            xp,vp=_lp_polish(xi,budget,DEADLINE)
            if vp<gval:
                gval=float(vp); gbest=xp.copy()

    # final safety normalize
    gbest=_norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest=_norm(base)
    return gbest

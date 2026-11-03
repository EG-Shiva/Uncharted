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

def _best_hole(X, rng, n_sample=50000):
    d = X.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    # chunk to limit memory
    best_y = None
    best_m = 1e9
    chunk = 10000
    for s in range(0, n_sample, chunk):
        Yc = Y[s:s+chunk]
        S = Yc @ X.T
        m = S.max(axis=1)
        idx = int(np.argmin(m))
        if float(m[idx]) < best_m:
            best_m = float(m[idx])
            best_y = Yc[idx].copy()
    return best_y, best_m

def _refine_hole(X, y, steps=120, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps-1, 1)
        alpha = np.exp(la0 + (la1-la0)*f)
        lr = np.exp(ll0 + (ll1-ll0)*f)
        s = X @ y
        mx = float(np.max(s))
        w = np.exp(alpha*(s-mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ X
        gt = g - float(np.dot(g, y))*y
        y = y - lr*gt/sw
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _global_anneal(x, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    la0=np.log(a0); la1=np.log(a1); ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(steps):
        f = t/max(steps-1,1)
        alpha=np.exp(la0+(la1-la0)*f)
        lr=np.exp(ll0+(ll1-ll0)*f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx=float(np.max(G))
        W=np.exp(alpha*(G-mx))
        np.fill_diagonal(W, 0.0)
        S=W.sum(axis=1,keepdims=True)+1e-30
        grad=W@x
        dot=np.sum(grad*x,axis=1,keepdims=True)
        gt=grad-dot*x
        direction=gt/S
        x=x-lr*direction
        x=_norm(x)
    return x

def _local_anneal(x, active, steps, a0, a1, lr0, lr1):
    n,d=x.shape
    act=np.asarray(active, dtype=bool)
    if not np.any(act):
        return x
    la0=np.log(a0); la1=np.log(a1); ll0=np.log(lr0); ll1=np.log(lr1)
    for t in range(steps):
        f=t/max(steps-1,1)
        alpha=np.exp(la0+(la1-la0)*f)
        lr=np.exp(ll0+(ll1-ll0)*f)
        G=x@x.T
        np.fill_diagonal(G,-np.inf)
        mx=float(np.max(G))
        W=np.exp(alpha*(G-mx))
        np.fill_diagonal(W,0.0)
        S=W.sum(axis=1,keepdims=True)+1e-30
        grad=W@x
        dot=np.sum(grad*x,axis=1,keepdims=True)
        gt=grad-dot*x
        direction=gt/S
        x[act]=x[act]-lr*direction[act]
        x[act]/=(np.linalg.norm(x[act],axis=1,keepdims=True)+1e-300)
    return x

def _lp_polish(x, seconds, t_deadline):
    x=_norm(x.copy())
    n,d=x.shape
    try:
        best=_max_cos(x)
    except Exception:
        return x
    xb=x.copy()
    radius=1e-3
    start=time.time()
    iu0=np.triu_indices(n,1)
    ar=np.arange(d)
    pcap=1500
    while time.time()-start<seconds and time.time()<t_deadline:
        if radius<1e-13:
            break
        if time.time()>t_deadline-1.0:
            break
        G=x@x.T
        cf=G[iu0]
        m=float(cf.max())
        thr=m-max(20.0*radius,1e-9)
        sel=cf>thr
        I=iu0[0][sel]; J=iu0[1][sel]; cs=cf[sel]
        p=len(I)
        if p==0:
            radius*=0.5
            continue
        if p>pcap:
            idx=np.argpartition(-cs,pcap)[:pcap]
            I,J,cs=I[idx],J[idx],cs[idx]
            p=pcap
        nv=n*d+1
        rows=np.concatenate([np.repeat(np.arange(p),d),np.repeat(np.arange(p),d),np.arange(p)])
        cols=np.concatenate([np.repeat(I*d,d)+np.tile(ar,p),np.repeat(J*d,d)+np.tile(ar,p),np.full(p,n*d)])
        vals=np.concatenate([x[J].ravel(),x[I].ravel(),-np.ones(p)])
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-cs,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs",
                        options={"time_limit":2.0})
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y=_norm(x+res.x[:-1].reshape(n,d))
        try:
            v=_max_cos(y)
        except Exception:
            radius*=0.5
            continue
        if v<best:
            best=v; xb=y.copy(); x=y
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return xb

def solve():
    t0=time.time()
    HARD=t0+168.0
    rng=np.random.default_rng(12345)
    n,d=184,8
    try:
        xb=np.load("best.npy")
    except Exception:
        xb=np.load("record.npy")
    xb=np.asarray(xb,dtype=np.float64).reshape(n,d)
    xb=_norm(xb)
    gval=_max_cos(xb)
    gbest=xb.copy()
    # contact analysis
    G=xb@xb.T
    np.fill_diagonal(G,-2.0)
    m=float(np.max(G))
    contact_thr=m-1e-9
    mask=G>contact_thr
    # degrees (symmetric, count per row, diagonal excluded)
    deg=np.sum(mask,axis=1).astype(int)
    # rattler: minimal degree
    rattler=int(np.argmin(deg))
    order=np.argsort(-deg, kind="stable")
    high_pool=[int(i) for i in order if int(deg[int(i)])>=22]
    if len(high_pool)<10:
        # extend with top degrees
        high_pool=[int(i) for i in order[:max(10,len(high_pool))]]
    # contact pair list for greedy coverage
    iu=np.triu_indices(n,1)
    pair_I=iu[0][mask[iu]]
    pair_J=iu[1][mask[iu]]
    P=len(pair_I)
    # map point -> incident pair indices (for greedy)
    # build incidence via lists
    ks=[1,2,3,4,6,8]
    cands=[]
    for k in ks:
        if time.time()>HARD-35:
            break
        # greedy selection of k points: rattler + (k-1) hubs maximizing new covered contacts
        selected=[rattler]
        sel_set=set(selected)
        # candidate pool: high_pool excluding rattler
        pool=[h for h in high_pool if h not in sel_set]
        # add fallback top-degree if pool short
        for o in order:
            o=int(o)
            if len(pool)>=max(0,k-1)+4:
                break
            if o not in sel_set and o not in pool:
                pool.append(o)
        covered=np.zeros(P,dtype=bool) if P>0 else np.zeros(0,dtype=bool)
        # precompute incident mask per point? on fly
        for _ in range(k-1):
            best_h=None; best_gain=-1
            for h in pool:
                if h in sel_set:
                    continue
                if P>0:
                    inc=(pair_I==h)|(pair_J==h)
                    gain=int(np.sum(inc & (~covered)))
                else:
                    gain=0
                # tie-break by degree
                gain2=gain+1e-6*int(deg[h])
                if gain2>best_gain:
                    best_gain=gain2; best_h=h
            if best_h is None:
                break
            selected.append(best_h); sel_set.add(best_h)
            if P>0:
                inc=(pair_I==best_h)|(pair_J==best_h)
                covered[inc]=True
        # build keep matrix
        keep_mask=np.ones(n,dtype=bool); keep_mask[selected]=False
        Xcur=xb[keep_mask].copy()
        inserted=[]  # track indices in current array (appended at end)
        # reinsert one at a time
        ok=True
        for t in range(k):
            if time.time()>HARD-35:
                ok=False; break
            y0,_=_best_hole(Xcur,rng,n_sample=50000)
            y=_refine_hole(Xcur,y0,steps=120)
            Xcur=np.vstack([Xcur,y[None,:]])
            Xcur=_norm(Xcur)
            # active set: all inserted + neighbours with cos>0.35
            cur_n=Xcur.shape[0]
            ins_idx=list(range(cur_n-(t+1),cur_n))
            Gn=Xcur@Xcur.T
            act=np.zeros(cur_n,dtype=bool)
            act[ins_idx]=True
            for j in ins_idx:
                act=np.logical_or(act, Gn[:,j]>0.35)
                act=np.logical_or(act, Gn[j,:]>0.35)
            act[ins_idx]=True
            Xcur=_local_anneal(Xcur,act,steps=90,a0=30.0,a1=250.0,lr0=0.02,lr1=0.005)
            Xcur=_global_anneal(Xcur,steps=110,a0=30.0,a1=250.0,lr0=0.018,lr1=0.004)
        if not ok:
            break
        # final short global tighten
        Xcur=_global_anneal(Xcur,steps=80,a0=80.0,a1=350.0,lr0=0.008,lr1=0.002)
        try:
            v=_max_cos(Xcur)
        except Exception:
            continue
        cands.append((v,Xcur.copy()))
        if v<gval:
            gval=v; gbest=Xcur.copy()
    # reserve LP for best two surgical candidates
    if cands:
        cands.sort(key=lambda t:t[0])
        for idx in range(min(2,len(cands))):
            if time.time()>HARD-15:
                break
            v,Xc=cands[idx]
            # only polish if within 0.01 of global best (avoid wasting on bad)
            if v>gval+0.01:
                continue
            secs=min(10.0, HARD-2.0-time.time())
            if secs<2.0:
                break
            Xp=_lp_polish(Xc,seconds=secs,t_deadline=HARD-1.0)
            try:
                vp=_max_cos(Xp)
            except Exception:
                continue
            if vp<gval:
                gval=vp; gbest=_norm(Xp)
            # also update cands entry
            cands[idx]=(vp,Xp)
    gbest=_norm(gbest)
    return gbest

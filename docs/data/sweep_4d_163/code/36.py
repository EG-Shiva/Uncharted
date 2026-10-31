import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_step(x, radius, cap=2200):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    thr = m - max(20.0*radius, 1e-9)
    sel = c_all > thr
    if int(np.count_nonzero(sel)) == 0:
        return x, m, None
    if int(np.count_nonzero(sel)) > cap:
        v = c_all[sel]
        kth = np.sort(v)[-cap]
        sel = c_all >= kth
        if int(np.count_nonzero(sel)) > cap+800:
            ids = np.where(sel)[0][::2][:cap]
            ns = np.zeros_like(sel); ns[ids]=True; sel=ns
    I = iu[0][sel]; J = iu[1][sel]; c = c_all[sel]
    p = len(I); nv = n*d+1
    rows = np.repeat(np.arange(p), 2*d+1)
    cd = np.arange(d)
    cols = np.concatenate([I[:,None]*d+cd, J[:,None]*d+cd, np.full((p,1),n*d)],1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
    A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
    A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
    obj = np.zeros(nv); obj[-1]=1.0
    try:
        res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                      bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
    except Exception:
        return x, m, None
    if res.status!=0:
        return x, m, res
    y = x + res.x[:-1].reshape(n,d)
    y /= np.linalg.norm(y,axis=1,keepdims=True)
    return y, _max_cos(y), res

def _lp_polish(x, seconds, r0=2e-3):
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    best=_max_cos(x); radius=r0; start=time.time()
    iu = np.triu_indices(x.shape[0],1)
    n,d=x.shape
    while time.time()-start<seconds and radius>1e-14:
        G=x@x.T; c_all=G[iu]; m=float(c_all.max())
        thr=m-max(20.0*radius,1e-9)
        sel=c_all>thr
        cnt=int(np.count_nonzero(sel))
        if cnt==0: break
        if cnt>2200:
            kth=np.sort(c_all[sel])[-2200]
            sel=c_all>=kth
            if int(np.count_nonzero(sel))>3000:
                ids=np.where(sel)[0][::2][:3000]
                ns=np.zeros_like(sel); ns[ids]=True; sel=ns
        I=iu[0][sel]; J=iu[1][sel]; c=c_all[sel]; p=len(I); nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1); cd=np.arange(d)
        cols=np.concatenate([I[:,None]*d+cd,J[:,None]*d+cd,np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best:
            x,best=y,v; radius=min(radius*1.6,0.05)
        else:
            radius*=0.5
    return x

def _smooth(x, iters, T0, T1, lr0, lr1):
    for k in range(iters):
        frac=k/max(iters-1,1)
        T=T0*(T1/T0)**frac; lr=lr0*(lr1/lr0)**frac
        G=x@x.T; np.fill_diagonal(G,-2.0); m=float(G.max())
        W=np.exp((G-m)/T); np.fill_diagonal(W,0.0)
        grad=W@x
        gt=grad-np.sum(grad*x,axis=1,keepdims=True)*x
        rs=W.sum(axis=1,keepdims=True)
        step=gt/np.maximum(rs,1e-12)
        sn=np.linalg.norm(step,axis=1,keepdims=True)
        step=step/np.maximum(sn,1.0)
        x=x-lr*step
        x/=np.linalg.norm(x,axis=1,keepdims=True)
    return x

def _forced_surgery(x0, force_pairs, Bfinal, nsteps=8, radius=0.015, cap_top=1500):
    # force_pairs: set of (i,j) tuples to force <= Bfinal
    n,d=x0.shape
    x=x0.copy()
    iu=np.triu_indices(n,1)
    # map pair to flat index for fast union: build dict flat->pos
    # instead keep set of (i,j) with i<j
    fset=set()
    for a,b in force_pairs:
        if a>b: a,b=b,a
        fset.add((int(a),int(b)))
    # precompute flat index lookup: flat id = ?
    # use dictionary from (i,j) to flat position in iu order
    # build once
    P=n*(n-1)//2
    fflat=set()
    # build map via loop over force pairs using search? faster: use hash of iu
    # create 2D index array pos matrix (n x n) with flat id
    # n=163 small
    posmat=np.full((n,n),-1,dtype=int)
    posmat[iu]=np.arange(P)
    for (a,b) in fset:
        fflat.add(int(posmat[a,b]))
    m0=_max_cos(x0)
    for step in range(nsteps):
        # ramped target
        Bk=m0+(Bfinal-m0)*(step+1)/nsteps
        G=x@x.T
        c_all=G[iu]
        order=np.argpartition(-c_all, min(cap_top,P-1))[:cap_top]
        # union with forced (ensure forced included)
        selmask=np.zeros(P,dtype=bool)
        selmask[order]=True
        for ff in fflat:
            selmask[ff]=True
        sidx=np.where(selmask)[0]
        I=iu[0][sidx]; J=iu[1][sidx]; c=c_all[sidx]
        is_forced=np.array([int(posmat[a,b]) in fflat for a,b in zip(I,J)],dtype=bool)
        # split
        p=len(sidx); nv=n*d+1
        # A rows: free rows have -1 in t col, forced rows 0
        # build efficiently with coo
        rows_list=[]; cols_list=[]; vals_list=[]; bub=[]
        # free
        # vectorized build
        # for forced rows cols: I*d.., J*d..
        # for free rows plus t col
        nfree=int(np.count_nonzero(~is_forced)); nforc=int(np.count_nonzero(is_forced))
        # build full matrix: each row has 2d (+1 if free)
        # construct via repeat
        rows=np.repeat(np.arange(p),2*d+1)
        cd=np.arange(d)
        base_cols=np.concatenate([I[:,None]*d+cd, J[:,None]*d+cd, np.full((p,1),n*d)],1)
        base_vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1)
        # zero t-coeff for forced rows
        base_vals[is_forced,-1]=0.0
        cols=base_cols.ravel(); vals=base_vals.ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        b_ub=np.empty(p)
        b_ub[~is_forced]=-c[~is_forced]
        b_ub[is_forced]=Bk-c[is_forced]
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            # infeasible: relax by increasing radius or raising Bk
            # try larger radius once
            try:
                res2=linprog(obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=np.zeros(n),
                            bounds=[(-radius*2,radius*2)]*(n*d)+[(None,None)],method="highs")
                if res2.status!=0:
                    radius*=0.5; continue
                res=res2
                radius=min(radius*1.2,0.05)
            except Exception:
                radius*=0.5; continue
        dx=res.x[:-1].reshape(n,d)
        x=x+dx
        x/=np.linalg.norm(x,axis=1,keepdims=True)
    return x

def _contact_mask(x, tol=1e-5):
    G=x@x.T; np.fill_diagonal(G,-2.0); m=float(G.max())
    return (G>=m-tol), m

def solve():
    rng=np.random.default_rng(0)
    t0=time.time()
    try: xb=np.load("best.npy")
    except Exception: xb=None
    try: xr=np.load("record.npy")
    except Exception: xr=None
    cands=[]
    if xb is not None: cands.append(np.asarray(xb,float))
    if xr is not None: cands.append(np.asarray(xr,float))
    best=None; best_val=1e9
    for c in cands:
        if c.shape!=(163,4): continue
        c=c/np.linalg.norm(c,axis=1,keepdims=True)
        v=_max_cos(c)
        if v<best_val: best_val=v; best=c.copy()
    if best is None:
        best=rng.normal(size=(163,4)); best/=np.linalg.norm(best,axis=1,keepdims=True); best_val=_max_cos(best)
    n,d=163,4
    # --- analyse record: degrees, duals ---
    G0=best@best.T; np.fill_diagonal(G0,-2.0); m0=float(G0.max())
    iu0=np.triu_indices(n,1); c0=G0[iu0]
    order0=np.argsort(-c0)
    # degrees at tight threshold
    tight=G0>=m0-1e-9
    deg=tight.sum(axis=1)
    # try to get duals from one LP step
    dual_score=None
    try:
        _,_,res0=_lp_step(best.copy(),1e-3,cap=1200)
        if res0 is not None and hasattr(res0,'ineqlin') and res0.ineqlin is not None:
            marg=None
            try: marg=np.asarray(res0.ineqlin.marginals,float).ravel()
            except Exception:
                try: marg=np.asarray(res0.ineqlin['marginals'],float).ravel()
                except Exception: marg=None
            if marg is not None and marg.size>0:
                # map to full pair ordering used in _lp_step (top cap) - approximate: use abs
                # we just rank the selected constraints; pad to full by zeros
                # reconstruct selection
                Gtmp=best@best.T; call=Gtmp[iu0]; mm=float(call.max())
                sel=call>mm-max(20*1e-3,1e-9)
                # take same cap logic roughly: if too many keep top 1200? approximate
                dual_score=np.zeros_like(call)
                # marg corresponds to sel (maybe truncated); if sizes match use it
                idx=np.where(sel)[0]
                if marg.size==idx.size:
                    dual_score[idx]=np.abs(marg)
    except Exception:
        dual_score=None
    # build clusters of 20-40 pairs
    P=n*(n-1)//2
    Iall, Jall = iu0[0], iu0[1]
    top100=[(int(Iall[order0[k]]),int(Jall[order0[k]])) for k in range(100)]
    top500=[(int(Iall[order0[k]]),int(Jall[order0[k]])) for k in range(min(500,P))]
    clusters=[]
    # C0: top-30
    clusters.append([(int(Iall[order0[k]]),int(Jall[order0[k]])) for k in range(30)])
    # C1: high-dual top 30 if available else top 30 shifted
    if dual_score is not None and np.any(dual_score>0):
        di=np.argsort(-dual_score)[:30]
        clusters.append([(int(Iall[k]),int(Jall[k])) for k in di])
    else:
        clusters.append([(int(Iall[order0[k]]),int(Jall[order0[k]])) for k in range(20,50)])
    # C2,C3: high-degree points
    hid=np.argsort(-deg)[:6]
    for h in hid[:3]:
        lst=[]
        for k in range(min(400,P)):
            a=int(Iall[order0[k]]); b=int(Jall[order0[k]])
            if a==h or b==h:
                lst.append((a,b))
            if len(lst)>=30: break
        if len(lst)>=15: clusters.append(lst)
    # combined top-2 points
    lst=[]
    s=set(hid[:2].tolist())
    for k in range(min(600,P)):
        a=int(Iall[order0[k]]); b=int(Jall[order0[k]])
        if a in s or b in s:
            lst.append((a,b))
        if len(lst)>=36: break
    if len(lst)>=15: clusters.append(lst[:30])
    # localized patches: centres = high-degree points, cap dot>0.5
    for h in hid[:3]:
        ch=best[h]
        dots=best@ch
        inside=np.where(dots>0.5)[0]
        s2=set(inside.tolist())
        lst2=[]
        for (a,b) in top500:
            if a in s2 and b in s2:
                lst2.append((a,b))
            if len(lst2)>=30: break
        if len(lst2)>=15:
            # pad to 20-30 with top edges incident to patch
            clusters.append(lst2)
    # random subsets of top100 for diversity
    for s_ in range(3):
        idx=rng.choice(100, size=28, replace=False)
        clusters.append([top100[k] for k in idx])
    # deltas
    deltas=[5e-4,8e-4,1.2e-3,2e-3,6e-4,1.5e-3]
    # reference contact mask for diversity
    refmask,_=_contact_mask(best,1e-5)
    pool=[best.copy()]; pool_vals=[best_val]
    trial=0; ci=0
    # surgery loop, reserve final polish
    while time.time()-t0<128 and trial<28:
        cl=clusters[ci%len(clusters)]
        delta=deltas[(trial//max(1,len(clusters)))%len(deltas)] if trial>=len(clusters) else deltas[trial%len(deltas)]
        # alternate start among pool (round robin) to explore basins
        start_idx=trial%len(pool) if len(pool)>1 else 0
        xstart=pool[start_idx].copy()
        # tiny shake on forced points to break symmetry
        fpts=np.unique(np.array(cl).ravel())
        xstart[fpts]+=rng.normal(scale=0.002,size=(len(fpts),d))
        xstart/=np.linalg.norm(xstart,axis=1,keepdims=True)
        B=m0-delta
        # forced phase
        try:
            y=_forced_surgery(xstart,cl,B,nsteps=7,radius=0.015)
        except Exception:
            trial+=1; ci+=1; continue
        # release: smooth + short LP
        y=_smooth(y,iters=280,T0=0.015,T1=0.002,lr0=0.008,lr1=0.001)
        remain=128-(time.time()-t0)
        sec=min(2.2,max(0.7,remain/10))
        y=_lp_polish(y,seconds=sec,r0=2e-3)
        v=_max_cos(y)
        # diversity: jaccard at 1e-5
        msk,_=_contact_mask(y,1e-5)
        inter=np.count_nonzero(msk&refmask); union=np.count_nonzero(msk|refmask)
        jac=inter/max(union,1)
        # update best
        if v<best_val-1e-12:
            best_val=v; best=y.copy(); m0=min(m0,v)
            pool.append(y.copy()); pool_vals.append(v)
            refmask=msk.copy() if v<pool_vals[0] else refmask
            # keep pool small
            if len(pool)>6:
                # drop worst
                j=int(np.argmax(pool_vals)); pool.pop(j); pool_vals.pop(j)
        elif jac<0.75 and v<best_val+0.004 and len(pool)<6:
            # diverse topology, keep as alternate basin
            pool.append(y.copy()); pool_vals.append(v)
        trial+=1; ci+=1
        if time.time()-t0>126: break
    # final polish best with remaining budget
    elapsed=time.time()-t0
    budget=max(5.0,min(28.0,168.0-elapsed))
    best=_lp_polish(best,seconds=budget*0.7,r0=2e-3)
    best=_smooth(best,iters=200,T0=0.004,T1=0.0008,lr0=0.003,lr1=0.0005)
    best=_lp_polish(best,seconds=max(3.0,budget*0.3),r0=1e-3)
    best=np.nan_to_num(best)
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best

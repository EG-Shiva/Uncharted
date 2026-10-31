import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, r0=2e-3, cap=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            if int(np.count_nonzero(sel)) > cap + 800:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None]*d+cd, J[:, None]*d+cd, np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = _max_cos(y)
        if v < best:
            x,best=y,v
            radius=min(radius*1.6,0.05)
        else:
            radius*=0.5
    return x

def _smooth(x, iters, T0, T1, lr0, lr1):
    n,d=x.shape
    for k in range(iters):
        frac=k/max(iters-1,1)
        T=T0*(T1/T0)**frac
        lr=lr0*(lr1/lr0)**frac
        G=x@x.T
        np.fill_diagonal(G,-2.0)
        m=float(G.max())
        W=np.exp((G-m)/T)
        np.fill_diagonal(W,0.0)
        grad=W@x
        dot=np.sum(grad*x,axis=1,keepdims=True)
        gt=grad-dot*x
        rs=W.sum(axis=1,keepdims=True)
        step=gt/np.maximum(rs,1e-12)
        sn=np.linalg.norm(step,axis=1,keepdims=True)
        step=step/np.maximum(sn,1.0)
        x=x-lr*step
        x/=np.linalg.norm(x,axis=1,keepdims=True)
    return x

def _dual_info(x, radius=1e-3, cap=2500):
    n,d=x.shape
    iu=np.triu_indices(n,1)
    G=x@x.T
    c_all=G[iu]
    m=float(c_all.max())
    thr=m-max(20.0*radius,1e-9)
    sel=c_all>thr
    if int(np.count_nonzero(sel))>cap:
        v=c_all[sel]
        kth=np.sort(v)[-cap]
        sel=c_all>=kth
    I=iu[0][sel]; J=iu[1][sel]
    c=c_all[sel]
    p=len(I)
    nv=n*d+1
    rows=np.repeat(np.arange(p),2*d+1)
    cd=np.arange(d)
    cols=np.concatenate([I[:,None]*d+cd,J[:,None]*d+cd,np.full((p,1),n*d)],1).ravel()
    vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
    A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
    A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
    obj=np.zeros(nv); obj[-1]=1.0
    try:
        res=linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                    bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
    except Exception:
        return None
    if res.status!=0:
        return None
    w=None
    try:
        if hasattr(res,'ineqlin') and res.ineqlin is not None:
            mg=np.asarray(res.ineqlin.marginals,dtype=float).ravel()
            if mg.size==p:
                w=-mg
    except Exception:
        w=None
    if w is None or np.any(~np.isfinite(w)) or np.max(np.abs(w))<=0:
        w=(c-m+max(20*radius,1e-9))
        w=np.maximum(w,0)
    w=np.maximum(w,0)
    if w.sum()>0:
        w=w/w.sum()
    return {"I":I,"J":J,"c":c,"w":w,"m":m,"iu":iu,"c_all":c_all}

def _edge_set(c_all, m, tol=1e-5):
    iu_n=len(c_all)
    # return set of indices where c>m-tol
    return set(np.where(c_all>m-tol)[0].tolist())

def _forced_phase(x, FI, FJ, delta, m_ref, radius=0.015, iters=5, cap=2000):
    n,d=x.shape
    iu=np.triu_indices(n,1)
    # explicit push first (fast feasible start)
    for _ in range(40):
        cf=np.sum(x[FI]*x[FJ],axis=1)
        # only push those still above m_ref-delta
        need=cf>(m_ref-delta)
        if not np.any(need):
            break
        step=np.zeros_like(x)
        for a,b in zip(FI[need],FJ[need]):
            dab=float(x[a]@x[b])
            pa=x[a]-dab*x[b]
            # project to tangent at a
            pa=pa-(pa@x[a])*x[a]
            pb=x[b]-dab*x[a]
            pb=pb-(pb@x[b])*x[b]
            step[a]+=pa; step[b]+=pb
        dot=np.sum(step*x,axis=1,keepdims=True)
        step=step-dot*x
        nrm=np.linalg.norm(step,axis=1,keepdims=True)
        step=step/np.maximum(nrm,1e-12)
        x=x-0.004*step
        x/=np.linalg.norm(x,axis=1,keepdims=True)
    # capped LP iterations
    for _ in range(iters):
        G=x@x.T
        c_all=G[iu]
        mcur=float(c_all.max())
        thr=mcur-max(20.0*radius,2e-4)
        sel=c_all>thr
        # exclude forced pairs from free set (identify by pair index)
        # build map: pair (min,max) -> idx; simpler: mask out exact FI,FJ
        # create forbidden mask
        # for speed, use set lookup via flat index
        flat=iu[0]*n+iu[1]
        fflat=FI*n+FJ
        # FI<FJ assumed
        is_f=np.isin(flat,fflat)
        sel_free=sel & (~is_f)
        # cap free
        if int(np.count_nonzero(sel_free))>cap:
            v=c_all[sel_free]
            kth=np.sort(v)[-cap]
            # need global mask
            sel_free=c_all>=kth
            sel_free=sel_free & (~is_f)
        If=FI; Jf=FJ
        cf=np.sum(x[If]*x[Jf],axis=1)
        Io=iu[0][sel_free]; Jo=iu[1][sel_free]
        co=c_all[sel_free]
        pf=len(If); po=len(Io)
        p=pf+po
        if p==0:
            break
        nv=n*d+1
        # rows for forced: 2d entries, no t; for free: 2d+1
        # build dense-ish coo via lists
        rows=[]; cols=[]; vals=[]
        b=np.empty(p)
        # forced
        cd=np.arange(d)
        for k in range(pf):
            r=k; a=int(If[k]); b2=int(Jf[k])
            for t_ in range(d):
                rows.append(r); cols.append(a*d+t_); vals.append(float(x[b2,t_]))
                rows.append(r); cols.append(b2*d+t_); vals.append(float(x[a,t_]))
            b[k]=(m_ref-delta)-float(cf[k])
        for k in range(po):
            r=pf+k; a=int(Io[k]); b2=int(Jo[k])
            for t_ in range(d):
                rows.append(r); cols.append(a*d+t_); vals.append(float(x[b2,t_]))
                rows.append(r); cols.append(b2*d+t_); vals.append(float(x[a,t_]))
            rows.append(r); cols.append(n*d); vals.append(-1.0)
            b[r]=-float(co[k])
        try:
            A_ub=coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=b,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            break
        if res.status!=0:
            # try smaller radius once
            radius*=0.5
            if radius<1e-4:
                break
            continue
        dx=res.x[:-1].reshape(n,d)
        y=x+dx
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        x=y
    return x

def solve():
    rng=np.random.default_rng(7)
    t0=time.time()
    BUDGET=168.0
    try:
        xb=np.load("best.npy")
    except Exception:
        xb=None
    try:
        xr=np.load("record.npy")
    except Exception:
        xr=None
    cands=[]
    if xb is not None:
        try: cands.append(np.asarray(xb,dtype=float))
        except Exception: pass
    if xr is not None:
        try: cands.append(np.asarray(xr,dtype=float))
        except Exception: pass
    if len(cands)==0:
        z=rng.normal(size=(163,4)); z/=np.linalg.norm(z,axis=1,keepdims=True)
        cands.append(z)
    best=None; best_val=1e9
    for c in cands:
        if c.shape!=(163,4): continue
        c=c/np.linalg.norm(c,axis=1,keepdims=True)
        v=_max_cos(c)
        if v<best_val: best_val=v; best=c.copy()
    n,d=163,4
    # initial polish to true local opt
    best=_lp_polish(best,seconds=3.0)
    best_val=_max_cos(best)
    # baseline edge set at 1e-5
    G0=best@best.T
    iu0=np.triu_indices(n,1)
    c0=G0[iu0]
    E0=_edge_set(c0,best_val,tol=1e-5)

    # dual info at optimum
    di=_dual_info(best,radius=1e-3)
    if di is None:
        # fallback: use closeness as weight
        I=iu0[0][c0>best_val-2e-4]; J=iu0[1][c0>best_val-2e-4]
        w=c0[c0>best_val-2e-4]
        w=w-w.min()+1e-12; w=w/w.sum()
        di={"I":I,"J":J,"c":np.sum(best[I]*best[J],axis=1),"w":w,"m":best_val,"iu":iu0,"c_all":c0}
    Iact,Jact,wact,mref=di["I"],di["J"],di["w"],di["m"]
    # order by dual
    order=np.argsort(-wact)
    # point stress
    pstress=np.zeros(n)
    for k in range(len(Iact)):
        pstress[int(Iact[k])]+=wact[k]
        pstress[int(Jact[k])]+=wact[k]
    hub_order=np.argsort(-pstress)
    # degree (count) for patch
    # contact graph adjacency at 1e-4
    thrA=mref-2e-4
    # build neighbor lists
    # jobs: list of (FI,FJ,delta)
    jobs=[]
    def take_top(k):
        idx=order[:k]
        return Iact[idx].copy(),Jact[idx].copy()
    # A clusters
    a,b=take_top(25); jobs.append((a,b,1e-3))
    a,b=take_top(35); jobs.append((a,b,2e-3))
    a,b=take_top(30); jobs.append((a,b,5e-4))
    a,b=take_top(20); jobs.append((a,b,1.5e-3))
    # B: hub incident
    for nh,delta in [(2,1e-3),(3,1.5e-3)]:
        hubs=hub_order[:nh]
        hset=set(hubs.tolist())
        mask=np.array([(int(Iact[k]) in hset) or (int(Jact[k]) in hset) for k in range(len(Iact))],dtype=bool)
        idx=np.where(mask)[0]
        # sort those by w
        idx=idx[np.argsort(-wact[idx])]
        idx=idx[:35]
        if len(idx)>=15:
            jobs.append((Iact[idx].copy(),Jact[idx].copy(),delta))
    # C: localized patches via euclidean nearest to hub
    for hi,delta in [(0,1e-3),(1,1.5e-3)]:
        h=int(hub_order[hi])
        dots=best@best[h]
        patch=np.argsort(-dots)[:16]
        pset=set(patch.tolist())
        mask=np.array([(int(Iact[k]) in pset) and (int(Jact[k]) in pset) for k in range(len(Iact))],dtype=bool)
        idx=np.where(mask)[0]
        if len(idx)<15:
            # expand to incident
            mask2=np.array([(int(Iact[k]) in pset) or (int(Jact[k]) in pset) for k in range(len(Iact))],dtype=bool)
            idx=np.where(mask2)[0]
            idx=idx[np.argsort(-wact[idx])][:30]
        else:
            # if too many, trim to top dual
            idx=idx[np.argsort(-wact[idx])][:30]
        if len(idx)>=15:
            jobs.append((Iact[idx].copy(),Jact[idx].copy(),delta))
    # D: random diverse
    for delta in [1e-3,2e-3]:
        idx=rng.choice(len(Iact),size=min(30,len(Iact)),replace=False)
        jobs.append((Iact[idx].copy(),Jact[idx].copy(),delta))

    # ensure FI<FJ ordering for isin logic (they already are since iu)
    pool=[]  # (val, x, overlap)
    for ji,(FI,FJ,delta) in enumerate(jobs):
        if time.time()-t0>BUDGET-25:
            break
        # enforce size 20-40
        if len(FI)>40:
            # keep highest dual among them: need w lookup; approximate by closeness
            cf=np.sum(best[FI]*best[FJ],axis=1)
            korder=np.argsort(-cf)[:40]
            FI, FJ = FI[korder], FJ[korder]
        # ensure FI<FJ for flat matching
        lo=np.minimum(FI,FJ); hi=np.maximum(FI,FJ)
        FI, FJ = lo, hi
        y=best.copy()
        y=_forced_phase(y,FI,FJ,delta,mref,radius=0.015,iters=4)
        # release: smooth + short lp
        y=_smooth(y,iters=120,T0=0.008,T1=0.0008,lr0=0.008,lr1=0.002)
        y=_lp_polish(y,seconds=1.2,r0=1.5e-3)
        v=_max_cos(y)
        Gy=y@y.T; cy=Gy[iu0]; Ey=_edge_set(cy,v,tol=1e-5)
        inter=len(Ey & E0); union=len(Ey | E0)
        ov=inter/max(union,1)
        pool.append((v,y.copy(),ov))
        if v<best_val-1e-12:
            # big topology check: accept even if slightly worse? only update best if better
            best=y.copy(); best_val=v
            # refresh reference? keep original mref for forcing consistency, but update E0
            E0=Ey
            # recompute duals for remaining jobs? keep old to save time, but update mref
            mref=best_val
        elif ov<0.88:
            # diverse but not better: keep for second round if promising (within 3e-3)
            pass
        # Metropolis-like: if close, maybe keep as base for next? no, always from best
    # second round on most diverse promising candidates
    pool_sorted=sorted(pool,key=lambda t:t[0])
    # pick up to 2 diverse with ov<0.85 and value within 5e-3 of best
    second=[]
    for v,y,ov in pool_sorted:
        if ov<0.85 and v<best_val+5e-3 and len(second)<2:
            second.append((v,y))
    for v0,y0 in second:
        if time.time()-t0>BUDGET-18:
            break
        # new dual at y0
        di2=_dual_info(y0,radius=1e-3)
        if di2 is None:
            continue
        I2,J2,w2=di2["I"],di2["J"],di2["w"]
        o2=np.argsort(-w2)
        for delta in [1e-3]:
            if time.time()-t0>BUDGET-14:
                break
            k=min(28,len(o2))
            FI=I2[o2[:k]]; FJ=J2[o2[:k]]
            lo=np.minimum(FI,FJ); hi=np.maximum(FI,FJ)
            FI,FJ=lo,hi
            y=y0.copy()
            y=_forced_phase(y,FI,FJ,delta,mref=v0,radius=0.015,iters=4)
            y=_smooth(y,iters=100,T0=0.006,T1=0.0008,lr0=0.007,lr1=0.002)
            y=_lp_polish(y,seconds=1.0,r0=1.5e-3)
            v=_max_cos(y)
            if v<best_val:
                best=y.copy(); best_val=v
    # final thorough polish of best with remaining time
    remain=BUDGET-(time.time()-t0)
    if remain>4:
        # alternate smooth + lp
        best=_smooth(best,iters=150,T0=0.004,T1=0.0005,lr0=0.005,lr1=0.001)
        best=_lp_polish(best,seconds=max(2.0,remain-2.0))
        best_val=_max_cos(best)
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    return best

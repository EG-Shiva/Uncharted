import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_IU = None

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _iu(n):
    global _IU
    if _IU is None or _IU[0].shape[0] != n*(n-1)//2:
        _IU = np.triu_indices(n, 1)
    return _IU

def _lp_polish(x, seconds, t_deadline=None, r0=2e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu = _iu(n)
    best = _max_cos(x)
    radius = r0
    start = time.time()
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14 or it >= 150:
            break
        it += 1
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0*radius, 1e-9)
        sel = c_all > thr
        if int(np.count_nonzero(sel)) == 0:
            break
        if int(np.count_nonzero(sel)) > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            if int(np.count_nonzero(sel)) > cap+800:
                ids = np.where(sel)[0][::2][:cap]
                ns = np.zeros_like(sel)
                ns[ids] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n*d+1
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
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.6, 0.05)
        else:
            radius*=0.5
    return x

def _stress_from_duals(x, radius=1e-3):
    n,d = x.shape
    iu = _iu(n)
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    thr = m - max(20.0*radius, 2e-9)
    sel = c_all > thr
    I = iu[0][sel]; J = iu[1][sel]
    c = c_all[sel]
    p = len(I)
    nv = n*d+1
    rows = np.repeat(np.arange(p),2*d+1)
    cd = np.arange(d)
    cols = np.concatenate([I[:,None]*d+cd,J[:,None]*d+cd,np.full((p,1),n*d)],1).ravel()
    vals = np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
    A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
    A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
    obj = np.zeros(nv); obj[-1]=1.0
    try:
        res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                      bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
    except Exception:
        return None
    if res.status!=0:
        return None
    w = None
    try:
        mg = res.ineqlin.marginals
        mg = np.asarray(mg).ravel()
        if mg.shape[0]==p:
            w = -mg
    except Exception:
        try:
            mg = res.ineqlin["marginals"]
            mg = np.asarray(mg).ravel()
            if mg.shape[0]==p:
                w = -mg
        except Exception:
            w = None
    if w is None:
        return None
    w = np.maximum(w,0.0)
    if w.max()<=0:
        return None
    return (I,J,c,m,w)

def _contact_info(x, tight=1e-9):
    n = x.shape[0]
    iu = _iu(n)
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    sel = c_all >= m-tight
    deg = np.zeros(n)
    np.add.at(deg, iu[0][sel], 1)
    np.add.at(deg, iu[1][sel], 1)
    return m, deg, sel, c_all

def _edge_mask(x, thresh=1e-5):
    n = x.shape[0]
    iu = _iu(n)
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    return c_all >= m-thresh

def _forced_steps(x, IF, JF, m0, delta, steps=4, radius=0.018):
    n,d = x.shape
    iu = _iu(n)
    nv = n*d+1
    cd = np.arange(d)
    for _ in range(steps):
        G = x @ x.T
        c_all = G[iu]
        m_cur = float(c_all.max())
        thr = m_cur - max(20.0*radius, 1e-9)
        sel = c_all > thr
        # map forced out of normal set
        # build normal I,J excluding forced pairs (check via set for small F)
        fset = set(zip(IF.tolist(),JF.tolist()))
        Iall = iu[0][sel]; Jall = iu[1][sel]
        keep = np.array([ (a,b) not in fset for a,b in zip(Iall,Jall)], dtype=bool)
        IN = Iall[keep]; JN = Jall[keep]
        # cap normal
        if len(IN)>2200:
            cn = G[IN,JN]
            kth = np.sort(cn)[-2200]
            m2 = cn>=kth
            IN = IN[m2]; JN = JN[m2]
        cF = G[IF,JF]
        cN = G[IN,JN] if len(IN)>0 else np.zeros(0)
        pF = len(IF); pN = len(IN); p = pF+pN
        if p==0:
            break
        # rows: F rows have 0*t coef, N rows have -1
        rowsF = np.repeat(np.arange(pF),2*d) if pF>0 else np.zeros(0,dtype=int)
        rowsN = np.repeat(np.arange(pF,p),2*d+1) if pN>0 else np.zeros(0,dtype=int)
        cols_list=[]; vals_list=[]; rows_list=[]
        if pF>0:
            colsF = np.concatenate([IF[:,None]*d+cd, JF[:,None]*d+cd],1).ravel()
            valsF = np.concatenate([x[JF],x[IF]],1).ravel()
            cols_list.append(colsF); vals_list.append(valsF); rows_list.append(rowsF)
        if pN>0:
            colsN = np.concatenate([IN[:,None]*d+cd, JN[:,None]*d+cd, np.full((pN,1),n*d)],1).ravel()
            valsN = np.concatenate([x[JN],x[IN],-np.ones((pN,1))],1).ravel()
            cols_list.append(colsN); vals_list.append(valsN); rows_list.append(rowsN)
        import numpy as _np
        rows_all = _np.concatenate(rows_list) if rows_list else _np.zeros(0,dtype=int)
        cols_all = _np.concatenate(cols_list) if cols_list else _np.zeros(0,dtype=int)
        vals_all = _np.concatenate(vals_list) if vals_list else _np.zeros(0)
        A_ub = coo_matrix((vals_all,(rows_all,cols_all)),shape=(p,nv)).tocsr()
        b_ub = _np.concatenate([ (m0-delta-cF) if pF>0 else _np.zeros(0), -cN ]) if p>0 else _np.zeros(0)
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            # try larger radius once, else abort forcing
            if radius<0.04:
                radius*=1.8
                try:
                    res2 = linprog(obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=np.zeros(n),
                                   bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
                    if res2.status!=0:
                        break
                    res=res2
                except Exception:
                    break
            else:
                break
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        x = y
    return x

def solve():
    t0=time.time()
    HARD_END=t0+172.0
    rng=np.random.default_rng(12345)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(float)
            if a.shape==(163,4):
                a/=np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(a)
                if v<best_val:
                    best_val=v; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=rng.standard_normal(size=(163,4))
        best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    n,d=best.shape
    iu=_iu(n)
    if HARD_END-time.time()>25:
        best=_lp_polish(best,seconds=3.0,t_deadline=HARD_END-8)
        best_val=_max_cos(best)
    best_mask=_edge_mask(best,1e-5)
    # initial stress
    dual=None
    try:
        dual=_stress_from_duals(best)
    except Exception:
        dual=None
    pool=[best.copy()]
    pool_vals=[best_val]
    pool_masks=[best_mask.copy()]
    trial=0
    deltas=[5e-4,8e-4,1.1e-3,1.5e-3,2e-3]
    while time.time()<HARD_END-12:
        trial+=1
        # refresh duals periodically from champion
        if trial%12==1 or dual is None:
            try:
                nd=_stress_from_duals(best)
                if nd is not None:
                    dual=nd
            except Exception:
                pass
            m0,deg,sel_tight,c_all=_contact_info(best,1e-9)
        else:
            m0,deg,sel_tight,c_all=_contact_info(best,1e-9)
        m0=float(best_val)
        # choose base: mostly champion, sometimes diverse pool
        if len(pool)>1 and trial%4==0:
            bi=int(rng.integers(0,len(pool)))
            base=pool[bi]
        else:
            base=best
        mode=trial%3
        IF=None; JF=None
        delta=deltas[(trial//3)%len(deltas)]
        try:
            if mode==0:
                # highest-dual edges
                if dual is not None:
                    I,J,c,mw,w=dual
                    # dual corresponds to old best; if base is not best, map by ranking c on base
                    if base is not best:
                        Gb=base@base.T; cb=Gb[iu]
                        mb=float(cb.max())
                        # take top active pairs on base
                        order=np.argsort(-cb)[:40]
                        # filter to near-max within 0.02
                        filt=cb[order]>=mb-0.02
                        order=order[filt]
                        IF=iu[0][order][:35]; JF=iu[1][order][:35]
                    else:
                        order=np.argsort(-w)
                        k=int(20+(trial%3)*5)  # 20-30
                        # take top k with positive weight, pad if needed
                        pos=order[w[order]>0]
                        if len(pos)<k:
                            k=len(pos) if len(pos)>0 else 20
                            if k==0:
                                raise ValueError("no dual")
                        take=pos[:k]
                        # ensure 20-40
                        if len(take)<20:
                            # pad with next highest c among tight
                            extra=np.argsort(-c)
                            for e in extra:
                                if e not in set(take.tolist()):
                                    take=np.append(take,e)
                                if len(take)>=20:
                                    break
                        IF=I[take]; JF=J[take]
                else:
                    # fallback: tightest pairs
                    order=np.argsort(-c_all)
                    k=30
                    take=order[:k]
                    IF=iu[0][take]; JF=iu[1][take]
            elif mode==1:
                # high-degree points: all incident tight pairs of top points
                order_pts=np.argsort(-deg)
                npick=3+(trial%3)  # 3-5
                pts=order_pts[:npick]
                inset=set(pts.tolist())
                Iall=iu[0][sel_tight]; Jall=iu[1][sel_tight]
                mask=np.array([(a in inset or b in inset) for a,b in zip(Iall,Jall)])
                candI=Iall[mask]; candJ=Jall[mask]
                if len(candI)<10 or len(candI)>45:
                    # if too many, keep highest c; if too few, add tightest
                    Gb=base@base.T; cb=Gb[iu]
                    if len(candI)>45:
                        cc=Gb[candI,candJ]
                        o=np.argsort(-cc)[:35]
                        candI=candI[o]; candJ=candJ[o]
                    else:
                        o=np.argsort(-c_all)[:35]
                        candI=iu[0][o]; candJ=iu[1][o]
                IF=candI[:40]; JF=candJ[:40]
            else:
                # spatially localized patch
                Gb=base
                anchor=int(rng.integers(0,n))
                dd=Gb@Gb[anchor]
                # nearest neighbours in dot (largest dot)
                nn=np.argsort(-dd)[:12+(trial%4)]
                inset=set(nn.tolist())
                G2=base@base.T; c2=G2[iu]
                m2=float(c2.max())
                # active-ish pairs with at least one end in patch and c within 0.02 of max
                sel2=c2>=m2-0.02
                Iall=iu[0][sel2]; Jall=iu[1][sel2]
                mask=np.array([(a in inset or b in inset) for a,b in zip(Iall,Jall)])
                candI=Iall[mask]; candJ=Jall[mask]
                if len(candI)<10 or len(candI)>45:
                    o=np.argsort(-c2)[:35]
                    candI=iu[0][o]; candJ=iu[1][o]
                else:
                    if len(candI)>35:
                        cc=G2[candI,candJ]
                        o=np.argsort(-cc)[:35]
                        candI=candI[o]; candJ=candJ[o]
                # random subset to vary cluster
                if len(candI)>25:
                    idx=rng.choice(len(candI),size=25,replace=False)
                    candI=candI[idx]; candJ=candJ[idx]
                IF=candI; JF=candJ
        except Exception:
            continue
        if IF is None or len(IF)<8:
            continue
        if len(IF)>40:
            # subsample deterministically
            idx=rng.choice(len(IF),size=35,replace=False)
            IF=IF[idx]; JF=JF[idx]
        # ensure i<j ordering (already)
        # forcing from base
        x0=base.copy()
        # small jitter to break degeneracy
        x0=x0+1e-4*rng.standard_normal(size=x0.shape)
        x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        m_base=_max_cos(x0)
        xf=_forced_steps(x0,np.asarray(IF),np.asarray(JF),m_base,delta,steps=4,radius=0.018)
        # check topology change at 1e-5 before polish (optional)
        # release polish
        remaining=HARD_END-12-time.time()
        if remaining<=0.3:
            break
        budget=min(1.4,remaining)
        if trial%16==0:
            budget=min(2.2,remaining)
        y=_lp_polish(xf,seconds=budget,t_deadline=HARD_END-12)
        v=_max_cos(y)
        # topology distance
        try:
            ym=_edge_mask(y,1e-5)
            diff=int(np.count_nonzero(ym!=best_mask))
        except Exception:
            diff=0
            ym=None
        if v<best_val-1e-12:
            best=y; best_val=v
            try:
                best_mask=_edge_mask(best,1e-5)
            except Exception:
                pass
            pool=[best.copy()]+pool[:2]
            pool_vals=[best_val]+pool_vals[:2]
            pool_masks=[best_mask.copy()]+pool_masks[:2]
            try:
                dual=_stress_from_duals(best)
            except Exception:
                pass
        elif ym is not None and diff>80 and v<best_val+0.002:
            # diverse promising topology: keep in pool
            if len(pool)<4:
                pool.append(y); pool_vals.append(v); pool_masks.append(ym)
            else:
                # replace worst pool entry if better
                wi=int(np.argmax(pool_vals))
                if v<pool_vals[wi]:
                    pool[wi]=y; pool_vals[wi]=v; pool_masks[wi]=ym
    # final polish champion
    fb=max(0.5,HARD_END-time.time()-1.0)
    fb=min(fb,22.0)
    if fb>0.3:
        best=_lp_polish(best,seconds=fb)
        best_val=_max_cos(best)
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best

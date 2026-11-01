import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G >= m - thr
    cnt = mask.sum(axis=1)
    return m, cnt, G

def _tight_polish(x, seconds, window_extra=0.0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        base = max(20.0 * radius, 1e-9, window_extra)
        sel = c > m - base
        if int(sel.sum()) > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[-4000:]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[sel]; J = iu1[sel]
        cc = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = _max_cos(y)
        if v < best:
            x,best,radius = y,v,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _frozen_phase(x0, r, C_r, record, seconds):
    x = x0.copy()
    n,d = x.shape
    # freeze r exactly
    radius = 5e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n,1)
    # merit to minimize: max(max_nonr, max_r+eps) where eps=record-C_r
    def merit_of(X):
        G = X@X.T
        np.fill_diagonal(G,-2.0)
        mr = float(np.max(np.concatenate([G[r,:r],G[r,r+1:]]))) if n>1 else -2
        # non-r max
        G2 = G.copy()
        G2[r,:]=-2.0; G2[:,r]=-2.0
        mn = float(G2.max())
        return max(mn, mr + (record - C_r)), mn, mr
    cur_merit,_,_ = merit_of(x)
    it=0
    while time.time()-start < seconds and radius>1e-14:
        it+=1
        G = x@x.T
        cfull = G[iu0,iu1]
        # split
        is_r = (iu0==r)|(iu1==r)
        c_r = cfull[is_r]
        I_r = iu0[is_r]; J_r = iu1[is_r]
        c_n = cfull[~is_r]
        I_n = iu0[~is_r]; J_n = iu1[~is_r]
        # select non-r pairs near record or near current max
        m_n = float(c_n.max()) if len(c_n)>0 else -2
        thr_n = record - max(20*radius, 2e-4)
        # also include tightest relative to m_n
        sel_n = c_n > min(thr_n, m_n - max(20*radius,1e-9))
        # cap count
        if int(sel_n.sum())>2500:
            idx=np.where(sel_n)[0]
            order=np.argsort(c_n[idx])[-2500:]
            mm=np.zeros_like(sel_n); mm[idx[order]]=True; sel_n=mm
        # select r pairs that violate or near cap
        thr_r = C_r - max(20*radius, 2e-4)
        sel_r = c_r > thr_r
        # always keep at least nearest 8 of r
        if int(sel_r.sum())<8:
            order=np.argsort(c_r)[-8:]
            sel_r[order]=True
        I1=I_n[sel_n]; J1=J_n[sel_n]; cc1=c_n[sel_n]
        I2=I_r[sel_r]; J2=J_r[sel_r]; cc2=c_r[sel_r]
        p1=len(I1); p2=len(I2)
        p=p1+p2
        if p==0:
            radius*=0.5; continue
        nv=n*d+1
        rows=[]; cols=[]; vals=[]; bub=[]
        # non-r rows: xJ dxI + xI dxJ - t <= -cc
        # r rows: xJ dxI + xI dxJ + 0*t <= C_r - cc
        # build via lists then coo
        # vectorize
        if p1>0:
            rws1=np.repeat(np.arange(p1),2*d+1)
            c1=np.concatenate([I1[:,None]*d+np.arange(d), J1[:,None]*d+np.arange(d), np.full((p1,1),n*d)],1).ravel()
            v1=np.concatenate([x[J1],x[I1],-np.ones((p1,1))],1).ravel()
        else:
            rws1=np.empty(0,dtype=int); c1=np.empty(0,dtype=int); v1=np.empty(0)
        if p2>0:
            rws2=np.repeat(np.arange(p1,p),2*d)  # no t column
            # for each r-pair, one index is r; both blocks still present but r block will be forced 0 by bounds
            c2=np.concatenate([I2[:,None]*d+np.arange(d), J2[:,None]*d+np.arange(d)],1).ravel()
            v2=np.concatenate([x[J2],x[I2]],1).ravel()
        else:
            rws2=np.empty(0,dtype=int); c2=np.empty(0,dtype=int); v2=np.empty(0)
        R=np.concatenate([rws1,rws2])
        C=np.concatenate([c1,c2])
        V=np.concatenate([v1,v2])
        A_ub=coo_matrix((V,(R,C)),shape=(p,nv)).tocsr()
        b_ub=np.concatenate([-cc1, (C_r-cc2)]) if (p1>0 and p2>0) else ((-cc1) if p2==0 else (C_r-cc2))
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        bounds=[(-radius,radius)]*(n*d)+[(None,None)]
        # freeze r
        for k in range(d):
            bounds[r*d+k]=(0.0,0.0)
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=np.zeros(n),bounds=bounds,method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        dx=res.x[:-1].reshape(n,d)
        # enforce frozen exactly
        dx[r,:]=0.0
        y=x+dx
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        y[r,:]=x[r,:]  # keep frozen direction exactly
        nm,_,_ = merit_of(y)
        if nm < cur_merit - 1e-12:
            x=y; cur_merit=nm
            radius=min(radius*1.5,0.05)
        else:
            # accept if non-r max didn't explode too much but r improved? use small tolerance
            # otherwise shrink
            radius*=0.5
            # occasional sideways move: if r max improved, accept even if merit slightly worse?
            # check components
            _,mn_new,mr_new=merit_of(y)
            _,mn_old,mr_old=merit_of(x)
            if mr_new < mr_old - 5e-6 and mn_new < record + 2e-3:
                x=y; cur_merit=nm if nm<cur_merit else cur_merit
        if time.time()-start>=seconds:
            break
    return x

def _expanded_polish(x, r, seconds):
    # include all pairs within 5e-3 of max plus nearest 15 of r
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    start=time.time()
    iu0,iu1=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        G=x@x.T
        c=G[iu0,iu1]
        m=float(c.max())
        sel = c > m - max(20*radius, 5e-3)
        # ensure nearest 15 of r included
        is_r=(iu0==r)|(iu1==r)
        # find threshold for top15
        cr=c[is_r]
        if len(cr)>=15:
            thr15=np.sort(cr)[-15]
            sel = sel | (is_r & (c>=thr15))
        else:
            sel = sel | is_r
        if int(sel.sum())>4000:
            idx=np.where(sel)[0]
            # always keep r pairs
            must=is_r[idx]
            # keep all r must, plus tightest others to fill 4000
            idx_must=idx[must]
            idx_other=idx[~must]
            need=4000-len(idx_must)
            if need<0:
                # keep tightest 4000 among must? keep 15 tightest of r + ...
                order=np.argsort(c[idx_must])[-4000:]
                idx_must=idx_must[order]
                need=0
                idx_other=np.empty(0,dtype=int)
            else:
                if len(idx_other)>need:
                    order=np.argsort(c[idx_other])[-need:]
                    idx_other=idx_other[order]
            keep=np.concatenate([idx_must,idx_other])
            mm=np.zeros_like(sel); mm[keep]=True; sel=mm
        I=iu0[sel]; J=iu1[sel]; cc=c[sel]
        p=len(I)
        nv=n*d+1
        cols=np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        rows=np.repeat(np.arange(p),2*d+1)
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cc,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best:
            x,best,radius=y,v,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    BUDGET=172.0
    rng=np.random.default_rng(0)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(float)
            if a.shape==(173,5):
                an=a/np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(an)
                if v<best_val:
                    best_val=v; best=an.copy()
        except Exception:
            pass
    if best is None:
        best=rng.normal(size=(173,5)); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    n,d=173,5
    # ensure local optimum baseline
    # quick tight polish of best (short) to stabilize
    # keep global best
    global_best=best.copy(); global_val=best_val
    cur_base=best.copy()
    eps_list=[1e-4,3e-4,1e-3,3e-3]
    # alternate starting points between global_best and cur_base
    round_no=0
    while time.time()-t0 < BUDGET-12.0:
        round_no+=1
        for ei,eps in enumerate(eps_list):
            if time.time()-t0 >= BUDGET-12.0:
                break
            # choose start: mostly cur_base, sometimes global_best
            if round_no%2==0 and ei%2==0:
                x0=global_best.copy()
            else:
                x0=cur_base.copy()
            # tiny deterministic kick to others to allow topology change
            # identify rattler
            _,cnt,_= _contacts(x0)
            r=int(np.argmin(cnt))
            # small shake of non-r points (frozen r direction initially)
            # scale with eps
            shake=eps*2.0
            if shake>0.004: shake=0.004
            if shake<0.0003: shake=0.0003
            # deterministic noise
            noise=rng.normal(size=x0.shape)*shake
            noise[r,:]=0.0
            # also shake overloaded points a bit more
            order=np.argsort(-cnt)
            for v in order[:3]:
                if v!=r:
                    noise[v]*=2.0
            x0=x0+noise
            x0/=np.linalg.norm(x0,axis=1,keepdims=True)
            record_here=global_val
            C_r=record_here-eps
            # frozen phase ~2.5s
            rem=BUDGET-12.0-(time.time()-t0)
            if rem<5: break
            ft=min(2.5, rem*0.35)
            x1=_frozen_phase(x0, r, C_r, record_here, ft)
            # unfreeze expanded polish ~2s
            rem=BUDGET-12.0-(time.time()-t0)
            if rem<4: break
            pt=min(2.0, rem*0.3)
            x2=_expanded_polish(x1, r, pt)
            v2=_max_cos(x2)
            # contacts of r
            G2=x2@x2.T
            np.fill_diagonal(G2,-2.0)
            m2=float(G2.max())
            cnt_r=int((G2[r]>=m2-1e-9).sum())
            # update global best
            if v2 < global_val:
                global_val=v2; global_best=x2.copy()
                cur_base=x2.copy()  # move base to improved
                continue
            # accept even if worse when rattler active
            if cnt_r>=3 and v2 < global_val+5e-3:
                cur_base=x2.copy()
                # try to drive it down with tight polish
                rem=BUDGET-12.0-(time.time()-t0)
                if rem>3:
                    x3=_tight_polish(x2, min(1.8, rem*0.25))
                    v3=_max_cos(x3)
                    if v3 < global_val:
                        global_val=v3; global_best=x3.copy()
                        cur_base=x3.copy()
                    else:
                        # check if x3 still active, keep it if active
                        G3=x3@x3.T; np.fill_diagonal(G3,-2.0); m3=float(G3.max())
                        # original r may no longer be min; check its contacts
                        cnt_r3=int((G3[r]>=m3-1e-9).sum())
                        if cnt_r3>=3 and v3 < global_val+5e-3:
                            cur_base=x3.copy()
            # if no active rattler but v2 close to record, occasionally diversify base
            elif v2 < global_val+1e-4 and (rng.random()<0.1):
                # small chance to move base to explore
                pass
        # if cur_base stuck, re-seed with global_best + moderate kick
        if round_no%4==0:
            # verify cur_base merit; if its max far worse, reset to global
            vc=_max_cos(cur_base)
            if vc > global_val+5e-3:
                cur_base=global_best.copy()
    # final intensive polish of global best
    rem=BUDGET-(time.time()-t0)
    if rem>4:
        global_best=_tight_polish(global_best, min(rem-1.0, 6.0))
        global_val=_max_cos(global_best)
    global_best/=np.linalg.norm(global_best,axis=1,keepdims=True)
    return global_best

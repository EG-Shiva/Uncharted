import time
import hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _topo_hash(x, m=None, thr=1e-6):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    sel = c > m - thr
    ids = (iu0[0][sel].astype(np.int64)*n + iu0[1][sel].astype(np.int64))
    ids.sort()
    h = hashlib.md5(ids.tobytes()).hexdigest()
    return h, m

def _anneal(x, steps, beta0=30.0, beta1=600.0, lr0=0.05, lr1=0.005, mu=0.85):
    x = _norm(x.copy())
    v = np.zeros_like(x)
    lb = np.log(beta1/beta0)
    for t in range(steps):
        frac = t/max(1,steps-1)
        beta = beta0*np.exp(lb*frac)
        lr = lr0+(lr1-lr0)*frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1,keepdims=True)+1e-300
        A = (W @ x)/S
        dot = (A*x).sum(axis=1,keepdims=True)
        Gt = A - dot*x
        v = mu*v + Gt
        vd = (v*x).sum(axis=1,keepdims=True)
        v = v - vd*x
        x = x - lr*v
        x = _norm(x)
    return x

def _lp_polish(x, seconds, r0=1e-3):
    x = _norm(np.asarray(x,float).copy())
    n,d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds<=0:
        return xb
    start=time.time()
    radius=r0
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        G=x@x.T
        c=G[iu]
        m=c.max()
        sel=c>m-max(20*radius,1e-9)
        I=iu[0][sel]; J=iu[1][sel]
        p=len(I)
        if p==0:
            break
        # cap to worst 3000 pairs for speed
        if p>3000:
            idx=np.argpartition(-c[sel],3000-1)[:3000]
            I=I[idx]; J=J[idx]
            csv=c[sel][idx]
            p=3000
        else:
            csv=c[sel]
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        ci=I[:,None]*d+np.arange(d)
        cj=J[:,None]*d+np.arange(d)
        cols=np.concatenate([ci,cj,np.full((p,1),n*d)],axis=1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],axis=1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-csv,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        try:
            y=x+res.x[:-1].reshape(n,d)
        except Exception:
            radius*=0.5
            continue
        y=_norm(y)
        vv=_max_cos(y)
        if vv<best:
            best=vv; xb=y.copy(); x=y
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return xb

def _refine_hole(z0,X,steps=80):
    z=z0/max(np.linalg.norm(z0),1e-12)
    for s in range(steps):
        frac=s/max(1,steps-1)
        beta=400.0*(8000.0/400.0)**frac
        lr=0.05*(0.008/0.05)**frac
        D=X@z
        dm=D.max()
        W=np.exp(beta*(D-dm))
        S=W.sum()+1e-300
        g=(W@X)/S
        gt=g-float(g@z)*z
        z=z-lr*gt
        nrm=np.linalg.norm(z)
        if nrm<1e-12:
            break
        z/=nrm
    return z

def _find_holes(Xkeep,rng,k,n_cand=20000):
    d=Xkeep.shape[1]
    Z=rng.standard_normal((n_cand,d))
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    # chunked
    worst=np.empty(n_cand)
    cs=5000
    for a in range(0,n_cand,cs):
        b=min(a+cs,n_cand)
        worst[a:b]=(Z[a:b]@Xkeep.T).max(axis=1)
    order=np.argsort(worst)
    # take diverse: greedy separation 0.9
    sel_idx=[]
    sel=[]
    for idx in order:
        if len(sel)>=max(8,k*3):
            break
        c=Z[idx]
        ok=True
        for s in sel:
            if float(s@c)>0.90:
                ok=False; break
        if ok:
            sel.append(c); sel_idx.append(idx)
        elif len(sel)<k:
            sel.append(c); sel_idx.append(idx)
    # refine top few and pick best k sequential later; just return refined pool
    out=[]
    for c in sel[:max(6,k*2)]:
        z=_refine_hole(c,Xkeep,steps=80)
        out.append(z)
    if len(out)==0:
        out=[Z[order[0]]]
    return out, worst[order[0]]

def solve():
    t0=time.time()
    TIME_LIMIT=172.0
    rng=np.random.default_rng(1234567)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        try:
            xb=np.load("record.npy").astype(float)
        except Exception:
            xb=rng.standard_normal((292,4))
    if xb.shape!=(292,4) or not np.all(np.isfinite(xb)):
        xb=rng.standard_normal((292,4))
    xb=_norm(xb)
    best_score=_max_cos(xb)
    x_best=xb.copy()
    try:
        r=np.load("record.npy").astype(float)
        if r.shape==(292,4) and np.all(np.isfinite(r)):
            r=_norm(r)
            rv=_max_cos(r)
            if rv<best_score:
                best_score=rv; x_best=r.copy()
    except Exception:
        pass
    cur_X=x_best.copy(); cur_score=best_score
    h0,_=_topo_hash(x_best)
    archive={h0:best_score}
    n,d=292,4
    T0,T1=2e-6,1e-8
    FINAL_RESERVE=28.0
    explore_end=t0+TIME_LIMIT-FINAL_RESERVE
    it=0
    # initial squeeze of incumbent a bit
    # quick diversity: evaluate annealed copy
    while time.time()<explore_end:
        it+=1
        remain=explore_end-time.time()
        if remain<3.0:
            break
        if it%10==0:
            base=x_best.copy(); base_score=best_score
        else:
            base=cur_X.copy(); base_score=cur_score
        G=base@base.T
        np.fill_diagonal(G,-2.0)
        mval=float(G.max())
        rowmax=G.max(axis=1)
        degL=(G>mval-1e-4).sum(axis=1)
        rpick=rng.random()
        if rpick<0.55:
            iu=np.triu_indices(n,1)
            c=G[iu]
            kTop=min(40,len(c))
            part=np.argpartition(-c,kTop-1)[:kTop]
            pick=int(rng.choice(part))
            a=int(iu[0][pick]); b=int(iu[1][pick])
            na=np.argsort(-G[a])[:10]
            nb=np.argsort(-G[b])[:10]
            neigh=np.unique(np.concatenate([[a,b],na,nb]))
        else:
            jit=rng.random(n)*2e-6
            sc=rowmax+0.0008*degL.astype(float)+jit
            p0=int(np.argmax(sc))
            nn=np.argsort(-G[p0])[:14]
            neigh=np.unique(np.concatenate([[p0],nn]))
        kdel=int(rng.integers(1,4))
        kdel=min(kdel,len(neigh))
        w=rowmax[neigh]+0.0008*degL[neigh].astype(float)+rng.random(len(neigh))*1e-6
        dsel=np.argsort(-w)[:kdel]
        if rng.random()<0.25:
            dsel[0]=int(rng.integers(len(neigh)))
        del_idx=np.unique(neigh[dsel])
        kdel=len(del_idx)
        mask=np.ones(n,bool); mask[del_idx]=False
        X_keep=base[mask]
        # sequential hole insertion
        Xcur=X_keep.copy()
        new_pts=[]
        pool,_=_find_holes(Xcur,rng,kdel,n_cand=20000)
        # use pool + sequential refinement
        for jj in range(kdel):
            if time.time()>explore_end-1.0:
                break
            if jj<len(pool):
                z0=pool[jj % len(pool)]
            else:
                z0=rng.standard_normal(4); z0/=np.linalg.norm(z0)
            # re-refine against current Xcur
            z=_refine_hole(z0,Xcur,steps=60)
            z=z+5e-5*rng.standard_normal(4)
            z/=max(np.linalg.norm(z),1e-12)
            new_pts.append(z)
            Xcur=np.vstack([Xcur,z.reshape(1,4)])
        if len(new_pts)<kdel:
            continue
        Xnew=np.vstack([X_keep,np.stack(new_pts)])
        try:
            Xc=_anneal(Xnew,steps=220,beta0=30.0,beta1=500.0,lr0=0.05,lr1=0.004,mu=0.85)
        except Exception:
            continue
        if not np.all(np.isfinite(Xc)):
            continue
        try:
            Xc2=_lp_polish(Xc,min(1.2,max(0.3,remain-1.0)),r0=8e-4)
        except Exception:
            Xc2=Xc
        if not np.all(np.isfinite(Xc2)):
            continue
        cheap=_max_cos(Xc2)
        h,_=_topo_hash(Xc2)
        seen_before=(h in archive)
        prev_best_arch=archive.get(h,1e9)
        if seen_before and cheap>=prev_best_arch-1e-12 and cheap>=best_score-1e-12:
            continue
        if (h not in archive) or cheap<archive[h]:
            archive[h]=cheap
        is_new=(not seen_before)
        frac_el=(time.time()-t0)/max(1e-9,(explore_end-t0))
        frac_el=min(max(frac_el,0.0),1.0)
        T=T0*((T1/T0)**frac_el)
        delta=cheap-cur_score
        accept=False
        if cheap<cur_score-1e-15:
            accept=True
        elif cheap<best_score+4e-4:
            if delta<=0:
                accept=True
            else:
                if rng.random()<np.exp(-delta/max(T,1e-12)):
                    accept=True
        if accept:
            cur_X=Xc2.copy(); cur_score=cheap
        if cheap<best_score:
            best_score=cheap; x_best=Xc2.copy()
            cur_X=Xc2.copy(); cur_score=cheap
        # full polish only for new hashes close to incumbent, limited
        if is_new and cheap<best_score+2e-5 and (it%2==0 or cheap<best_score+3e-6):
            if (explore_end-time.time())>12.0:
                try:
                    Xf=_lp_polish(Xc2.copy(),min(7.0,explore_end-time.time()-0.8),r0=5e-4)
                    vf=_max_cos(Xf)
                    hf,_=_topo_hash(Xf)
                    if hf not in archive or vf<archive[hf]:
                        archive[hf]=vf
                    if vf<best_score:
                        best_score=vf; x_best=Xf.copy()
                    if vf<cur_score:
                        cur_X=Xf.copy(); cur_score=vf
                    elif rng.random()<np.exp(-max(0.0,vf-cur_score)/max(T,1e-12))*0.5:
                        cur_X=Xf.copy(); cur_score=vf
                except Exception:
                    pass
    # final intensive polish of best
    time_left=TIME_LIMIT-(time.time()-t0)-1.5
    time_left=float(max(6.0,min(30.0,time_left)))
    try:
        # alternate anneal + LP
        x_best=_anneal(x_best,steps=600,beta0=40.0,beta1=2000.0,lr0=0.03,lr1=0.001,mu=0.9)
        x_best=_lp_polish(x_best,time_left,r0=8e-4)
    except Exception:
        pass
    x_best=_norm(x_best)
    return x_best

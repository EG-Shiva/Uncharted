import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _pair_stats(X, gap=1e-6):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    m = float(c.max())
    cnt = int(np.count_nonzero(c > m - gap))
    return m, cnt, c

def _lse_full(X0, betas, iters, deadline):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm(X0.copy())
    for beta, mi in zip(betas, iters):
        if time.time() > deadline:
            break
        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            c = G[iu]
            mv = c.max()
            E = np.exp(beta*(c-mv))
            S = E.sum()+1e-300
            f = float(mv + np.log(S)/beta)
            W = np.zeros((n, n))
            W[iu] = E/S
            W = W + W.T
            GX = W @ X
            dot = np.sum(GX*X, axis=1, keepdims=True)
            Gt = GX - dot*X
            GY = Gt/nr
            return f, GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol':1e-14, 'gtol':1e-5, 'maxcor':10})
            Y = _norm(res.x.reshape(n, d))
        except Exception:
            pass
        if time.time() > deadline:
            break
    return _norm(Y)

def _lp_polish(X, deadline, radius0=8e-4, cap=2800):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = c.max()
        thr = m - max(20.0*radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            o = np.argsort(-c[idx])[:cap]
            mk = np.zeros_like(sel, dtype=bool)
            mk[idx[o]] = True
            sel = mk
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius*=0.5
            continue
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method='highs')
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = _norm(X+res.x[:-1].reshape(n,d))
        v = _max_cos(y)
        if v < best:
            X,best=y,v
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return X

def _refine_y(y0, Xcur, beta=250, maxiter=80):
    from scipy.optimize import minimize
    def fun(yf):
        nr = float(np.sqrt(np.sum(yf*yf)))+1e-30
        y = yf/nr
        c = Xcur @ y
        mv = float(c.max())
        E = np.exp(beta*(c-mv))
        S = float(E.sum())+1e-300
        f = float(mv+np.log(S)/beta)
        w = E/S
        G = w @ Xcur
        dot = float(G @ y)
        Gt = G - dot*y
        g = Gt/nr
        return f, g
    try:
        res = minimize(fun, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter':int(maxiter),'ftol':1e-14,'gtol':1e-6,'maxcor':8})
        y = res.x/np.linalg.norm(res.x)
        if not np.all(np.isfinite(y)):
            return _norm(y0[None,:])[0]
        return y
    except Exception:
        return _norm(y0[None,:])[0]

def _local_relax(Xfull, sub_idx, beta=250, maxiter=60):
    from scipy.optimize import minimize
    Xfull = _norm(Xfull)
    Nf, d = Xfull.shape
    sub_idx = np.asarray(sub_idx, dtype=int)
    s = len(sub_idx)
    if s==0 or s>=Nf:
        return Xfull
    mask = np.ones(Nf, dtype=bool); mask[sub_idx]=False
    fix_idx = np.where(mask)[0]
    Xfix = Xfull[fix_idx]
    Y0 = Xfull[sub_idx].ravel().copy()
    ss_iu = np.triu_indices(s,1) if s>1 else None
    def fun(Yf):
        Ym = Yf.reshape(s,d)
        nr = np.sqrt(np.sum(Ym*Ym,axis=1,keepdims=True))+1e-30
        Xs = Ym/nr
        Csf = Xs @ Xfix.T
        mv1 = Csf.max() if Csf.size else -2.0
        if s>1:
            Gss = Xs @ Xs.T
            c2 = Gss[ss_iu]
            mv2 = c2.max() if c2.size else -2.0
            mv = mv1 if mv1>mv2 else mv2
            E1 = np.exp(beta*(Csf-mv))
            E2 = np.exp(beta*(c2-mv))
            S = E1.sum()+E2.sum()+1e-300
            f = float(mv+np.log(S)/beta)
            W1 = E1/S
            W2m = np.zeros((s,s))
            W2m[ss_iu]=E2/S
            W2m = W2m+W2m.T
            GX = W1 @ Xfix + W2m @ Xs
        else:
            mv = mv1
            E1 = np.exp(beta*(Csf-mv))
            S = E1.sum()+1e-300
            f = float(mv+np.log(S)/beta)
            W1 = E1/S
            GX = W1 @ Xfix
        dot = np.sum(GX*Xs,axis=1,keepdims=True)
        Gt = GX-dot*Xs
        GY = Gt/nr
        return f, GY.ravel()
    try:
        res = minimize(fun, Y0, jac=True, method='L-BFGS-B',
                       options={'maxiter':int(maxiter),'ftol':1e-14,'gtol':1e-5,'maxcor':8})
        Ym = res.x.reshape(s,d)
        Xs = _norm(Ym)
        Xnew = Xfull.copy()
        Xnew[sub_idx]=Xs
        return _norm(Xnew)
    except Exception:
        return Xfull

def solve():
    t0=time.time()
    deadline=t0+170.0
    surg_deadline=t0+138.0
    rng=np.random.default_rng(13808)
    n,d=138,8
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            vv=np.load(fn).astype(float)
            if vv.shape==(n,d):
                vv=_norm(vv)
                c=_max_cos(vv)
                if c<best_val:
                    best_val=c; best=vv.copy()
        except Exception:
            pass
    if best is None:
        best=_norm(rng.normal(size=(n,d))); best_val=_max_cos(best)
    incumb=best.copy(); incumb_val=best_val
    _,best_cnt,_=_pair_stats(best,gap=1e-6)

    # degrees at m-1e-6
    def degrees(X):
        m,cnt,c=_pair_stats(X,gap=1e-6)
        N=X.shape[0]
        iu=np.triu_indices(N,1)
        # degree count per point for pairs > m-1e-6
        sel=c>m-1e-6
        I=iu[0][sel]; J=iu[1][sel]
        deg=np.zeros(N,dtype=int)
        np.add.at(deg,I,1); np.add.at(deg,J,1)
        return m,cnt,deg
    try:
        m0,_,deg0=degrees(best)
    except Exception:
        deg0=np.zeros(n,dtype=int)
    order_asc=np.argsort(deg0, kind='stable')
    order_desc=order_asc[::-1]
    low2=list(order_asc[:2])

    ks=[3,5,8,12,18]
    # build removal variants: for each k, primary top-k and alternate (skip top1)
    jobs=[]
    for k in ks:
        topk=list(order_desc[:k])
        jobs.append((k,topk,0))
        alt=list(order_desc[1:k+1])
        if len(alt)==k and set(alt)!=set(topk):
            jobs.append((k,alt,1))
    # limit jobs to fit time: interleave small first
    # order: (3,0),(5,0),(8,0),(8,1),(12,0),(12,1),(18,0),(18,1),(3,1),(5,1)
    pref=[]
    for k in ks:
        for v in (0,1):
            for (kk,tt,vv) in jobs:
                if kk==k and vv==v and (kk,tt,vv) not in pref:
                    pref.append((kk,tt,vv))
    jobs=pref

    NCAND=20000
    for (k,topk,variant) in jobs:
        if time.time()>surg_deadline:
            break
        removal=sorted(set(low2+topk))
        # want exactly 2+k distinct
        if len(removal)<2+k:
            # fill with next highest
            for idx in order_desc:
                if idx not in removal:
                    removal.append(int(idx))
                    if len(removal)>=2+k:
                        break
        removal=sorted(removal)[:2+k]
        keep=np.ones(n,dtype=bool); keep[removal]=False
        Xcore=best[keep].copy()
        # relax remaining core briefly
        t_rem=min(surg_deadline-2.0, time.time()+12.0)
        try:
            Xcore=_lse_full(Xcore,[150.0,350.0],[70,70],min(t_rem,time.time()+6.0))
        except Exception:
            pass
        Xcur=Xcore.copy()
        nrem=len(removal)
        ok=True
        for step in range(nrem):
            if time.time()>surg_deadline-6.0:
                ok=False; break
            mcur=Xcur.shape[0]
            # 20k random candidates
            try:
                Cand=_norm(rng.normal(size=(NCAND,d)))
                Dots=Cand @ Xcur.T
                worst=Dots.max(axis=1)
                bi=np.argsort(worst, kind='stable')[:4]
                best_starts=Cand[bi]
                best_w=worst[bi]
            except Exception:
                ok=False; break
            # refine top 3 starts by minimising max_j
            cands=[best_starts[0]]
            cands_w=[float(best_w[0])]
            for si in range(min(3,len(best_starts))):
                if time.time()>surg_deadline-5.0:
                    break
                try:
                    yref=_refine_y(best_starts[si].copy(),Xcur,beta=250,maxiter=70)
                    wref=float((Xcur @ yref).max())
                    cands.append(yref); cands_w.append(wref)
                except Exception:
                    pass
            # also add locally minimised from best random is already included
            q=np.argmin(np.asarray(cands_w))
            ynew=cands[q]
            ynew=ynew/np.linalg.norm(ynew)
            # append
            Xcur=np.vstack([Xcur,ynew[None,:]])
            Xcur=_norm(Xcur)
            # local relax with neighbours >0.35
            try:
                dots_n=(Xcur[:-1] @ Xcur[-1])
                nidx=np.where(dots_n>0.35)[0]
                # cap to 30 closest if too many
                if len(nidx)>30:
                    o=np.argsort(-dots_n[nidx])[:30]
                    nidx=nidx[o]
                sub=np.concatenate([nidx,np.array([Xcur.shape[0]-1])])
                if len(nidx)>0:
                    Xcur=_local_relax(Xcur,sub,beta=250,maxiter=50)
            except Exception:
                pass
        if not ok:
            continue
        # after all insertions, LSE + LP polish
        try:
            tpol=min(deadline-14.0, surg_deadline)
            Xcur=_lse_full(Xcur,[120.0,300.0,550.0],[90,120,120],min(tpol,time.time()+7.0))
            v_cur=_max_cos(Xcur)
            # LP polish with remaining budget
            lp_dl=min(deadline-10.0, time.time()+4.0)
            if time.time()<lp_dl-0.5:
                Xp=_lp_polish(Xcur,lp_dl)
                vp=_max_cos(Xp)
                if vp<=v_cur:
                    Xcur, v_cur = Xp, vp
        except Exception:
            continue
        try:
            v_cur=_max_cos(Xcur)
            _,cnt_cur,_=_pair_stats(Xcur,gap=1e-6)
        except Exception:
            continue
        # retain only if true max or near-contact count improves
        if (v_cur<best_val-1e-12) or (v_cur<=best_val+1e-9 and cnt_cur<best_cnt):
            best=Xcur.copy(); best_val=v_cur; best_cnt=cnt_cur
            # recompute degrees for next jobs based on new best
            try:
                _,_,deg0=degrees(best)
                order_asc=np.argsort(deg0,kind='stable'); order_desc=order_asc[::-1]
                low2=list(order_asc[:2])
            except Exception:
                pass
            # refresh remaining jobs' removal sets? keep as is for determinism
        # strict internal budget: if time low break
        if time.time()>surg_deadline:
            break

    # final polish on global best
    try:
        if time.time()<deadline-8.0:
            Xj=_lse_full(best,[150.0,400.0],[120,150],deadline-5.0)
            vj=_max_cos(Xj)
            _,cj,_=_pair_stats(Xj,gap=1e-6)
            if (vj<best_val-1e-12) or (vj<=best_val+1e-9 and cj<best_cnt):
                best, best_val, best_cnt = Xj, vj, cj
    except Exception:
        pass
    try:
        if time.time()<deadline-3.0:
            Xp=_lp_polish(best,deadline-1.0)
            vp=_max_cos(Xp)
            _,cp,_=_pair_stats(Xp,gap=1e-6)
            if (vp<best_val-1e-12) or (vp<=best_val+1e-9 and cp<best_cnt):
                best= Xp
    except Exception:
        pass
    # safety fallback
    try:
        if _max_cos(best)>incumb_val+1e-12:
            best=incumb.copy()
    except Exception:
        best=incumb.copy()
    best=_norm(best)
    return best

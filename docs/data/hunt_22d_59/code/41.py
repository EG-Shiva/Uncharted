import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _adam_lse(X0, steps, beta0, beta1, lr, rng, t_end):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    pow1 = b1; pow2 = b2
    for t in range(1, steps+1):
        if (t & 63) == 0 and time.time() > t_end:
            break
        beta = beta0 + (beta1-beta0)*(t/steps)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        W = np.exp(beta*(G-gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)+1e-16
        D = (W @ X)/rs
        dot = (D*X).sum(axis=1, keepdims=True)
        Gt = D - dot*X
        m *= b1; m += (1.0-b1)*Gt
        v *= b2; v += (1.0-b2)*(Gt*Gt)
        mhat = m/(1.0-pow1); vhat = v/(1.0-pow2)
        pow1*=b1; pow2*=b2
        step = lr*mhat/(np.sqrt(vhat)+eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True)+1e-12
        scale = np.minimum(1.0, 0.05/(rn+1e-12))
        X = X - step*scale
        X = _normalize(X)
        if (t % 25)==0:
            cur = _max_cos(X)
            if cur < best:
                best = cur; bestX = X.copy()
    return bestX, best

def _greedy_polish(X0, iters, step0, step1, rng, t_end):
    Xcur = X0.copy()
    best = _max_cos(Xcur)
    bestX = Xcur.copy()
    decay = (step1/step0)**(1.0/max(iters-1,1))
    step = step0
    for t in range(iters):
        if (t & 63)==0 and time.time()>t_end:
            break
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        mval = G.max()
        delta = max(0.004, 1.5*step)
        thr = mval-delta
        Mask = G>thr
        W = np.where(Mask,(G-thr+0.01),0.0)
        np.fill_diagonal(W,0.0)
        Gr = W @ Xcur
        dot = (Gr*Xcur).sum(axis=1,keepdims=True)
        Gt = Gr-dot*Xcur
        nrm = np.linalg.norm(Gt,axis=1,keepdims=True)+1e-12
        has = (nrm[:,0]>1e-9)
        upd = np.zeros_like(Xcur)
        upd[has]=Gt[has]/nrm[has]
        if np.any(~has):
            nh = int((~has).sum())
            R = rng.standard_normal(size=(nh,Xcur.shape[1]))
            Xr = Xcur[~has]
            R = R-(R*Xr).sum(axis=1,keepdims=True)*Xr
            rn2 = np.linalg.norm(R,axis=1,keepdims=True)+1e-12
            upd[~has]=R/rn2*0.3
        Xcur = Xcur-step*upd
        Xcur=_normalize(Xcur)
        cur=_max_cos(Xcur)
        if cur<best:
            best=cur; bestX=Xcur.copy()
        step*=decay
    return bestX,best

def _tangent_bases(X):
    n,d = X.shape
    Tb = np.empty((n,d,d-1))
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        cols = [j for j in range(d) if j!=k]
        M = np.zeros((d,d-1))
        for c,j in enumerate(cols):
            e = np.zeros(d); e[j]=1.0
            M[:,c]=e - x[j]*x
        Q,R = np.linalg.qr(M)
        # ensure tangent (fix sign drift)
        # Q spans tangent space
        Tb[i]=Q
    return Tb

def _tight_pairs(X, thr_window, max_p):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G,-2.0)
    mval = float(G.max())
    iu,ju = np.triu_indices(n,1)
    g = G[iu,ju]
    # select within window, tightest first
    mask = g >= (mval-thr_window)
    idx = np.where(mask)[0]
    if idx.size==0:
        # take top 500
        o = np.argsort(-g)[:min(max_p,g.size)]
        return [(int(iu[k]),int(ju[k]),float(g[k])) for k in o], mval
    # sort by g desc
    order = np.argsort(-g[idx])
    idx = idx[order]
    if idx.size>max_p:
        idx=idx[:max_p]
    return [(int(iu[k]),int(ju[k]),float(g[k])) for k in idx], mval

def _solve_lp(X, Tb, pairs, cvals, cur_max, trust, forbid_pair=None, gap=0.003, t_end=None):
    try:
        from scipy.optimize import linprog
    except Exception:
        return None
    n,d1 = X.shape[0], X.shape[1]
    dm = d1-1
    N = n*dm
    P = len(pairs)
    extra = 1 if forbid_pair is not None else 0
    # build dense A_ub (P+extra) x (N+1)
    try:
        A = np.zeros((P+extra, N+1), dtype=np.float64)
        b = np.zeros((P+extra,), dtype=np.float64)
    except Exception:
        return None
    # precompute Tx products on demand
    for r,(i,j,c) in enumerate(zip(pairs,cvals) if isinstance(pairs[0],tuple) else []):
        pass
    # pairs is list of (i,j,g)
    for r,pr in enumerate(pairs):
        i,j,g = pr
        ci = Tb[i].T @ X[j]  # (dm,)
        cj = Tb[j].T @ X[i]
        A[r, i*dm:(i+1)*dm]=ci
        A[r, j*dm:(j+1)*dm]=cj
        A[r,N]=-1.0
        b[r]=-float(g)
    if forbid_pair is not None:
        fp,fq = forbid_pair
        ci = Tb[fp].T @ X[fq]
        cj = Tb[fq].T @ X[fp]
        row = P
        A[row, fp*dm:(fp+1)*dm]=-ci
        A[row, fq*dm:(fq+1)*dm]=-cj
        A[row,N]=0.0
        b[row]=-float(gap)
    cvec = np.zeros(N+1); cvec[N]=1.0
    bounds = [(-trust,trust)]*N + [(-1.0,2.0)]
    try:
        res = linprog(cvec, A_ub=A, b_ub=b, bounds=bounds, method='highs', options={'time_limit':4.0,'presolve':True})
    except Exception:
        return None
    return res

def solve():
    import time as _time
    import numpy as _np
    t0=_time.time()
    BUDGET=168.0
    t_end_global=t0+BUDGET
    rng=_np.random.default_rng(0)
    n,d=59,22
    def rand_X():
        Y=rng.standard_normal(size=(n,d))
        return _normalize(Y)
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=_np.load(fn)
            if a.shape==(n,d):
                a=_np.array(a,dtype=float)
                if not _np.all(_np.isfinite(a)):
                    continue
                a=_normalize(a)
                if _np.all(_np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands)==0:
        cands.append(rand_X())
    vals=[_max_cos(c) for c in cands]
    bi=int(_np.argmin(vals))
    gbestX=cands[bi].copy()
    gbest=float(vals[bi])
    # quick polish baseline to jammed optimum
    try:
        Xp,vp=_adam_lse(gbestX,1200,50.0,300.0,0.010,rng,min(t_end_global-8.0,_time.time()+6.0))
        if vp<gbest:
            gbest=vp; gbestX=Xp.copy()
        Xp2,vp2=_greedy_polish(Xp if vp<gbest+1 else gbestX,600,0.004,0.0002,rng,t_end_global-6.0)
        if vp2<gbest:
            gbest=vp2; gbestX=Xp2.copy()
    except Exception:
        pass
    curX=gbestX.copy(); curMax=gbest
    tried_forbid=set()
    # LP-guided topology loop: at most ~22 attempts
    for it in range(22):
        if _time.time()>t_end_global-12.0:
            break
        # build tight set
        try:
            pairs,mval=_tight_pairs(curX,1e-4,1200)
        except Exception:
            break
        if len(pairs)<10:
            break
        cvals=[p[2] for p in pairs]
        # tangent bases
        try:
            Tb=_tangent_bases(curX)
        except Exception:
            break
        if _time.time()>t_end_global-11.0:
            break
        res=_solve_lp(curX,Tb,pairs,cvals,curMax,trust=1e-3,forbid_pair=None,gap=0.003)
        if res is None or not hasattr(res,'x') or res.x is None:
            # fallback: small shake + polish
            amp=0.02
            Xt=curX+amp*rng.standard_normal(size=(n,d)); Xt=_normalize(Xt)
            Xt2,vt2=_adam_lse(Xt,600,40.0,250.0,0.012,rng,t_end_global-9.0)
            if vt2<curMax-1e-9:
                curX=Xt2.copy(); curMax=vt2
                if vt2<gbest:
                    gbest=vt2; gbestX=Xt2.copy()
            continue
        try:
            t_pred=float(res.fun)
        except Exception:
            t_pred=curMax
        Nloc=n*(d-1)
        # extract step if predicted improvement
        improved=False
        if t_pred < curMax-5e-7 and _np.all(_np.isfinite(res.x)):
            V=res.x[:Nloc]
            # backtracking
            Vm=V.reshape(n,d-1)
            # construct U
            U=_np.empty((n,d))
            for i in range(n):
                U[i]=Tb[i]@Vm[i]
            for scale in (1.0,0.5,0.25):
                Xt=_normalize(curX+scale*U)
                av=_max_cos(Xt)
                if av<curMax-1e-9:
                    # short polish from there
                    rem=t_end_global-_time.time()
                    if rem<3:
                        curX=Xt.copy(); curMax=av
                        if av<gbest:
                            gbest=av; gbestX=Xt.copy()
                    else:
                        Xp,vo=_adam_lse(Xt,700,60.0,300.0,0.008,rng,min(t_end_global-4.0,_time.time()+min(4.0,rem*0.5)))
                        if vo<curMax:
                            curX=Xp.copy(); curMax=vo
                            if vo<gbest:
                                gbest=vo; gbestX=Xp.copy()
                        elif av<curMax:
                            curX=Xt.copy(); curMax=av
                    improved=True
                    break
            if improved:
                continue
        # certified locally optimal (or no step) -> break a stressed contact
        # get dual stress if available
        stress=None
        try:
            iq=res.get('ineqlin',None) if isinstance(res,dict) else getattr(res,'ineqlin',None)
            marg=None
            if iq is not None:
                if isinstance(iq,dict):
                    marg=iq.get('marginals',None)
                else:
                    marg=getattr(iq,'marginals',None)
            if marg is not None:
                marg=_np.asarray(marg,dtype=float).ravel()
                if marg.size>=len(pairs) and _np.all(_np.isfinite(marg)):
                    stress=-marg[:len(pairs)]
                    stress=_np.maximum(stress,0.0)
        except Exception:
            stress=None
        # choose forbid pair
        cand_order=None
        if stress is not None and _np.max(stress)>1e-12:
            cand_order=_np.argsort(-stress)
        else:
            # degree heuristic: endpoints degree within 1e-9
            try:
                Gc=curX@curX.T; _np.fill_diagonal(Gc,-2.0)
                mv=float(Gc.max())
                maskc=Gc>=(mv-1e-9)
                deg=maskc.sum(axis=1)
            except Exception:
                deg=_np.zeros(n)
            scores=[]
            for k,pr in enumerate(pairs):
                i,j,g=pr
                scores.append(deg[i]+deg[j]-abs(g-mv)*1000)
            cand_order=_np.argsort(-_np.asarray(scores))
        chosen=None
        for k in cand_order[:60]:
            pr=pairs[int(k)]
            key=(min(pr[0],pr[1]),max(pr[0],pr[1]))
            if key not in tried_forbid:
                chosen=(pr[0],pr[1])
                tried_forbid.add(key)
                break
        if chosen is None:
            # all tried: pick random tight pair
            pr=pairs[int(rng.integers(len(pairs)))]
            chosen=(pr[0],pr[1])
        # resolve with forced gap, larger trust
        gap=float(0.002+0.001*(it%3))  # 0.002-0.004
        trust_b=8e-3
        if _time.time()>t_end_global-10.0:
            break
        res2=_solve_lp(curX,Tb,pairs,cvals,curMax,trust=trust_b,forbid_pair=chosen,gap=gap)
        Xt_forced=None
        if res2 is not None and getattr(res2,'success',False) and res2.x is not None and _np.all(_np.isfinite(res2.x)):
            V2=res2.x[:Nloc].reshape(n,d-1)
            U2=_np.empty((n,d))
            for i in range(n):
                U2[i]=Tb[i]@V2[i]
            Xt_forced=_normalize(curX+U2)
        else:
            # manual push apart
            a,b_ = chosen
            xa=curX[a].copy(); xb=curX[b_].copy()
            da=xb-_np.dot(xb,xa)*xa; db=xa-_np.dot(xa,xb)*xb
            na=_np.linalg.norm(da)+1e-12; nb=_np.linalg.norm(db)+1e-12
            da/=na; db/=nb
            stepm=0.015
            Xt_forced=curX.copy()
            Xt_forced[a]-=stepm*da; Xt_forced[b_]-=stepm*db
            # small random to neighbourhood
            Xt_forced+=0.002*rng.standard_normal(size=(n,d))
            Xt_forced=_normalize(Xt_forced)
        # polish forced point
        rem=t_end_global-_time.time()
        if rem<2.5:
            break
        Xp,vo=_adam_lse(Xt_forced,800,50.0,320.0,0.010,rng,min(t_end_global-3.0,_time.time()+min(5.0,rem*0.6)))
        # greedy touch
        if _time.time()<t_end_global-3.5:
            Xp,vo2=_greedy_polish(Xp,400,0.003,0.0002,rng,t_end_global-3.0)
            vo=min(vo,vo2) if vo2<vo else vo
            # _greedy returns best; vo already best
            # re-evaluate to be safe
            # Xp is bestX from greedy
        if vo<gbest-1e-12:
            gbest=vo; gbestX=Xp.copy()
            curX=Xp.copy(); curMax=vo
        elif vo<curMax+0.0015:
            # move to new basin even if slightly worse to explore topology
            curX=Xp.copy(); curMax=vo
        else:
            # stay, will try different contact next iter
            # add tiny shake to avoid exact repeat
            pass
    # final fine polish of global best
    if _time.time()<t_end_global-1.0:
        try:
            Xf,vf=_greedy_polish(gbestX,700,0.0015,0.00005,rng,t_end_global-0.2)
            if vf<gbest:
                gbest=vf; gbestX=Xf.copy()
        except Exception:
            pass
    gbestX=_normalize(gbestX)
    return gbestX

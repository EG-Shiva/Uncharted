import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_pool(rng):
    pts = []
    d = 9
    # signed basis
    for i in range(d):
        for s in (-1.0, 1.0):
            v = np.zeros(d); v[i] = s
            pts.append(v)
    # (±1,±1,0..)/sqrt2 : all 144
    for i in range(d):
        for j in range(i+1, d):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(d); v[i]=si; v[j]=sj; v/=np.sqrt(2.0)
                    pts.append(v)
    # (±1,±1,±1)/sqrt3 : all 672
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        for sk in (-1.0, 1.0):
                            v=np.zeros(d); v[i]=si; v[j]=sj; v[k]=sk; v/=np.sqrt(3.0)
                            pts.append(v)
    # cube vertices (±1..)/3 : sample 400 of 512 to cap size
    idx = rng.choice(512, size=400, replace=False)
    for c in idx:
        v = np.ones(d)
        for i in range(d):
            if (c>>i)&1: v[i]=-1.0
        v/=3.0
        pts.append(v)
    P = np.stack(pts, axis=0)
    # add random sphere points for diversity
    R = rng.standard_normal((500, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    P = np.concatenate([P, R], axis=0)
    # normalize just in case
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P

def _fps(P, n, start):
    M = P.shape[0]
    sel = [int(start % M)]
    # max dot to selected set
    cur = P @ P[sel[0]]
    cur[sel[0]] = np.inf
    selset = {sel[0]}
    for _ in range(1, n):
        bi = int(np.argmin(cur))
        sel.append(bi); selset.add(bi)
        cur[bi] = np.inf
        # update
        d = P @ P[bi]
        # cur = max(cur, d) for unselected
        # keep inf for selected
        mask = np.ones(M, dtype=bool)
        # faster: only update where d>cur
        upd = d > cur
        # don't touch selected (inf)
        cur[upd] = d[upd]
        cur[bi] = np.inf
    return np.array(sel, dtype=np.int64)

def _swap_improve(P, sel, max_iter=80, rng=None):
    n = len(sel)
    M = P.shape[0]
    sel = sel.copy()
    in_set = np.zeros(M, dtype=bool); in_set[sel]=True
    X = P[sel]
    best = _max_cos(X)
    for it in range(max_iter):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = np.max(G, axis=1)
        m = float(np.max(rm))
        # order worst
        order = np.argsort(-rm)[:6]
        improved = False
        # outside indices
        out_idx = np.where(~in_set)[0]
        # precompute for speed: for each worst candidate find best replacement
        # limit out candidates evaluated: use all (fast matmul)
        for wi_pos, wi in enumerate(order):
            # X without wi
            mask = np.ones(n, dtype=bool); mask[wi_pos if False else 0]=False
            # actually need X_rest = X without row wi
            Xr = np.delete(X, wi, axis=0)  # (n-1,d)
            D = out_idx  # alias
            S = P[out_idx] @ Xr.T  # (Mo, n-1)
            mo = np.max(S, axis=1)
            bi = int(np.argmin(mo))
            if mo[bi] < m - 1e-12:
                # perform swap, check true global max
                cand_sel = sel.copy()
                old = sel[wi]
                new = out_idx[bi]
                cand_sel[wi] = new
                Xc = P[cand_sel]
                v = _max_cos(Xc)
                if v < best - 1e-12:
                    # accept
                    in_set[old]=False; in_set[new]=True
                    sel = cand_sel; X = Xc; best = v
                    improved = True
                    break
        if not improved:
            # random kick to escape: replace a random of top-20 worst with greedy best
            if it % 3 == 2 and rng is not None:
                pool_w = np.argsort(-rm)[:20]
                wi = int(rng.choice(pool_w))
                Xr = X[np.arange(n)!=wi]
                # sample subset of outside for speed
                take = min(600, len(out_idx))
                samp = rng.choice(out_idx, size=take, replace=False)
                S = P[samp] @ Xr.T
                mo = np.max(S, axis=1)
                bi = int(np.argmin(mo))
                cand_sel = sel.copy(); cand_sel[wi]=int(samp[bi])
                Xc = P[cand_sel]; v=_max_cos(Xc)
                # accept if not much worse (allow side step)
                if v < best + 0.01:
                    in_set[sel[wi]]=False; in_set[samp[bi]]=True
                    sel=cand_sel; X=Xc; best=v
            else:
                break
    return sel, best

def _build_from_reps(R, extra):
    # R: (nr,9), cyclic rolls -> nr*9 points
    nr = R.shape[0]
    lst = []
    for k in range(nr):
        for s in range(9):
            lst.append(np.roll(R[k], s))
    lst.append(extra)
    X = np.stack(lst, axis=0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _symm_opt(nr, seed, steps=320, k0=15.0, k1=160.0, lr0=0.06, lr1=0.008):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((nr, 9)); R/=np.linalg.norm(R,axis=1,keepdims=True)
    e = rng.standard_normal(9); e/=np.linalg.norm(e)
    for s in range(steps):
        frac = s/max(1,steps-1)
        k = k0+(k1-k0)*frac
        lr = lr0+(lr1-lr0)*frac
        X = _build_from_reps(R, e)
        n = X.shape[0]
        G = X @ X.T; np.fill_diagonal(G,-2.0)
        gmax = float(np.max(G))
        W = np.exp(np.clip(k*(G-gmax),-60,0.0))
        np.fill_diagonal(W,0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i',Gr,X)[:,None]
        Gt = Gr - dot*X
        # map Gt to grads of R and e
        # Gt rows 0..nr*9-1 correspond to rolls; last is e
        Gt_core = Gt[:-1].reshape(nr,9,9)  # [k,s,:]
        gradR = np.zeros_like(R)
        for kk in range(nr):
            for ss in range(9):
                gradR[kk] += np.roll(Gt_core[kk,ss], -ss)
        grade = Gt[-1]
        # tangent projection for R rows and e (approx: project)
        gradR -= (np.sum(gradR*R,axis=1,keepdims=True))*R
        grade = grade - np.dot(grade,e)*e
        nrm = max(float(np.max(np.linalg.norm(gradR,axis=1))), float(np.linalg.norm(grade)))
        if not np.isfinite(nrm) or nrm<1e-12:
            break
        R = R - (lr/nrm)*gradR
        e = e - (lr/nrm)*grade
        R/=np.linalg.norm(R,axis=1,keepdims=True); e/=np.linalg.norm(e)
    X = _build_from_reps(R,e)
    return X

def _smooth_full(X, steps, k0=30.0, k1=220.0, lr0=0.03, lr1=0.004):
    X = X.copy()
    for s in range(steps):
        frac = s/max(1,steps-1)
        k = k0+(k1-k0)*frac
        lr = lr0+(lr1-lr0)*frac
        G = X @ X.T; np.fill_diagonal(G,-2.0)
        gmax=float(np.max(G))
        W = np.exp(np.clip(k*(G-gmax),-50,0.0))
        np.fill_diagonal(W,0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i',Gr,X)[:,None]
        Gt = Gr - dot*X
        nrm = float(np.max(np.linalg.norm(Gt,axis=1)))
        if not np.isfinite(nrm) or nrm<1e-12:
            break
        X = X - (lr/nrm)*Gt
        X/=np.linalg.norm(X,axis=1,keepdims=True)
    return X

def _lp_polish(X, budget, t_dead=None):
    t0=time.time()
    x = X/np.linalg.norm(X,axis=1,keepdims=True)
    n,d = x.shape
    try: best=_max_cos(x)
    except Exception: return x
    radius=1e-3
    iu0=np.triu_indices(n,1)
    ar=np.arange(d)
    it=0
    while True:
        if time.time()-t0>=budget: break
        if t_dead is not None and time.time()>=t_dead: break
        if radius<1e-14 or it>=25: break
        it+=1
        G=x@x.T; c=G[iu0]; m=float(np.max(c))
        sel=c>m-max(20.0*radius,1e-9)
        I=iu0[0][sel]; J=iu0[1][sel]; p=len(I)
        if p==0:
            radius*=0.5; continue
        if p>1500:
            csel=c[sel]; idx=np.argpartition(-csel,1500)[:1500]
            I=I[idx]; J=J[idx]; csel2=csel[idx]; p=1500
        else:
            csel2=c[sel]
        nv=n*d+1
        cols_mat=np.empty((p,2*d+1),dtype=np.int64)
        cols_mat[:,:d]=I[:,None]*d+ar
        cols_mat[:,d:2*d]=J[:,None]*d+ar
        cols_mat[:,2*d]=n*d
        rows=np.repeat(np.arange(p),2*d+1); cols=cols_mat.ravel()
        vals_mat=np.empty((p,2*d+1)); vals_mat[:,:d]=x[J]; vals_mat[:,d:2*d]=x[I]; vals_mat[:,2*d]=-1.0
        vals=vals_mat.ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-csel2,A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        try:
            y=x+res.x[:-1].reshape(n,d)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            v=_max_cos(y)
        except Exception:
            radius*=0.5; continue
        if v<best:
            x,best=y,v; radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time(); BUDGET=168.0
    n,d=181,9
    rng=np.random.default_rng(0)
    cands_best=[]  # list of (X,v)
    base=None; baseV=np.inf
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn); a=np.asarray(a,dtype=float)
            if a.shape==(n,d):
                a=a/np.linalg.norm(a,axis=1,keepdims=True)
                if np.all(np.isfinite(a)):
                    v=_max_cos(a)
                    cands_best.append((a.copy(),v))
                    if v<baseV: baseV=v; base=a.copy()
        except Exception:
            pass
    if base is None:
        base=rng.standard_normal((n,d)); base/=np.linalg.norm(base,axis=1,keepdims=True)
        baseV=_max_cos(base)
        cands_best.append((base.copy(),baseV))
    # Phase A: lattice pool selection
    try:
        P=_build_pool(np.random.default_rng(1))
        for st in (0, len(P)//3, len(P)//2):
            if time.time()-t0>90: break
            sel=_fps(P,n,st)
            sel,v=_swap_improve(P,sel,max_iter=60,rng=rng)
            X=P[sel]/np.linalg.norm(P[sel],axis=1,keepdims=True)
            # short smooth to make comparable
            X=_smooth_full(X,steps=120,k0=25.0,k1=140.0,lr0=0.03,lr1=0.006)
            v=_max_cos(X)
            cands_best.append((X.copy(),v))
    except Exception:
        pass
    # Phase B: cyclic-orbit symmetric reps (nr=20 -> 181 points)
    try:
        for sd in (10,11):
            if time.time()-t0>110: break
            X=_symm_opt(20,seed=sd,steps=260)
            v=_max_cos(X)
            cands_best.append((X.copy(),v))
    except Exception:
        pass
    # pick top 3 distinct candidates for full break-symmetry smoothing
    cands_best.sort(key=lambda t: t[1])
    # keep base anyway
    top=cands_best[:4]
    refined=[]
    for (X,v) in top:
        if time.time()-t0>BUDGET-35: break
        try:
            Y=_smooth_full(X,steps=260,k0=30.0,k1=240.0,lr0=0.025,lr1=0.004)
            refined.append((Y.copy(),_max_cos(Y)))
        except Exception:
            refined.append((X.copy(),v))
    if refined:
        refined.sort(key=lambda t:t[1])
        bestX,bestV=refined[0][0].copy(),float(refined[0][1])
    else:
        bestX,bestV=base.copy(),baseV
    # ensure we never lose base
    if baseV<bestV:
        bestX,bestV=base.copy(),baseV
    # final short LP polish on current best
    rem=BUDGET-(time.time()-t0)
    lp_time=max(4.0,min(22.0,rem-2.0))
    if rem>6.0:
        try:
            px=_lp_polish(bestX,budget=lp_time,t_dead=t0+BUDGET-1.5)
            pv=_max_cos(px)
            if np.all(np.isfinite(px)) and pv<bestV:
                bestX,bestV=px,pv
        except Exception:
            pass
    # fallback guard
    try:
        if _max_cos(bestX)>baseV:
            bestX=base.copy()
    except Exception:
        bestX=base.copy()
    bestX=bestX/np.linalg.norm(bestX,axis=1,keepdims=True)
    bestX=np.where(np.isfinite(bestX),bestX,0.0)
    # replace any bad rows
    nrms=np.linalg.norm(bestX,axis=1,keepdims=True)
    bad=(~np.isfinite(nrms[:,0]))| (nrms[:,0]<1e-12)
    if np.any(bad):
        rr=rng.standard_normal((int(np.sum(bad)),d))
        rr/=np.linalg.norm(rr,axis=1,keepdims=True)
        bestX[bad]=rr
        bestX/=np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX

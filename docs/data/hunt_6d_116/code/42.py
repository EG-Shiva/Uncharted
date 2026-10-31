import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contact_set(X, tol=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(n, 1)
    sel = G[iu] > m - tol
    I = iu[0][sel]; J = iu[1][sel]
    s = set((int(i)*1000+int(j)) for i, j in zip(I, J))
    return s, m

def _riesz_quench(X, s, steps, lr0=0.15, lr1=0.005, mx0=0.05, mx1=0.005):
    X = _norm(X.copy())
    n, d = X.shape
    for it in range(steps):
        f = it / max(steps-1, 1)
        lr = lr0*(1-f)+lr1*f
        mx = mx0*(1-f)+mx1*f
        D = X[:, None, :] - X[None, :, :]
        R2 = np.sum(D*D, axis=2)
        np.fill_diagonal(R2, 1.0)
        R = np.sqrt(np.maximum(R2, 1e-12))
        np.fill_diagonal(R, 1.0)
        W = R**(-(s+2.0))
        np.fill_diagonal(W, 0.0)
        # cap to avoid overflow
        np.clip(W, 0, 1e9, out=W)
        Grad = np.sum(W[:, :, None]*D, axis=1)
        dot = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - dot*X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            continue
        step = lr*Gt
        sn = gn*lr
        if sn > mx:
            step = step*(mx/sn)
        X = X - step
        X = _norm(X)
    return _norm(X)

def _softmax_quench(X, steps, b0=15.0, b1=400.0, lr0=2.0, lr1=0.25, mx=0.05):
    X = _norm(X.copy())
    n, d = X.shape
    for it in range(steps):
        f = it / max(steps-1, 1)
        beta = b0+(b1-b0)*f
        lr = lr0*(1-f)+lr1*f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(beta*(G-M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X)/S
        dot = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - dot*X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            continue
        step = lr*Gt
        if gn*lr > mx:
            step = step*(mx/(gn*lr))
        X = X - step
        X = _norm(X)
    return _norm(X)

def _deep_hole(fixed, rng, ntry=8000, refine=50):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    st = 0.06
    for _ in range(refine):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y)*y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - st*g
        ny = float(np.linalg.norm(y))
        if ny == 0 or not np.isfinite(ny):
            break
        y /= ny
        st *= 0.97
    return y/np.linalg.norm(y)

def _align_match(A, B):
    n, d = A.shape
    Rcur = np.eye(d)
    perm = np.arange(n)
    for _ in range(3):
        Br = B @ Rcur
        Dots = A @ Br.T
        try:
            _, col = linear_sum_assignment(-Dots)
        except Exception:
            break
        perm = np.asarray(col, dtype=int)
        Bp = B[perm]
        H = A.T @ Bp
        try:
            U, S, Vt = np.linalg.svd(H)
            Rnew = Vt.T @ U.T
            # ensure proper rotation (det +1); allow reflection for matching
            Rcur = Rnew
        except Exception:
            break
    Bal = B[perm] @ Rcur
    return Rcur, perm, Bal

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time()+0.4 > deadline:
            break
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0*radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj,A_ub=A_ub,b_ub=-cs,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best=v; bx=y.copy(); X=y
            radius=min(radius*1.5,0.05)
        else:
            if v <= best+1e-15:
                X=y
            radius*=0.5
    return bx,best

def solve():
    t0=time.time()
    HARD_END=t0+176.0
    PHASE1_END=t0+82.0
    BREED_END=t0+155.0
    rng=np.random.default_rng(777)
    try:
        X0=np.load("best.npy").astype(float)
    except Exception:
        try:
            X0=np.load("record.npy").astype(float)
        except Exception:
            X0=rng.standard_normal((116,6))
    if X0.shape!=(116,6):
        X0=rng.standard_normal((116,6))
    X0=_norm(X0)
    best_val=_maxcos(X0)
    best_X=X0.copy()
    pool=[(best_val,best_X.copy())]
    try:
        inc_set,_=_contact_set(best_X,tol=1e-7)
    except Exception:
        inc_set=set()
    n,d=116,6
    # Phase 1: diverse local optima
    nwant=9
    for k in range(nwant):
        if time.time()>PHASE1_END:
            break
        if time.time()+6.0>HARD_END:
            break
        Xr=rng.standard_normal((n,d)); Xr=_norm(Xr)
        s_choice=[1,4,12][k%3]
        # short Riesz quench
        try:
            Xr=_riesz_quench(Xr,s=s_choice,steps=300)
        except Exception:
            pass
        if time.time()>PHASE1_END:
            # still softmax briefly
            pass
        b0=12.0+ (k%3)*5.0
        try:
            Xr=_softmax_quench(Xr,steps=300,b0=b0,b1=350.0,lr0=2.0,lr1=0.25)
        except Exception:
            pass
        v=_maxcos(Xr)
        pool.append((v,Xr.copy()))
        if v<best_val and np.all(np.isfinite(Xr)):
            best_val=v; best_X=Xr.copy()
            try:
                inc_set,_=_contact_set(best_X,tol=1e-7)
            except Exception:
                pass
    # one heavily deleted/reinserted variant of incumbent
    if time.time()<PHASE1_END-4.0:
        try:
            base=best_X.copy()
            K=30
            del_idx=rng.choice(n,size=K,replace=False)
            keep=np.ones(n,bool); keep[del_idx]=False
            Xw=base.copy()
            fill_order=del_idx[rng.permutation(K)]
            for kk,pos in enumerate(fill_order):
                active=keep.copy(); active[fill_order[:kk]]=True
                fixed=Xw[active]
                y=_deep_hole(fixed,rng,ntry=8000,refine=50)
                Xw[pos]=y
            Xw=_softmax_quench(Xw,steps=350,b0=15,b1=400)
            v=_maxcos(Xw)
            pool.append((v,Xw.copy()))
            if v<best_val:
                best_val=v; best_X=Xw.copy()
                try:
                    inc_set,_=_contact_set(best_X,tol=1e-7)
                except Exception:
                    pass
        except Exception:
            pass
    # rank parents, keep diverse subset for breeding
    pool_sorted=sorted(pool,key=lambda t:t[0])
    # select up to 6 diverse parents
    sel_parents=[]
    sel_sets=[]
    for v,X in pool_sorted:
        try:
            s,_=_contact_set(X,tol=1e-7)
        except Exception:
            s=set()
        div=True
        for s2 in sel_sets:
            u=len(s|s2)
            j=(len(s&s2)/u) if u>0 else 0.0
            if j>0.92:
                div=False; break
        if div or len(sel_parents)<2:
            sel_parents.append((v,X))
            sel_sets.append(s)
        if len(sel_parents)>=6:
            break
    if len(sel_parents)<2:
        sel_parents=pool_sorted[:min(2,len(pool_sorted))]
    # breeding pairs
    # limit number of pairs to fit time
    pairs=[]
    L=len(sel_parents)
    for a in range(L):
        for b in range(a+1,L):
            pairs.append((a,b))
    # shuffle deterministic to mix
    order=rng.permutation(len(pairs))
    pairs=[pairs[i] for i in order]
    # at most 8 pairs
    pairs=pairs[:8]
    for pi,(a,b) in enumerate(pairs):
        if time.time()>BREED_END:
            break
        if time.time()+7.0>HARD_END:
            break
        _,A=sel_parents[a]; _,Bp0=sel_parents[b]
        vA=_maxcos(A); vB=_maxcos(Bp0)
        thr=max(vA,vB)
        try:
            _,_,Bal=_align_match(A,Bp0)
        except Exception:
            Bal=Bp0.copy()
        # two children types
        for ctype in [0,1]:
            if time.time()>BREED_END:
                break
            try:
                if ctype==0:
                    # matched halves random mask
                    mask=rng.random(n)<0.5
                    # enforce ~58
                    if np.sum(mask)==0 or np.sum(mask)==n:
                        mask[:58]=True; mask[58:]=False
                    C=np.empty_like(A)
                    C[mask]=A[mask]
                    C[~mask]=Bal[~mask]
                    # complement child on second iteration? we only do one per pair for halves; hemi next
                else:
                    u=rng.standard_normal(d); u/=np.linalg.norm(u)
                    mid=_norm(A+Bal)
                    proj=mid@u
                    med=float(np.median(proj))
                    takeA=proj>med
                    # adjust to exactly n (already n, one per index)
                    C=np.empty_like(A)
                    C[takeA]=A[takeA]
                    C[~takeA]=Bal[~takeA]
                C=_norm(C)
                # repair points above parents range
                for rep in range(8):
                    G=C@C.T; np.fill_diagonal(G,-2.0)
                    m=float(np.max(G))
                    if m<=thr+2e-9:
                        break
                    ij=int(np.argmax(G))
                    ii=ij//n; jj=ij%n
                    # choose worse of the two (higher mean dot)
                    mi=float(np.mean(G[ii])); mj=float(np.mean(G[jj]))
                    victim=jj if mj>=mi else ii
                    rest=np.ones(n,bool); rest[victim]=False
                    y=_deep_hole(C[rest],rng,ntry=6000,refine=40)
                    C[victim]=y
                # quench child briefly
                C=_softmax_quench(C,steps=250,b0=20,b1=350,lr0=1.5,lr1=0.2)
                v=_maxcos(C)
                # rank by max and signature: add if diverse or better
                try:
                    cs,_=_contact_set(C,tol=1e-7)
                except Exception:
                    cs=set()
                # diversity check vs pool
                dup=False
                for vv,XX in pool:
                    # quick jaccard via stored? recompute cheap: use cs vs contact of XX? approximate by value closeness
                    if abs(vv-v)<1e-9:
                        # need set compare; compute on fly for XX
                        try:
                            s2,_=_contact_set(XX,tol=1e-7)
                            u=len(cs|s2)
                            j=(len(cs&s2)/u) if u>0 else 0
                            if j>0.97:
                                dup=True; break
                        except Exception:
                            pass
                if not dup:
                    pool.append((v,C.copy()))
                if v<best_val and np.all(np.isfinite(C)):
                    best_val=v; best_X=C.copy()
                    try:
                        inc_set,_=_contact_set(best_X,tol=1e-7)
                    except Exception:
                        pass
                    # promising basin: LP polish immediately if time allows
                    if time.time()+4.0<BREED_END:
                        try:
                            Xp,vp=_lp_polish(C, min(time.time()+3.0,HARD_END-18.0))
                            pool.append((vp,Xp.copy()))
                            if vp<best_val:
                                best_val=vp; best_X=Xp.copy()
                        except Exception:
                            pass
            except Exception:
                continue
    # final LP polish best
    try:
        fd=min(HARD_END-1.0, time.time()+19.0)
        if time.time()<fd-1.0:
            Xf,vf=_lp_polish(best_X,fd)
            if vf<best_val and np.all(np.isfinite(Xf)):
                best_X=_norm(Xf); best_val=vf
    except Exception:
        pass
    best_X=_norm(best_X)
    return best_X

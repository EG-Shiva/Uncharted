import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = cnt; nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([ (I[:,None]*d+np.arange(d)), (J[:,None]*d+np.arange(d)), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y,axis=1,keepdims=True); yn[yn==0]=1.0
        y = y/yn
        v = _max_cos(y)
        if v < best:
            best=v; best_x=y.copy(); x=y
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
            if radius<1e-7:
                x=best_x.copy()
        if time.time()-start>=seconds:
            break
    return best_x

def _build_pool():
    d=13
    pool=[]
    # precomputed sign tables
    s2=[[a,b] for a in (-1,1) for b in (-1,1)]
    s3=[[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)]
    s4=[[a,b,c,e] for a in (-1,1) for b in (-1,1) for c in (-1,1) for e in (-1,1)]
    # F: (1,1,1)/sqrt3
    for combo in itertools.combinations(range(d),3):
        for s in s3:
            v=np.zeros(d); v[list(combo)]=s; pool.append(v/np.sqrt(3.0))
    # A: (1,1,1,1)/2
    for combo in itertools.combinations(range(d),4):
        for s in s4:
            v=np.zeros(d); v[list(combo)]=s; pool.append(v/2.0)
    # B: (2,1,1)/sqrt6  pos2 distinct
    sq6=np.sqrt(6.0)
    for p2 in range(d):
        rem=[i for i in range(d) if i!=p2]
        for combo in itertools.combinations(rem,2):
            for sgn2 in (-1,1):
                for s in s2:
                    v=np.zeros(d); v[p2]=2*sgn2; v[combo[0]]=s[0]; v[combo[1]]=s[1]
                    pool.append(v/sq6)
    # D: (3,1,1)/sqrt11
    sq11=np.sqrt(11.0)
    for p3 in range(d):
        rem=[i for i in range(d) if i!=p3]
        for combo in itertools.combinations(rem,2):
            for sgn3 in (-1,1):
                for s in s2:
                    v=np.zeros(d); v[p3]=3*sgn3; v[combo[0]]=s[0]; v[combo[1]]=s[1]
                    pool.append(v/sq11)
    # C: (2,1,1,1)/sqrt7
    sq7=np.sqrt(7.0)
    for p2 in range(d):
        rem=[i for i in range(d) if i!=p2]
        for combo in itertools.combinations(rem,3):
            for sgn2 in (-1,1):
                for s in s3:
                    v=np.zeros(d); v[p2]=2*sgn2
                    v[combo[0]]=s[0]; v[combo[1]]=s[1]; v[combo[2]]=s[2]
                    pool.append(v/sq7)
    # E: (3,2,1)/sqrt14 ordered
    sq14=np.sqrt(14.0)
    for perm in itertools.permutations(range(d),3):
        for s in s3:
            v=np.zeros(d); v[perm[0]]=3*s[0]; v[perm[1]]=2*s[1]; v[perm[2]]=1*s[2]
            pool.append(v/sq14)
    X=np.stack(pool,axis=0)
    # deduplicate exactly (different patterns never coincide, but be safe with rounding)
    # use void view for speed
    Xr=np.round(X,12)
    _,idx=np.unique(Xr,axis=0,return_index=True)
    X=X[np.sort(idx)]
    return X.astype(np.float64)

def _try_improve(S_idx, X, t):
    # 1-for-2 swaps, returns new list (may be larger by 1..)
    S_idx=list(S_idx)
    k=len(S_idx)
    if k==0:
        return S_idx
    S_mat=X[np.array(S_idx)]
    # D = X @ S_mat.T  (N,k)
    try:
        D=X@S_mat.T
    except Exception:
        return S_idx
    N=X.shape[0]
    is_in=np.zeros(N,dtype=bool); is_in[np.array(S_idx)]=True
    conf=(D>t+1e-12)
    cnt=conf.sum(axis=1)
    # only consider outside nodes
    cnt[is_in]=99
    # for each j try to find pair
    Sset=set(S_idx)
    for j in range(k):
        # candidates conflicting only with j
        mask=(cnt==1)&conf[:,j]
        cand=np.where(mask)[0]
        if cand.shape[0]<2:
            continue
        # limit to first 60 for speed
        if cand.shape[0]>60:
            cand=cand[:60]
        C=X[cand]
        G=C@C.T
        np.fill_diagonal(G,2.0)
        # find pair with dot<=t
        found=None
        m2=cand.shape[0]
        for a in range(m2):
            row=G[a]
            # find b>a with row<=t
            bl=np.where(row[a+1:]<=t+1e-12)[0]
            if bl.size>0:
                found=(int(cand[a]),int(cand[a+1+int(bl[0])]))
                break
        if found is not None:
            # replace S_idx[j] with found pair
            old=S_idx[j]
            S_idx[j]=found[0]
            S_idx.append(found[1])
            return S_idx
    return S_idx

def solve():
    t0=time.time()
    LIMIT=176.0
    rng=np.random.default_rng(0)
    n,d=123,13
    # fallback base
    base=None; base_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,dtype=float)
            if a.shape==(n,d):
                a=a/np.linalg.norm(a,axis=1,keepdims=True)
                if np.all(np.isfinite(a)):
                    v=_max_cos(a)
                    if v<base_val:
                        base_val=v; base=a.copy()
        except Exception:
            pass
    if base is None:
        base=rng.normal(size=(n,d)); base/=np.linalg.norm(base,axis=1,keepdims=True)
        base_val=_max_cos(base)
    # build pool
    try:
        X=_build_pool()
    except Exception:
        # fallback immediately
        rem=LIMIT-(time.time()-t0)-1.0
        if rem>2:
            try: base=_lp_polish(base,min(rem,15.0))
            except Exception: pass
        return base/np.linalg.norm(base,axis=1,keepdims=True)
    N=X.shape[0]
    thresholds=[0.270,0.268,0.266,0.264,0.262,0.260]
    found=None
    best_size=0
    # search budget: leave >=35s for polish
    search_deadline=t0+130.0
    # trials per threshold adaptive
    for t in thresholds:
        if time.time()>search_deadline: break
        if time.time()-t0>LIMIT-40: break
        trials=0
        max_trials=160
        while trials<max_trials and time.time()<search_deadline and time.time()-t0<LIMIT-40:
            trials+=1
            perm=rng.permutation(N)
            remaining=np.ones(N,dtype=bool)
            S=[]
            # iterate in random order
            for idx in perm:
                if not remaining[idx]:
                    continue
                S.append(int(idx))
                if len(S)>=123:
                    break
                s=X[idx]
                dots=X@s
                remaining[dots>t+1e-12]=False
                if len(S)%12==0:
                    # early abort if impossible
                    # count remaining quickly every 12 steps
                    rc=int(np.count_nonzero(remaining))
                    if len(S)+rc<123:
                        break
                    if rc==0:
                        break
                if time.time()>search_deadline:
                    break
            sz=len(S)
            if sz>best_size: best_size=sz
            if sz>=123:
                found=X[np.array(S[:123])].copy()
                break
            # swap augmentation when close
            if 112<=sz<123 and (trials%3==0):
                try:
                    S2=_try_improve(S,X,t)
                    if len(S2)>sz:
                        sz=len(S2); S=S2
                        if sz>best_size: best_size=sz
                        if sz>=123:
                            found=X[np.array(S[:123])].copy()
                            break
                        # try second round
                        if sz>=112:
                            S3=_try_improve(S,X,t)
                            if len(S3)>len(S):
                                S=S3; sz=len(S)
                                if sz>best_size: best_size=sz
                                if sz>=123:
                                    found=X[np.array(S[:123])].copy()
                                    break
                except Exception:
                    pass
        if found is not None:
            break
    if found is not None:
        rem=LIMIT-(time.time()-t0)-1.5
        pol=min(30.0,max(1.0,rem))
        try:
            P=_lp_polish(found,pol)
            v=_max_cos(P)
            # keep better of polished and base
            if v<base_val:
                return P/np.linalg.norm(P,axis=1,keepdims=True)
            else:
                # still return polished found if it beats record? prefer best
                # return the better
                return P/np.linalg.norm(P,axis=1,keepdims=True) if v<base_val+0.01 else base/np.linalg.norm(base,axis=1,keepdims=True)
        except Exception:
            return found/np.linalg.norm(found,axis=1,keepdims=True)
    # no 123-set: polished record fallback
    rem=LIMIT-(time.time()-t0)-1.0
    if rem>2:
        try:
            base=_lp_polish(base,min(rem,18.0))
        except Exception:
            pass
    base=base/np.linalg.norm(base,axis=1,keepdims=True)
    return base

import time
import itertools
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

# ---------- pools ----------
def _axis():
    P = []
    for i in range(7):
        v = np.zeros(7); v[i] = 1.0; P.append(v)
        v = np.zeros(7); v[i] = -1.0; P.append(v)
    return np.array(P)  # 14

_AXIS = _axis()

# precompute index lists for weight-2 and weight-3 patterns
_W2_PAIRS = [(i, j) for i in range(7) for j in range(7) if i != j]  # 42 ordered, x4 signs=168
_W2_UNORD = [(i, j) for i in range(7) for j in range(i+1, 7)]  # 21
_W3_COMBS = list(itertools.combinations(range(7), 3))  # 35
_SIGNS2 = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_SIGNS3 = list(itertools.product([1, -1], repeat=3))  # 8

def pool_A(a):
    # 14 + 168(a) type (1,a) + 280 fixed (1,1,1)
    n2 = np.sqrt(1.0 + a*a)
    P = [_AXIS]
    B = np.zeros((168, 7))
    k = 0
    for (i, j) in _W2_PAIRS:
        for (s1, s2) in _SIGNS2:
            # to avoid 4x duplicate we only use one sign combo per ordered pair?
            # ordered pair already distinguishes positions, need all 4 signs:
            pass
    # build explicitly 168
    B = np.zeros((len(_W2_PAIRS)*4, 7))
    k = 0
    for (i, j) in _W2_PAIRS:
        for (s1, s2) in _SIGNS2:
            v = np.zeros(7); v[i] = float(s1); v[j] = float(s2)*float(a)
            v /= n2
            B[k] = v; k += 1
    # Actually this is 42*4=168, good (each unordered appears twice with swapped roles, distinct)
    C = np.zeros((len(_W3_COMBS)*8, 7))
    k = 0
    s3 = np.sqrt(3.0)
    for (i, j, l) in _W3_COMBS:
        for (s1, s2, s3_) in _SIGNS3:
            v = np.zeros(7); v[i]=s1; v[j]=s2; v[l]=s3_
            v /= s3
            C[k]=v; k+=1
    return np.vstack([_AXIS, B, C])

def pool_B(b):
    # 14 + 84 fixed (1,1) + 280(b) type (1,1,b)
    P0 = _AXIS
    B = np.zeros((len(_W2_UNORD)*4, 7))
    k=0
    s2=np.sqrt(2.0)
    for (i,j) in _W2_UNORD:
        for (s1,s2_) in _SIGNS2:
            v=np.zeros(7); v[i]=s1; v[j]=s2_; v/=s2
            B[k]=v; k+=1
    n3=np.sqrt(2.0+b*b)
    C=np.zeros((len(_W3_COMBS)*8,7))
    k=0
    for (i,j,l) in _W3_COMBS:
        for (s1,s2_,s3_) in _SIGNS3:
            # put b on last index l to keep deterministic but varied
            v=np.zeros(7); v[i]=s1; v[j]=s2_; v[l]=s3_*b
            v/=n3
            C[k]=v; k+=1
    return np.vstack([P0,B,C])

def pool_C(a,b):
    n2=np.sqrt(1.0+a*a)
    B=np.zeros((len(_W2_PAIRS)*4,7))
    k=0
    for (i,j) in _W2_PAIRS:
        for (s1,s2_) in _SIGNS2:
            v=np.zeros(7); v[i]=s1; v[j]=s2_*a
            v/=n2
            B[k]=v; k+=1
    n3=np.sqrt(2.0+b*b)
    C=np.zeros((len(_W3_COMBS)*8,7))
    k=0
    for (i,j,l) in _W3_COMBS:
        for (s1,s2_,s3_) in _SIGNS3:
            v=np.zeros(7); v[i]=s1; v[j]=s2_; v[l]=s3_*b
            v/=n3
            C[k]=v; k+=1
    return np.vstack([_AXIS,B,C])

def pool_D(theta, a):
    # two-shell: shell1 fixed 126 pseudo-E7 (84 D7 + 42 consecutive triples),
    # shell2 197 points from (1,a) family subset, rotated by theta in plane 0-1 and 2-3
    # build shell1
    s2=np.sqrt(2.0)
    S1a=np.zeros((84,7))
    k=0
    for (i,j) in _W2_UNORD:
        for (s1,s2_) in _SIGNS2:
            v=np.zeros(7); v[i]=s1; v[j]=s2_; v/=s2
            S1a[k]=v; k+=1
    # 42 consecutive-triple all-equal-sign patterns
    S1b=np.zeros((42,7))
    k=0
    s3=np.sqrt(3.0)
    for i in range(7):
        j=(i+1)%7; l=(i+3)%7
        for rep in range(6):
            # 6 sign patterns with product +1 and variations to get 42=7*6
            pat=_SIGNS3[rep] if rep<8 else _SIGNS3[0]
            # use first 6 of 8
            if rep>=6: continue
            v=np.zeros(7); v[i]=pat[0]; v[j]=pat[1]; v[l]=pat[2]
            # only keep even parity to limit? just take first 6
            v/=s3
            S1b[k]=v; k+=1
    # S1b currently 7*6=42 filled
    S1=np.vstack([S1a,S1b])  # 126
    # shell2: take 197 points from weight-2/3 pool with param a
    full=pool_A(float(a))
    # deterministic subset: every other starting at 1
    # full is 462, need 197
    idx=np.linspace(0,len(full)-1,197).astype(int)
    S2=full[idx]
    # rotation of S2
    c=np.cos(float(theta)); s=np.sin(float(theta))
    R=np.eye(7)
    R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    c2=np.cos(2*float(theta)); s2_=np.sin(2*float(theta))
    R[2,2]=c2; R[2,3]=-s2_; R[3,2]=s2_; R[3,3]=c2
    S2r=S2@R.T
    X=np.vstack([S1,S2r])
    # X is 323 exactly (126+197)
    return _norm(X)

def _greedy_select(pool, need=323):
    P=len(pool)
    # start from index 0
    sel=[0]
    best=pool@pool[0]  # max dot to selected
    best[0]=np.inf
    # mask to avoid reselect: use inf
    for _ in range(1,need):
        idx=int(np.argmin(best))
        if best[idx]==np.inf:
            # fallback: pick first unused
            used=set(sel)
            for q in range(P):
                if q not in used:
                    idx=q; break
        sel.append(idx)
        d=pool@pool[idx]
        # max
        np.maximum(best,d,out=best)
        best[idx]=np.inf
        for s_ in sel:
            best[s_]=np.inf
    return np.array(sel,dtype=int)

def _build_select(pool_fn, params):
    pool=pool_fn(*params)
    pool=_norm(pool)
    sel=_greedy_select(pool,323)
    return pool[sel], sel, pool

def _score_fixed(pool_fn, sel_idx_pool_unused, params, cache_pool=None):
    # for fixed combinatorial type we rebuild pool and index same positions
    # sel here is indices into pool; pool size fixed per family so valid
    pool=pool_fn(*params)
    pool=_norm(pool)
    X=pool[sel_idx_pool_unused]
    return _max_cos(X), X

def _adam_polish(X, steps=200, tau0=0.05, tau1=0.008, lr=0.02, seed=0):
    X=_norm(np.array(X,float))
    n,d=X.shape
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1=0.9; b2=0.999; eps=1e-8
    for t in range(1,steps+1):
        tau=tau0+(tau1-tau0)*t/steps
        G=X@X.T
        np.fill_diagonal(G,-1e9)
        mx=float(G.max())
        # avoid overflow
        W=np.exp((G-mx)/tau)
        np.fill_diagonal(W,0.0)
        S=float(W.sum())
        if S==0: break
        grad=(2.0/S)*(W@X)
        nd=(grad*X).sum(axis=1,keepdims=True)
        gt=grad-nd*X
        m=b1*m+(1-b1)*gt
        vv=b2*v+(1-b2)*(gt*gt)
        v[:]=vv
        mhat=m/(1-b1**t); vhat=v/(1-b2**t)
        step=lr*mhat/(np.sqrt(vhat)+eps)
        # tangentialize step
        nd2=(step*X).sum(axis=1,keepdims=True)
        st=step-nd2*X
        # clip per-point norm
        nl=np.linalg.norm(st,axis=1,keepdims=True)+1e-12
        cap=0.05
        scale=np.minimum(1.0,cap/(nl+1e-12))
        st=st*scale
        X=X-st
        X=_norm(X)
    return X

def solve():
    t0=time.time()
    rng=np.random.default_rng(0)
    # baseline
    base=None; bv=np.inf
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(float)
            if a.shape==(323,7):
                a=_norm(a)
                vv=_max_cos(a)
                if vv<bv:
                    bv=vv; base=a.copy()
        except Exception:
            pass
    if base is None:
        base=_norm(rng.standard_normal(size=(323,7)))
        bv=_max_cos(base)

    cands=[]  # (score, X)
    # --- variant V0: pool_A(a) ---
    try:
        best=None
        for trial in range(28):
            if time.time()-t0>70: break
            a=float(rng.uniform(0.1,1.8))
            X,sel,_=_build_select(pool_A,(a,))
            sc=_max_cos(X)
            if best is None or sc<best[0]:
                best=(sc,a,sel,X)
        if best is not None:
            sc,a,sel,X=best
            # hill climb on a with fixed sel
            step=0.2
            for _ in range(14):
                if time.time()-t0>80: break
                improved=False
                for da in (step,-step):
                    na=float(np.clip(a+da,0.02,2.5))
                    s2,_=_score_fixed(pool_A,sel,(na,))
                    if s2<sc:
                        sc=s2; a=na; improved=True
                if not improved:
                    step*=0.5
                    if step<1e-4: break
            pool=pool_A(a); pool=_norm(pool); Xf=pool[sel]
            cands.append((sc,Xf,f"A a={a:.4f}"))
    except Exception:
        pass
    # --- variant V1: pool_B(b) ---
    try:
        best=None
        for trial in range(28):
            if time.time()-t0>95: break
            b=float(rng.uniform(0.1,1.8))
            X,sel,_=_build_select(pool_B,(b,))
            sc=_max_cos(X)
            if best is None or sc<best[0]:
                best=(sc,b,sel,X)
        if best is not None:
            sc,b,sel,X=best
            step=0.2
            for _ in range(14):
                if time.time()-t0>105: break
                improved=False
                for db in (step,-step):
                    nb=float(np.clip(b+db,0.02,2.5))
                    s2,_=_score_fixed(pool_B,sel,(nb,))
                    if s2<sc:
                        sc=s2; b=nb; improved=True
                if not improved:
                    step*=0.5
                    if step<1e-4: break
            pool=pool_B(b); pool=_norm(pool); Xf=pool[sel]
            cands.append((sc,Xf,f"B b={b:.4f}"))
    except Exception:
        pass
    # --- variant V2: pool_C(a,b) ---
    try:
        best=None
        for trial in range(30):
            if time.time()-t0>120: break
            a=float(rng.uniform(0.2,1.5)); b=float(rng.uniform(0.2,1.5))
            X,sel,_=_build_select(pool_C,(a,b))
            sc=_max_cos(X)
            if best is None or sc<best[0]:
                best=(sc,(a,b),sel,X)
        if best is not None:
            sc,(a,b),sel,X=best
            step=0.15
            for _ in range(12):
                if time.time()-t0>130: break
                improved=False
                for (da,db_) in ((step,0),(-step,0),(0,step),(0,-step)):
                    na=float(np.clip(a+da,0.02,2.5)); nb=float(np.clip(b+db_,0.02,2.5))
                    s2,_=_score_fixed(pool_C,sel,(na,nb))
                    if s2<sc:
                        sc=s2; a,b=na,nb; improved=True
                if not improved:
                    step*=0.5
                    if step<2e-4: break
            pool=pool_C(a,b); pool=_norm(pool); Xf=pool[sel]
            cands.append((sc,Xf,f"C a={a:.3f} b={b:.3f}"))
    except Exception:
        pass
    # --- variant D: two-shell rotation ---
    try:
        best=None
        for trial in range(24):
            if time.time()-t0>140: break
            th=float(rng.uniform(0,np.pi)); a=float(rng.uniform(0.3,1.5))
            X=_norm(pool_D(th,a))
            sc=_max_cos(X)
            if best is None or sc<best[0]:
                best=(sc,(th,a),X)
        if best is not None:
            sc,(th,a),X=best
            cands.append((sc,X,f"D th={th:.3f} a={a:.3f}"))
    except Exception:
        pass

    # polish best 3 structured with Adam (fast, no LP)
    cands=sorted(cands,key=lambda z:z[0])[:3]
    polished=[]
    for (sc,X,tag) in cands:
        if time.time()-t0>155: break
        try:
            Y=_adam_polish(X,steps=220,lr=0.02)
            s2=_max_cos(Y)
            # second finer stage
            if time.time()-t0<155:
                Y2=_adam_polish(Y,steps=120,tau0=0.02,tau1=0.005,lr=0.008)
                s3=_max_cos(Y2)
                if s3<s2:
                    Y,s2=Y2,s3
            polished.append((s2,Y))
        except Exception:
            polished.append((sc,X))
    best_struct=None
    if polished:
        polished.sort(key=lambda z:z[0])
        best_struct=polished[0]

    # light polish of baseline to try to squeeze epsilon (no heavy LP to avoid timeout)
    try:
        if time.time()-t0<165:
            Yb=_adam_polish(base,steps=150,lr=0.005,tau0=0.02,tau1=0.005)
            sb=_max_cos(Yb)
            if sb<bv:
                base,bv=Yb,sb
    except Exception:
        pass

    if best_struct is not None and best_struct[0]<bv:
        # structured wins (rare); return it (at most one short LP would go here, skipped for time)
        return _norm(best_struct[1])
    return _norm(base)

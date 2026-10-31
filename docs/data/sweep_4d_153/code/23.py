import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _expand(R, alist, blist, m):
    K = R.shape[0]
    N = K * m
    X = np.empty((N, 4), dtype=float)
    jj = np.arange(m)
    for k in range(K):
        r0, r1, r2, r3 = R[k]
        a = int(alist[k]); b = int(blist[k])
        th1 = (2.0 * np.pi * a / m) * jj
        th2 = (2.0 * np.pi * b / m) * jj
        c1 = np.cos(th1); s1 = np.sin(th1)
        c2 = np.cos(th2); s2 = np.sin(th2)
        X[k*m:(k+1)*m, 0] = c1*r0 - s1*r1
        X[k*m:(k+1)*m, 1] = s1*r0 + c1*r1
        X[k*m:(k+1)*m, 2] = c2*r2 - s2*r3
        X[k*m:(k+1)*m, 3] = s2*r2 + c2*r3
    return X

def _sym_descent(R0, alist, blist, m, steps, T0, T1, s0, s1, rng):
    R = R0.copy()
    K = R.shape[0]
    jj = np.arange(m)
    th1s = []
    th2s = []
    for k in range(K):
        a = int(alist[k]); b = int(blist[k])
        th1s.append((2.0*np.pi*a/m)*jj)
        th2s.append((2.0*np.pi*b/m)*jj)
    th1s = th1s
    th2s = th2s
    for it in range(steps):
        frac = it / max(1, steps-1)
        T = T0 * (T1/T0) ** frac
        s = s0 + (s1-s0)*frac
        X = _expand(R, alist, blist, m)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        GradF = (W @ X) / T
        Gr = np.zeros_like(R)
        for k in range(K):
            Gk = GradF[k*m:(k+1)*m]
            c1 = np.cos(th1s[k]); s1 = np.sin(th1s[k])
            c2 = np.cos(th2s[k]); s2 = np.sin(th2s[k])
            g0 = Gk[:,0]; g1 = Gk[:,1]; g2 = Gk[:,2]; g3 = Gk[:,3]
            y0 = c1*g0 + s1*g1
            y1 = -s1*g0 + c1*g1
            y2 = c2*g2 + s2*g3
            y3 = -s2*g2 + c2*g3
            Gr[k,0] = y0.sum(); Gr[k,1] = y1.sum()
            Gr[k,2] = y2.sum(); Gr[k,3] = y3.sum()
        dot = np.sum(Gr*R, axis=1, keepdims=True)
        Gt = Gr - dot*R
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True)+1e-12
        R = R - s*(Gt/nrm)
        R /= np.linalg.norm(R, axis=1, keepdims=True)+1e-30
    return R

def _smooth_descent(x, steps, T0, T1, s0, s1, rng, mu=0.0):
    x = x.copy()
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it/max(1,steps-1)
        T = T0*(T1/T0)**frac
        s = s0+(s1-s0)*frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = G.max()
        W = np.exp((G-cmax)/T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x)/T
        dot = np.sum(Grad*x, axis=1, keepdims=True)
        Gt = Grad - dot*x
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True)+1e-12
        direc = Gt/nrm
        if mu>0:
            vel = mu*vel+(1.0-mu)*direc
            step = vel
        else:
            step = direc
        x = x - s*step
        x /= np.linalg.norm(x, axis=1, keepdims=True)+1e-30
    return x

def _lp_polish(x, budget):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n,d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n,1)
    while time.time()-start < budget and radius>1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = c.max()
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p==0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
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
        if v<best:
            x,best,radius = y,v,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    DEADLINE=t0+172.0
    rng=np.random.default_rng(12345)
    # baseline
    base=None; base_val=np.inf
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,dtype=float)
            if a.shape==(153,4):
                a/=np.linalg.norm(a,axis=1,keepdims=True)+1e-30
                if np.all(np.isfinite(a)):
                    v=_max_cos(a)
                    if v<base_val:
                        base_val=v; base=a.copy()
        except Exception:
            pass
    if base is None:
        base=rng.normal(size=(153,4)); base/=np.linalg.norm(base,axis=1,keepdims=True); base_val=_max_cos(base)
    best=base.copy(); best_val=base_val
    # ---- build symmetry patterns ----
    patterns=[]
    for b in [0,1,2,3,4,5,6,7,8]:
        patterns.append((17,[1]*9,[b]*9))
    for b in [1,2,3,5,8,13]:
        patterns.append((51,[1]*3,[b]*3))
    for b in [1,2,4]:
        patterns.append((9,[1]*17,[b]*17))
    patterns.append((3,[1]*51,[1]*51))
    # mixed exponents for m=17 (different twist per orbit)
    rr=np.random.default_rng(999)
    for _ in range(2):
        bl=[int(rr.integers(0,17)) for _ in range(9)]
        patterns.append((17,[1]*9,bl))
    cands=[]  # (val, R, pattern_idx)
    T_PHASE1=t0+72.0
    # screening: few inits per pattern
    for pi,(m,al,bl) in enumerate(patterns):
        if time.time()>T_PHASE1:
            break
        K=len(al)
        ninit=6 if K<=17 else 4
        if m==3:
            ninit=3
        for ni in range(ninit):
            if time.time()>T_PHASE1:
                break
            R=rng.normal(size=(K,4)); R/=np.linalg.norm(R,axis=1,keepdims=True)+1e-30
            # cheap relax, vary T slightly deterministically
            T0=0.018+0.008*((pi+ni)%3)/2.0
            R2=_sym_descent(R,al,bl,m,steps=320,T0=T0,T1=0.0022,s0=0.06,s1=0.008,rng=rng)
            X=_expand(R2,al,bl,m)
            v=_max_cos(X)
            cands.append((v,R2.copy(),pi))
            # keep sorted, prune worst to save memory
            if len(cands)>60:
                cands.sort(key=lambda t:t[0])
                cands=cands[:40]
    cands.sort(key=lambda t:t[0])
    # fully relax best few symmetric
    relaxed=[]
    nrelax=min(5,len(cands))
    T_PHASE2=t0+88.0
    for i in range(nrelax):
        if time.time()>T_PHASE2:
            break
        v,R0,pi=cands[i]
        m,al,bl=patterns[pi]
        R1=_sym_descent(R0,al,bl,m,steps=900,T0=0.008,T1=0.0008,s0=0.03,s1=0.003,rng=rng)
        X=_expand(R1,al,bl,m)
        vv=_max_cos(X)
        relaxed.append((vv,X.copy(),R1.copy(),pi))
    relaxed.sort(key=lambda t:t[0])
    # symmetry breaking: gradual perturbation 0 -> 0.05-0.15 + smooth relax
    broken=[]
    # include top relaxed (up to 3)
    for bi in range(min(3,len(relaxed))):
        if time.time()>t0+138.0:
            break
        vv,X0,_,pi=relaxed[bi]
        for amp,seed in [(0.03,11+bi),(0.07,31+bi),(0.12,51+bi)]:
            if time.time()>t0+138.0:
                break
            r2=np.random.default_rng(2000+seed*100+bi)
            Y=X0+r2.normal(size=X0.shape)*amp
            Y/=np.linalg.norm(Y,axis=1,keepdims=True)+1e-30
            Y=_smooth_descent(Y,steps=550,T0=0.020,T1=0.0020,s0=0.014,s1=0.0011,rng=r2,mu=0.3)
            v=_max_cos(Y)
            broken.append((v,Y.copy()))
            if v<best_val-1e-12:
                best=Y.copy(); best_val=v
    # also try staged ramp on single best symmetric
    if relaxed and time.time()<t0+130.0:
        relaxed.sort(key=lambda t:t[0])
        Xc=relaxed[0][1].copy()
        r3=np.random.default_rng(777)
        for amp in [0.02,0.05,0.09]:
            if time.time()>t0+138.0:
                break
            Xc=Xc+r3.normal(size=Xc.shape)*amp*0.5
            Xc/=np.linalg.norm(Xc,axis=1,keepdims=True)+1e-30
            Xc=_smooth_descent(Xc,steps=400,T0=0.012,T1=0.0012,s0=0.010,s1=0.0010,rng=r3,mu=0.4)
        broken.append((_max_cos(Xc),Xc.copy()))
        if _max_cos(Xc)<best_val:
            best=Xc.copy(); best_val=_max_cos(Xc)
    # collect pool for final LP
    pool=[(best_val,best.copy())]
    for v,Y in broken:
        pool.append((v,Y.copy()))
    for v,X0,_,_ in relaxed:
        pool.append((v,X0.copy()))
    pool.sort(key=lambda t:t[0])
    pool=pool[:4]
    # final polish with remaining time
    remaining=DEADLINE-time.time()-2.0
    if remaining>5:
        per=min(6.0,remaining/(len(pool)+1))
        refined=[]
        for vv,xx in pool:
            if time.time()>DEADLINE-9:
                break
            z=_lp_polish(xx,budget=per*0.7)
            # small smooth squeeze
            z2=_smooth_descent(z,steps=220,T0=0.003,T1=0.0006,s0=0.002,s1=0.0004,rng=rng,mu=0.5)
            z3=_lp_polish(z2,budget=per*0.4)
            refined.append((_max_cos(z3),z3))
        # also polish baseline once more if time
        if time.time()<DEADLINE-8:
            zb=_lp_polish(base,budget=2.0)
            refined.append((_max_cos(zb),zb))
        if refined:
            refined.sort(key=lambda t:t[0])
            if refined[0][0]<best_val:
                best,best_val=refined[0][1],refined[0][0]
            else:
                # keep best
                pass
    else:
        # at least ensure best is polished a bit
        if time.time()<DEADLINE-3:
            best=_lp_polish(best,budget=max(0.5,DEADLINE-time.time()-2.0))
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    # safety: if somehow worse than base, return base polished
    try:
        vb=_max_cos(best)
        if not np.isfinite(vb) or vb>base_val+1e-9:
            best=base.copy()
    except Exception:
        best=base.copy()
    return best

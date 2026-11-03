import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    rowmax = G.max(axis=1)
    return cnt, rowmax, m, G

def _refine_hole(v, Xfr, steps=150, a=500.0, lr0=0.08):
    v = v / max(1e-12, np.linalg.norm(v))
    for t in range(steps):
        f = t / max(1, steps-1)
        lr = lr0 * (1.0 - 0.7*f)
        g = Xfr @ v
        m = float(g.max())
        w = np.exp(a*(g-m))
        s = float(w.sum())
        if not np.isfinite(s) or s<=0:
            break
        gr = (w @ Xfr) / s
        gt = gr - float(gr @ v)*v
        v = v - lr*gt
        n = np.linalg.norm(v)
        if n < 1e-12:
            break
        v = v / n
    return v

def _find_holes(Xfr, k, rng, n_samp=50000):
    d = Xfr.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ Xfr.T
    sc = S.max(axis=1)
    order = np.argsort(sc)
    picked = []
    for idx in order:
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > 0.60:
                ok = False
                break
        if ok:
            picked.append(v.copy())
            if len(picked) >= k:
                break
    while len(picked) < k:
        v = rng.randn(d); v/=np.linalg.norm(v)
        picked.append(v)
    H = np.array(picked)
    # local minimisation of max cosine to frozen code
    for i in range(k):
        H[i] = _refine_hole(H[i], Xfr, steps=120, a=400.0, lr0=0.08)
    # re-separate if collapsed
    for i in range(k):
        for j in range(i):
            if float(H[i] @ H[j]) > 0.85:
                nv = H[i] + rng.randn(d)*0.05
                nv/=np.linalg.norm(nv)
                H[i] = _refine_hole(nv, Xfr, steps=60, a=400.0, lr0=0.05)
    return _norm(H)

def _joint_lbfgs(H0, Xfr, temps=(100.0, 333.0, 1000.0), iters=150):
    k, d = H0.shape
    mf = Xfr.shape[0]
    # precompute triu indices for free-free
    iu = np.triu_indices(k, 1)
    ni = iu[0].size
    y = H0.copy().ravel()
    # ensure finite
    for a in temps:
        def fun(yflat):
            Y = yflat.reshape(k, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Xf = Y / nr
            G1 = Xf @ Xfr.T  # k x mf
            G2 = Xf @ Xf.T
            v1 = G1.ravel()
            v2 = G2[iu] if ni>0 else np.empty(0)
            if ni>0:
                Vall = np.concatenate([v1, v2])
            else:
                Vall = v1
            m0 = float(Vall.max())
            E = np.exp(a*(Vall - m0))
            S = float(E.sum())
            if not np.isfinite(S) or S<=0:
                return float(m0), np.zeros_like(yflat)
            lse = m0 + np.log(S)/a
            W = E / S
            W1 = W[:v1.size].reshape(k, mf)
            Gr = W1 @ Xfr  # k x d
            if ni>0:
                W2 = W[v1.size:]
                # accumulate free-free
                # Gr[i] += sum_{j} W2_{ij} Xf_j
                # use dense symmetric
                WM = np.zeros((k,k))
                WM[iu] = W2
                WM = WM + WM.T
                Gr = Gr + WM @ Xf
            # chain through normalisation
            dot = np.sum(Gr * Xf, axis=1, keepdims=True)
            Gt = Gr - dot * Xf
            Gy = Gt / nr
            return float(lse), Gy.ravel()
        try:
            res = minimize(fun, y, method='L-BFGS-B', jac=True,
                           options={'maxiter': iters, 'ftol':1e-20, 'gtol':1e-12})
            y = res.x
        except Exception:
            break
    Y = y.reshape(k,d)
    return _norm(Y)

def _smooth_all(X, steps=150, a0=120.0, a1=1200.0, lr0=0.008, lr1=0.0008):
    n, d = X.shape
    for t in range(steps):
        f = t/max(1,steps-1)
        a = a0*((a1/a0)**f)
        lr = lr0*((lr1/lr0)**f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(a*(G-m))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s<=0:
            break
        P = W/s
        Gr = P @ X
        dot = np.sum(Gr*X, axis=1, keepdims=True)
        Gt = Gr - dot*X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            continue
        X = X - lr*Gt/mx
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.array(X,float))
    n,d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    start = time.time()
    iu0 = np.triu_indices(n,1)
    I0 = iu0[0]; J0 = iu0[1]
    while time.time()-start < seconds:
        if deadline is not None and time.time()>deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[I0,J0]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0*radius, 1e-9)
        cnt = int(sel.sum())
        if cnt==0:
            radius*=0.5
            continue
        if cnt>3000:
            idx = np.argpartition(-cf,3000)[:3000]
            mask = np.zeros_like(sel); mask[idx]=True
            sel = sel & mask; cnt=3000
        I = I0[sel]; J=J0[sel]; c=cf[sel]; p=cnt
        nv = n*d+1
        rows = np.repeat(np.arange(p),2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([X[J],X[I],-np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method='highs')
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = _norm(X+res.x[:-1].reshape(n,d))
        v = _max_cos(y)
        if v < best-1e-15:
            X,best=y,v
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return X

def solve():
    t0=time.time()
    deadline=t0+170.0
    rng=np.random.RandomState(20261103)
    Xb=None; sb=1e9
    for fn in ["best.npy","record.npy"]:
        try:
            v=np.load(fn).astype(float)
            if v.shape==(162,8):
                v=_norm(v); s=_max_cos(v)
                if s<sb: sb=s; Xb=v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb=_norm(rng.randn(162,8)); sb=_max_cos(Xb)
    X_best=Xb.copy(); s_best=float(sb)
    n,d=162,8
    Ks=[3,4,6,8,10]
    it=0
    # time for exploration until ~150s from start
    while time.time() < t0+150.0:
        if time.time()>deadline-12:
            break
        it+=1
        k=Ks[(it-1)%len(Ks)]
        mode=(it-1)%3
        cnt,rowmax,m,G=_contacts(X_best,tol=1e-9)
        # identify sets
        rattlers=np.where(cnt==0)[0]
        if rattlers.size==0:
            rattler=int(np.argmin(cnt))
        else:
            rattler=int(rattlers[0])
        high=np.where(cnt>=20)[0]
        tied=np.where(rowmax>m-1e-12)[0]
        if tied.size==0:
            tied=np.where(rowmax>m-1e-9)[0]
        # choose delete set of size k including rattler
        if mode==0:
            order=np.argsort(-cnt)
            sel=[rattler]
            for idx in order:
                if len(sel)>=k: break
                if int(idx)==rattler: continue
                sel.append(int(idx))
            D=np.array(sel,dtype=int)
        elif mode==1:
            pool=np.union1d(high,tied) if high.size+tied.size>0 else np.arange(n)
            hub=int(rng.choice(pool))
            dist=G[hub].copy(); dist[hub]=-10; dist[rattler]=-10 if rattler!=hub else dist[rattler]
            nn=np.argsort(-dist)
            sel=[hub]
            for idx in nn:
                if len(sel)>=k-1: break
                if int(idx)==rattler or int(idx)==hub: continue
                sel.append(int(idx))
            # sel has hub + (k-2 or k-1); force rattler
            if rattler not in sel and rattler!=hub:
                sel.append(rattler)
            # adjust to exactly k
            sel=list(sel)
            # if too many, trim farthest (keep rattler+hub)
            while len(sel)>k:
                # remove last non-hub non-rattler
                for ii in range(len(sel)-1,-1,-1):
                    if sel[ii]!=hub and sel[ii]!=rattler:
                        sel.pop(ii); break
                else:
                    sel.pop()
            while len(sel)<k:
                cand=int(rng.randint(n))
                if cand not in sel:
                    sel.append(cand)
            D=np.array(sel[:k],dtype=int)
            if rattler not in D:
                D[-1]=rattler
        else:
            # random mixture among high/tied/low + rattler
            union=np.union1d(high,tied)
            low=np.where(cnt<=8)[0]
            pool=np.union1d(union,low)
            if pool.size<k:
                pool=np.arange(n)
            sel=[rattler]
            perm=rng.permutation(pool)
            for idx in perm:
                if len(sel)>=k: break
                if int(idx)==rattler: continue
                sel.append(int(idx))
            while len(sel)<k:
                cand=int(rng.randint(n))
                if cand not in sel:
                    sel.append(cand)
            D=np.array(sel,dtype=int)
        keep=np.ones(n,bool); keep[D]=False
        Xfr=X_best[keep].copy()
        # distinct holes + local minimisation
        try:
            H=_find_holes(Xfr,k,rng,n_samp=50000)
        except Exception:
            continue
        try:
            Hopt=_joint_lbfgs(H,Xfr,temps=(100.0,333.0,1000.0),iters=150)
        except Exception:
            continue
        Xcand=np.empty((n,d)); Xcand[keep]=Xfr; Xcand[D]=Hopt
        Xcand=_norm(Xcand)
        # short smooth refinement unfreezing all
        try:
            Xcand=_smooth_all(Xcand,steps=120,a0=150.0,a1=1500.0,lr0=0.006,lr1=0.0006)
        except Exception:
            pass
        # 3-5 s LP polish
        remain=deadline-time.time()-2.0
        if remain<2:
            break
        lp_sec=min(4.0 if k>=6 else 3.0, remain)
        # leave time for more iterations: cap
        # if close to end, polish best only; else polish candidate
        try:
            Xp=_lp_polish(Xcand,seconds=lp_sec,deadline=deadline-2.0)
        except Exception:
            continue
        sp=_max_cos(Xp)
        if sp < s_best:
            s_best=sp; X_best=Xp.copy()
    return _norm(X_best)

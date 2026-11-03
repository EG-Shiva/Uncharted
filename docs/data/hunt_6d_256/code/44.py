import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _signs64():
    S = np.ones((64, 6))
    for m in range(64):
        for l in range(6):
            if (m >> l) & 1:
                S[m, l] = -1.0
    return S

def _expand(S, signs):
    # S (ns,6) -> (ns*g,6)
    ns = S.shape[0]; g = signs.shape[0]
    X = np.empty((ns * g, 6))
    for p in range(ns):
        X[p*g:(p+1)*g, :] = S[p][None, :] * signs
    # S rows assumed unit -> X rows unit
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    X /= nrm
    return X

def _cluster_abs(A, k=4, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    # kmeans++ style
    centers = np.empty((k, 6))
    centers[0] = A[rng.integers(n)]
    for c in range(1, k):
        d2 = np.min(((A[:, None, :] - centers[None, :c, :]) ** 2).sum(axis=2), axis=1)
        s = d2.sum()
        if s <= 0:
            centers[c] = A[rng.integers(n)]
        else:
            centers[c] = A[rng.choice(n, p=d2 / s)]
    for _ in range(40):
        d2 = ((A[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab = np.argmin(d2, axis=1)
        for c in range(k):
            m = A[lab == c]
            if len(m) > 0:
                centers[c] = m.mean(axis=0)
            else:
                centers[c] = A[rng.integers(n)]
    # enforce positive and >=0.12 then normalize
    centers = np.abs(centers)
    centers = np.maximum(centers, 0.12)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # diversify: if two centers too close in abs, perturb
    for a in range(k):
        for b in range(a+1, k):
            if float(np.dot(centers[a], centers[b])) > 0.999:
                centers[b] = np.abs(centers[b] + rng.normal(scale=0.1, size=6))
                centers[b] = np.maximum(centers[b], 0.12)
                centers[b] /= np.linalg.norm(centers[b])
    return centers

def _sym_adam(S0, signs, steps, K0, K1, lr, deadline=1e18):
    S = S0.copy()
    S = np.abs(S); S = np.maximum(S, 0.12)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    ns = S.shape[0]
    m = np.zeros_like(S); v = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    Xb = _expand(S, signs)
    best_val = _max_cos(Xb)
    best_S = S.copy()
    for t in range(1, steps+1):
        if time.time() > deadline and t > 50:
            break
        K = float(np.exp(la0 + (la1-la0)*t/steps))
        # cap K to avoid overflow: shift by gmax
        X = _expand(S, signs)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # exp in stable way
        W = np.exp(K*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum()*0.5)
        if not np.isfinite(Ssum) or Ssum <= 0:
            continue
        gradX = (W @ X) / Ssum
        GXr = gradX.reshape(ns, signs.shape[0], 6)
        gradS = (GXr * signs[None, :, :]).sum(axis=1)
        dot = np.sum(gradS*S, axis=1, keepdims=True)
        gt = gradS - dot*S
        b1p *= b1; b2p *= b2
        m *= b1; m += (1-b1)*gt
        v *= b2; v += (1-b2)*(gt*gt)
        mh = m/(1-b1p); vh = v/(1-b2p)
        step = lr*mh/(np.sqrt(vh)+eps)
        sdot = np.sum(step*S, axis=1, keepdims=True)
        step = step - sdot*S
        S = S - step
        # enforce positive orthant + min magnitude 0.12
        S = np.abs(S)
        S = np.maximum(S, 0.12)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            Xc = _expand(S, signs)
            val = _max_cos(Xc)
            if val < best_val:
                best_val = val; best_S = S.copy()
    return best_S, _expand(best_S, signs), best_val

def _lp_sym_polish(S0, signs, deadline, r0=4e-3, cap=3000):
    S = S0.copy()
    S = np.abs(S); S = np.maximum(S, 0.12)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    ns, d = S.shape
    g = signs.shape[0]
    n = ns*g
    X = _expand(S, signs)
    best = _max_cos(X)
    best_S = S.copy()
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-12:
        X = _expand(S, signs)
        G = X @ X.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0*radius, 5e-5)
        sel = c > (mval - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            radius *= 0.5; continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]; pnum = cap
        # map flat idx -> (p,m)
        Ip = I // g; Jp = J // g
        # coefficients: for pair (I,J): dC = sum_l s_a[l] X_J[l] ds_Ip[l] + s_b[l] X_I[l] ds_Jp[l]
        # where s_a = signs[I%g], s_b = signs[J%g]
        sa = signs[I % g]  # (pnum,6)
        sb = signs[J % g]
        XI = X[I]; XJ = X[J]
        coeffI = sa * XJ  # (pnum,6)
        coeffJ = sb * XI
        # Build A_ub: vars ns*d + 1
        # rows repeat; handle case Ip==Jp (same prototype): sum coeffs
        rows = np.repeat(np.arange(pnum), 2*d+1)
        colI = (Ip[:, None]*d + np.arange(d)).ravel()
        colJ = (Jp[:, None]*d + np.arange(d)).ravel()
        # for t column
        cols = np.concatenate([colI, colJ, np.full(pnum, ns*d)])
        # need per-row ordering: first d -> I, next d -> J, last -> -1
        # rows construction must interleave; easier: build row by row via arrays
        # rebuild correctly:
        R = np.repeat(np.arange(pnum), 2*d+1)
        C = np.empty(pnum*(2*d+1), dtype=int)
        V = np.empty(pnum*(2*d+1))
        for k in range(pnum):
            base = k*(2*d+1)
            # I part
            C[base:base+d] = Ip[k]*d + np.arange(d)
            V[base:base+d] = coeffI[k]
            # J part: if same prototype, add to same cols (will duplicate entries -> sum by coo)
            C[base+d:base+2*d] = Jp[k]*d + np.arange(d)
            V[base+d:base+2*d] = coeffJ[k]
            C[base+2*d] = ns*d
            V[base+2*d] = -1.0
        try:
            A_ub = coo_matrix((V, (R, C)), shape=(pnum, ns*d+1)).tocsr()
            A_eq = coo_matrix((S.ravel(), (np.repeat(np.arange(ns), d), np.arange(ns*d))), shape=(ns, ns*d+1)).tocsr()
        except Exception:
            break
        obj = np.zeros(ns*d+1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(ns),
                          bounds=[(-radius, radius)]*(ns*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5; continue
        if res.status != 0:
            radius *= 0.5; continue
        ds = res.x[:-1].reshape(ns, d)
        S2 = S + ds
        S2 = np.abs(S2); S2 = np.maximum(S2, 0.12)
        S2 /= np.linalg.norm(S2, axis=1, keepdims=True)
        val = _max_cos(_expand(S2, signs))
        if val < best:
            S, best = S2, val; best_S = S2.copy()
            radius = min(radius*1.4, 0.02)
        else:
            radius *= 0.5
    return best_S, _expand(best_S, signs), best

def _adam_lse(x0, steps, K0, K1, lr, deadline=1e18):
    x = x0.copy()
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1,b2,eps = 0.9,0.999,1e-8
    b1p,b2p = 1.0,1.0
    la0,la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x); best_x = x.copy()
    for t in range(1, steps+1):
        if time.time() > deadline and t > 10:
            break
        K = float(np.exp(la0+(la1-la0)*t/steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum()*0.5)
        if not np.isfinite(Ssum) or Ssum <= 0:
            continue
        grad = (W @ x)/Ssum
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        b1p*=b1; b2p*=b2
        m*=b1; m+=(1-b1)*gt
        v*=b2; v+=(1-b2)*(gt*gt)
        mh=m/(1-b1p); vh=v/(1-b2p)
        step = lr*mh/(np.sqrt(vh)+eps)
        sdot=np.sum(step*x, axis=1, keepdims=True)
        step=step-sdot*x
        x=x-step
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        if t%100==0 or t==steps:
            val=_max_cos(x)
            if val<best_val:
                best_val=val; best_x=x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.0e-3, cap=3000):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n,d = x.shape
    best=_max_cos(x)
    radius=r0
    iuI,iuJ=np.triu_indices(n,1)
    while time.time()<deadline and radius>1e-14:
        G=x@x.T
        c=G[iuI,iuJ]
        mval=float(c.max())
        margin=max(20.0*radius,5e-5)
        sel=c>(mval-margin)
        I=iuI[sel]; J=iuJ[sel]; csel=c[sel]
        pnum=len(I)
        if pnum==0:
            radius*=0.5; continue
        if pnum>cap:
            part=np.argpartition(-csel,cap)[:cap]
            I=I[part]; J=J[part]; csel=csel[part]; pnum=cap
        rows=np.repeat(np.arange(pnum),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((pnum,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((pnum,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(pnum,n*d+1)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,n*d+1)).tocsr()
        except Exception:
            break
        obj=np.zeros(n*d+1); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        val=_max_cos(y)
        if val<best:
            x,best=y,val
            radius=min(radius*1.4,0.02)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time(); T_END=t0+172.0
    signs=_signs64()
    # load incumbent
    x_best=None; best=1e9; incumbent=None
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn); a=np.asarray(a,float)
            if a.shape==(256,6) and np.all(np.isfinite(a)):
                a/=np.linalg.norm(a,axis=1,keepdims=True)
                vv=_max_cos(a)
                if vv<best:
                    best=vv; x_best=a.copy()
                if incumbent is None or vv<_max_cos(incumbent):
                    incumbent=a.copy()
        except Exception:
            continue
    if incumbent is None:
        rng=np.random.default_rng(0)
        incumbent=rng.normal(size=(256,6))
        incumbent/=np.linalg.norm(incumbent,axis=1,keepdims=True)
        vv=_max_cos(incumbent)
        if vv<best:
            best=vv; x_best=incumbent.copy()
    # 4 prototypes from clusters of |x|
    Aabs=np.abs(incumbent)
    S0=_cluster_abs(Aabs,k=4,seed=1)
    # symmetry optimization high-K softmax 200->20000 on 24 coords
    try:
        S1,X1,v1=_sym_adam(S0,signs,steps=1400,K0=200.0,K1=20000.0,lr=0.008,deadline=min(T_END-90,time.time()+45))
        if v1<best:
            best=v1; x_best=X1.copy()
    except Exception:
        S1, X1, v1 = S0, _expand(S0,signs), _max_cos(_expand(S0,signs))
    # prototype-space LP
    try:
        S2,X2,v2=_lp_sym_polish(S1,signs,deadline=min(T_END-80,time.time()+12))
        if v2<best and np.all(np.isfinite(X2)):
            best=v2; x_best=X2.copy()
        else:
            S2=S1
    except Exception:
        S2=S1
    # staged symmetry breaking
    amps=[0.0,1e-4,1e-3,1e-2]
    Xsym=_expand(S2,signs)
    for si,amp in enumerate(amps):
        if time.time()>T_END-15:
            break
        rng=np.random.default_rng(9000+si)
        X0=Xsym.copy()
        if amp>0:
            z=rng.normal(size=X0.shape)
            z-=np.sum(z*X0,axis=1,keepdims=True)*X0
            X0=X0+amp*z
            X0/=np.linalg.norm(X0,axis=1,keepdims=True)
        # fully released Adam: moderate K then sharp K
        rem=T_END-12
        try:
            Xa,va=_adam_lse(X0,steps=350,K0=300.0,K1=6000.0,lr=0.006,deadline=rem)
        except Exception:
            continue
        try:
            Xp=_lp_polish(Xa,deadline=min(rem,time.time()+8))
            vp=_max_cos(Xp)
        except Exception:
            Xp,vp=Xa,va
        if vp<best and np.all(np.isfinite(Xp)):
            best=vp; x_best=Xp.copy()
        # feed forward best released as next sym base? keep Xsym as best symmetric + track best overall
        # also try short sharp from best stage
        if time.time()>T_END-12:
            break
    # final safety: include incumbent polish if nothing beat it
    if x_best is None:
        x_best=incumbent.copy()
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best

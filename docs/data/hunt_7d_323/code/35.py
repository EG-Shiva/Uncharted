import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_tables(q, speeds):
    k = np.arange(q)
    s1, s2, s3 = speeds
    p1 = 2*np.pi*k*s1/q
    p2 = 2*np.pi*k*s2/q
    p3 = 2*np.pi*k*s3/q
    return (np.cos(p1), np.sin(p1), np.cos(p2), np.sin(p2), np.cos(p3), np.sin(p3))

def _expand(U, C1,S1,C2,S2,C3,S3, q, pole):
    M = U.shape[0]
    N = M*q + (1 if pole else 0)
    X = np.empty((N,7), dtype=np.float64)
    # Mxq blocks
    u0 = U[:,0,None]; u1=U[:,1,None]; u2=U[:,2,None]; u3=U[:,3,None]; u4=U[:,4,None]; u5=U[:,5,None]; u6=U[:,6,None]
    b0 = u0*C1 - u1*S1
    b1 = u0*S1 + u1*C1
    b2 = u2*C2 - u3*S2
    b3 = u2*S2 + u3*C2
    b4 = u4*C3 - u5*S3
    b5 = u4*S3 + u5*C3
    # ravel row-major gives o*q+k order
    X[:M*q,0]=b0.ravel(); X[:M*q,1]=b1.ravel()
    X[:M*q,2]=b2.ravel(); X[:M*q,3]=b3.ravel()
    X[:M*q,4]=b4.ravel(); X[:M*q,5]=b5.ravel()
    X[:M*q,6]=np.repeat(U[:,6], q)
    if pole:
        X[M*q,:]=np.array([0,0,0,0,0,0,1.0])
    return X

def _adam_refine(Rinit, tables, q, M, pole, steps, lr, rng_seed_unused, deadline, tau0=0.02, tau1=0.0025):
    C1,S1,C2,S2,C3,S3 = tables
    R = Rinit.copy()
    m1 = np.zeros_like(R); m2 = np.zeros_like(R)
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    # track best exact
    Rn = np.linalg.norm(R,axis=1,keepdims=True); Rn[Rn==0]=1.0
    U = R/Rn
    X = _expand(U,C1,S1,C2,S2,C3,S3,q,pole)
    best_v = _max_cos(X)
    best_U = U.copy()
    N = X.shape[0]
    for step in range(1, steps+1):
        if time.time() > deadline:
            break
        Rn = np.linalg.norm(R,axis=1,keepdims=True); Rn[Rn==0]=1.0
        U = R/Rn
        X = _expand(U,C1,S1,C2,S2,C3,S3,q,pole)
        # annealed tau
        frac = step/steps
        tau = tau0*(1-frac)+tau1*frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        # smooth weights
        # E = exp((G-m0)/tau), diag 0
        D = (G - m0)/tau
        # clip to avoid overflow (max 0)
        np.clip(D, -50, 0, out=D)
        E = np.exp(D, out=D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()*0.5)
        if S <= 0 or not np.isfinite(S):
            break
        # grad_X = (E/S) @ X
        GX = (E / S) @ X  # N x 7
        # map to grad_U
        # reshape blocks
        if pole:
            GXb = GX[:M*q]
        else:
            GXb = GX
        G0 = GXb[:,0].reshape(M,q); G1 = GXb[:,1].reshape(M,q)
        G2 = GXb[:,2].reshape(M,q); G3 = GXb[:,3].reshape(M,q)
        G4 = GXb[:,4].reshape(M,q); G5 = GXb[:,5].reshape(M,q)
        G6 = GXb[:,6].reshape(M,q)
        gU = np.empty((M,7))
        gU[:,0]=np.sum(C1*G0 + S1*G1,axis=1)
        gU[:,1]=np.sum(-S1*G0 + C1*G1,axis=1)
        gU[:,2]=np.sum(C2*G2 + S2*G3,axis=1)
        gU[:,3]=np.sum(-S2*G2 + C2*G3,axis=1)
        gU[:,4]=np.sum(C3*G4 + S3*G5,axis=1)
        gU[:,5]=np.sum(-S3*G4 + C3*G5,axis=1)
        gU[:,6]=np.sum(G6,axis=1)
        # chain through normalization: gR = (gU - (gU.U)U)/|R|
        dot = np.sum(gU*U,axis=1,keepdims=True)
        rn = Rn[:,0]
        gR = (gU - dot*U)/rn[:,None]
        # Adam update (minimise)
        m1 = b1_*m1 + (1-b1_)*gR
        m2 = b2_*m2 + (1-b2_)*(gR*gR)
        h1 = m1/(1-b1_**step); h2 = m2/(1-b2_**step)
        R -= lr * h1/(np.sqrt(h2)+eps)
        # exact tracking every 25 steps and at end
        if step % 25 == 0 or step == steps:
            Rn2 = np.linalg.norm(R,axis=1,keepdims=True); Rn2[Rn2==0]=1.0
            U2 = R/Rn2
            X2 = _expand(U2,C1,S1,C2,S2,C3,S3,q,pole)
            v = _max_cos(X2)
            if v < best_v:
                best_v = v; best_U = U2.copy()
    return best_U, best_v

def _lp_polish(X, deadline, max_iters=60):
    X = X / np.linalg.norm(X,axis=1,keepdims=True)
    n,d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    iuI, iuJ = np.triu_indices(n,1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iters:
        it += 1
        if time.time() >= deadline:
            break
        G = X @ X.T
        c = G[iuI,iuJ]
        m = float(c.max())
        thresh = m - max(20.0*radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            part = np.argpartition(c[idx], -3000)[-3000:]
            sel = np.zeros_like(sel,dtype=bool)
            sel[idx[part]]=True
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt; nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.empty(p*(2*d+1),dtype=np.int64)
        vals = np.empty(p*(2*d+1),dtype=np.float64)
        Cb = cols.reshape(p,2*d+1); Vb = vals.reshape(p,2*d+1)
        ar = np.arange(d)
        Cb[:,:d]=I[:,None]*d+ar; Cb[:,d:2*d]=J[:,None]*d+ar; Cb[:,2*d]=n*d
        Vb[:,:d]=X[J]; Vb[:,d:2*d]=X[I]; Vb[:,2*d]=-1.0
        A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        remain = deadline - time.time()
        if remain < 0.2:
            break
        tl = min(2.0, max(0.3, remain-0.2))
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs",options={"time_limit":tl})
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = X + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y,axis=1,keepdims=True); yn[yn==0]=1.0
        y/=yn
        v = _max_cos(y)
        if v < best:
            X=y; best=v; radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return X, best

def solve():
    t0=time.time()
    DEAD=t0+165.0
    SYM_DEAD=t0+95.0
    rng=np.random.default_rng(0)
    best_X=None; best_v=np.inf
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape==(323,7):
                a/=np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(a)
                if v<best_v:
                    best_v=v; best_X=a.copy()
        except Exception:
            pass
    if best_X is None:
        a=rng.standard_normal((323,7)); a/=np.linalg.norm(a,axis=1,keepdims=True)
        best_X=a; best_v=_max_cos(a)
    # symmetric configs: (q,M,speeds,pole)
    configs=[
        (23,14,(1,5,11),True),
        (23,14,(1,2,7),True),
        (19,17,(1,4,9),False),
        (17,19,(1,4,8),False),
        (23,14,(1,7,11),True),
        (19,17,(1,5,11),False),
    ]
    sym_best_X=None; sym_best_v=np.inf; sym_best_cfg=None
    for (q,M,speeds,pole) in configs:
        if time.time()>SYM_DEAD or time.time()>DEAD-70:
            break
        tables=_build_tables(q,speeds)
        # restarts: fewer for larger M to save time
        n_rest=5 if M<=17 else 4
        steps=350
        for r in range(n_rest):
            if time.time()>SYM_DEAD or time.time()>DEAD-70:
                break
            Rinit=rng.standard_normal((M,7))
            Rn=np.linalg.norm(Rinit,axis=1,keepdims=True); Rn[Rn==0]=1.0
            Rinit/=Rn
            # small bias: keep x7 moderate
            # Adam
            Uopt, vopt=_adam_refine(Rinit,tables,q,M,pole,steps,0.08,None,min(DEAD-5,SYM_DEAD+10))
            if vopt<sym_best_v:
                sym_best_v=vopt
                sym_best_X=_expand(Uopt,*tables,q,pole)
                sym_best_cfg=(q,M,speeds,pole)
            # extra quick perturb of global sym best under same config (exploit)
            if sym_best_X is not None and r==n_rest-1:
                pass
    # LP polish: first symmetric best, then incumbent, within remaining time
    cur_best_X=best_X; cur_best_v=best_v
    if sym_best_X is not None:
        # only polish if somewhat competitive (within 0.03) to save time, else brief polish
        if sym_best_v < cur_best_v+0.04 and time.time()<DEAD-10:
            Xp,vp=_lp_polish(sym_best_X, min(DEAD-25, time.time()+35))
            if vp<cur_best_v:
                cur_best_X,cur_best_v=Xp,vp
            # keep sym polished even if worse? no
            # also store if sym polished beats sym
            sym_best_X, sym_best_v = Xp, vp
    # polish incumbent with remaining time
    if time.time()<DEAD-5:
        secs_left=DEAD-time.time()-1.0
        if secs_left>3:
            Xp,vp=_lp_polish(cur_best_X, time.time()+min(secs_left,40))
            if vp<cur_best_v:
                cur_best_X,cur_best_v=Xp,vp
    # fallback: if somehow cur worse than loaded best, return loaded best
    # ensure finite
    cur_best_X/=np.linalg.norm(cur_best_X,axis=1,keepdims=True)
    # final safety: if NaN, return best.npy
    if not np.all(np.isfinite(cur_best_X)):
        cur_best_X=best_X
    return cur_best_X

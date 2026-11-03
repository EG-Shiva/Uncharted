import time
import numpy as np

def _norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(X0, steps, beta0, beta1, lr0, lr1, mu, seed_off=0):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    bv = _mc(X)
    bX = X.copy()
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / float(max(1, steps-1))
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta*(G-m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum()*0.5)
        if S < 1e-30:
            continue
        Gr = (W @ X)/S
        Gr -= X*((Gr*X).sum(axis=1, keepdims=True))
        vel = mu*vel + (1.0-mu)*Gr
        X = _norm(X - lr*vel)
        if (s%100==99) or (s==steps-1):
            v = _mc(X)
            if v < bv:
                bv = v; bX = X.copy()
    return bX, bv

def _hard(X0, steps, lr0, lr1, mu, marg0, marg1):
    X = X0.copy()
    vel = np.zeros_like(X)
    bv = _mc(X)
    bX = X.copy()
    ll0, ll1 = np.log(lr0), np.log(lr1)
    lm0, lm1 = np.log(marg0), np.log(marg1)
    for s in range(steps):
        f = s / float(max(1, steps-1))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        marg = float(np.exp(lm0+(lm1-lm0)*f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m-marg)).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        Gr = (W @ X)/np.maximum(deg, 1.0)
        zm = (deg[:,0]==0)
        if np.any(zm):
            Gr[zm,:]=0.0
        Gr -= X*((Gr*X).sum(axis=1, keepdims=True))
        vel = mu*vel + (1.0-mu)*Gr
        X = _norm(X - lr*vel)
        if (s%150==149) or (s==steps-1):
            v = _mc(X)
            if v < bv:
                bv=v; bX=X.copy()
    return bX, bv

def _void_for(M, rng):
    # largest empty cap for set M (k,22): minmax dot
    d = M.shape[1]
    T = 1500
    C = _norm(rng.standard_normal((T,d)))
    S = M @ C.T
    mx = S.max(axis=0)
    bi = int(np.argmin(mx))
    v = C[bi].copy()
    # refine single vector with annealed softmax
    lr0, lr1 = 0.08, 0.005
    b0, b1 = 40.0, 600.0
    steps = 500
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best = float(mx[bi]); bv = v.copy()
    for s in range(steps):
        f = s/float(steps-1)
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        g = M @ v
        m = float(g.max())
        w = np.exp(beta*(g-m))
        sw = float(w.sum())
        if sw<=0 or not np.isfinite(sw):
            continue
        w/=sw
        gr = M.T @ w
        gr -= v*float(gr @ v)
        v = v - lr*gr
        vn = float(np.linalg.norm(v))
        if vn<1e-12:
            v = bv.copy(); continue
        v/=vn
        cur = float((M@v).max())
        if cur<best:
            best=cur; bv=v.copy()
    return bv

def solve():
    t0=time.time()
    DEAD=t0+168.0
    rng=np.random.default_rng(1)
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.asarray(np.load(fn),dtype=np.float64)
            if a.shape==(59,22) and np.all(np.isfinite(a)):
                a=_norm(a); cands.append(a)
        except Exception:
            pass
    if not cands:
        best=_norm(rng.standard_normal((59,22)))
    else:
        vs=[_mc(c) for c in cands]
        best=cands[int(np.argmin(vs))].copy()
    bval=_mc(best)
    def left(slack=8.0):
        return (time.time()<DEAD-slack)
    # quick polish of incumbent
    if left(12):
        X,v=_smooth(best,steps=2500,beta0=40.0,beta1=350.0,lr0=0.02,lr1=0.002,mu=0.9)
        if v<bval: bval=v; best=X.copy()
        X,v=_hard(best,steps=1200,lr0=0.004,lr1=0.0003,mu=0.85,marg0=0.005,marg1=0.0005)
        if v<bval: bval=v; best=X.copy()
    trial=0
    sigmas=[0.01,0.03,0.07,0.015,0.05,0.12,0.025,0.09]
    while left(22.0) and trial<80:
        mode=trial%10
        Xt=best.copy()
        if mode<=1:
            sg=sigmas[trial%len(sigmas)]
            Xt=_norm(best+sg*rng.standard_normal((59,22)))
        elif mode==2 or mode==3:
            # kill worst 1 -> void
            try:
                G=best@best.T; np.fill_diagonal(G,-2.0)
                rm=G.max(axis=1)
                idx=int(np.argmax(rm))
                M=np.delete(best,idx,axis=0)
                v=_void_for(M,rng)
                Xt=best.copy()
                Xt[idx]=v
                # tiny jitter rest
                Xt=_norm(Xt+0.005*rng.standard_normal((59,22)))
            except Exception:
                Xt=_norm(best+0.03*rng.standard_normal((59,22)))
        elif mode==4 or mode==5:
            # kill worst 2 -> voids sequentially
            try:
                G=best@best.T; np.fill_diagonal(G,-2.0)
                rm=G.max(axis=1)
                idxs=np.argsort(-rm)[:2]
                Xt=best.copy()
                for idx in idxs:
                    M=np.delete(Xt,idx,axis=0)
                    Xt[idx]=_void_for(M,rng)
                Xt=_norm(Xt+0.004*rng.standard_normal((59,22)))
            except Exception:
                Xt=_norm(best+0.05*rng.standard_normal((59,22)))
        elif mode==6:
            # kill highest degree hub
            try:
                G=best@best.T; np.fill_diagonal(G,-2.0)
                m=float(G.max())
                deg=(G>=(m-0.004)).sum(axis=1)
                idx=int(np.argmax(deg))
                M=np.delete(best,idx,axis=0)
                v=_void_for(M,rng)
                Xt=best.copy(); Xt[idx]=v
                Xt=_norm(Xt+0.006*rng.standard_normal((59,22)))
            except Exception:
                Xt=_norm(best+0.04*rng.standard_normal((59,22)))
        elif mode==7:
            # break max pair: push both apart + jitter
            try:
                G=best@best.T; np.fill_diagonal(G,-2.0)
                ij=np.unravel_index(int(G.argmax()),G.shape)
                Xt=best.copy()
                d=Xt[ij[0]]-Xt[ij[1]]
                d/=max(float(np.linalg.norm(d)),1e-12)
                Xt[ij[0]]=_norm((Xt[ij[0]]+0.05*d).reshape(1,-1))[0]
                Xt[ij[1]]=_norm((Xt[ij[1]]-0.05*d).reshape(1,-1))[0]
                Xt=_norm(Xt+0.01*rng.standard_normal((59,22)))
            except Exception:
                Xt=_norm(best+0.02*rng.standard_normal((59,22)))
        elif mode==8:
            sg=0.15
            Xt=_norm(best+sg*rng.standard_normal((59,22)))
        else:
            # random subspace shake of 5 points
            try:
                sel=rng.choice(59,size=5,replace=False)
                Xt=best.copy()
                Xt[sel]=_norm(rng.standard_normal((5,22)))
                Xt=_norm(Xt+0.008*rng.standard_normal((59,22)))
            except Exception:
                Xt=_norm(best+0.02*rng.standard_normal((59,22)))
        # short polish
        st=1800 if mode in (2,3,4,5,6) else 1400
        X,v=_smooth(Xt,steps=st,beta0=50.0,beta1=450.0,lr0=0.015,lr1=0.0015,mu=0.9)
        X2,v2=_hard(X,steps=700,lr0=0.003,lr1=0.0003,mu=0.85,marg0=0.004,marg1=0.0004)
        if v2<bval:
            bval=v2; best=X2.copy()
        elif v<bval:
            bval=v; best=X.copy()
        trial+=1
    # final deep polish
    if left(4.0):
        X,v=_smooth(best,steps=4000,beta0=80.0,beta1=900.0,lr0=0.008,lr1=0.0005,mu=0.9)
        if v<bval: bval=v; best=X.copy()
    if left(2.0):
        X,v=_hard(best,steps=2000,lr0=0.002,lr1=0.00015,mu=0.85,marg0=0.003,marg1=0.0003)
        if v<bval: bval=v; best=X.copy()
    best=_norm(best)
    return best

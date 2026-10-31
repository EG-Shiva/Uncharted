import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return m, cnt, G

def _smooth(X, steps, b0, b1, lr0, lr1, seed_off=0):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1_ = 0.9; b2_ = 0.999; eps = 1e-8
    p1 = 1.0; p2 = 1.0
    for s in range(1, steps+1):
        f = s / max(steps, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = G.max()
        W = np.exp(beta * (G - cm))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        Gr = W @ X
        pr = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - pr * X
        p1 *= b1_; p2 *= b2_
        m1 *= b1_; m1 += (1-b1_) * Gt
        vv *= b2_; vv += (1-b2_) * Gt * Gt
        mh = m1 / (1-p1); vh = vv / (1-p2)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _lp(X, seconds):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    bx = X.copy()
    rad = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-t0 < seconds and rad > 1e-14:
        if time.time()-t0 > seconds-0.1:
            break
        G = X @ X.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20*rad, 1e-9)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            break
        if len(idx) > 1500:
            o = np.argsort(c[idx])[-1500:]
            idx = idx[o]
        I, J = iu[0][idx], iu[1][idx]
        cs = c[idx]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            Aub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            Aeq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj,A_ub=Aub,b_ub=-cs,A_eq=Aeq,b_eq=np.zeros(n),bounds=[(-rad,rad)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            rad*=0.5; continue
        if res.status!=0:
            rad*=0.5; continue
        y = _norm(X + res.x[:-1].reshape(n,d))
        v = _max_cos(y)
        if v < best:
            best=v; bx=y.copy(); X=y; rad=min(rad*1.5,0.05)
        else:
            rad*=0.5
    return bx, best

def _hole(others, rng, nc=12000, beta=60, refine=250):
    cand = _norm(rng.standard_normal((nc, others.shape[1])))
    dots = cand @ others.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    y = cand[bi].copy()
    # refine best candidate by softmax descent on max-dot
    for _ in range(refine):
        d = others @ y
        mx = d.max()
        w = np.exp(beta*(d-mx)); w/=w.sum()
        g = w @ others
        g = g - (g@y)*y
        # adaptive step
        st = 0.05/(1.0+_ *0.01)
        y = _norm((y - st*g)[None,:])[0]
        # occasional small random nudge to escape
    # re-evaluate among refined + sampled top
    return y

def solve():
    t0=time.time()
    DEAD=178.0
    rng=np.random.default_rng(7)
    try:
        xb=np.load("best.npy").astype(float)
    except Exception:
        xb=None
    try:
        xr=np.load("record.npy").astype(float)
    except Exception:
        xr=None
    cands=[]
    if xb is not None and xb.shape==(116,6):
        cands.append(_norm(xb))
    if xr is not None and xr.shape==(116,6):
        cands.append(_norm(xr))
    if not cands:
        cands.append(_norm(rng.standard_normal((116,6))))
    vals=[_max_cos(c) for c in cands]
    bi=int(np.argmin(vals))
    Xbest=cands[bi].copy(); best=float(vals[bi])
    # short initial squeeze already? keep elitist
    cur=Xbest.copy(); cur_v=best
    # quick LP on best if time allows (5s) to ensure local opt
    if time.time()-t0 < 150:
        Xp,vp=_lp(Xbest, min(5.0, max(0.5, DEAD-(time.time()-t0)-140)))
        if vp<best:
            Xbest, best = Xp, vp
            cur, cur_v = Xp.copy(), vp
    it=0
    explore_end=t0+140.0
    while time.time() < explore_end and time.time()-t0 < DEAD-35:
        it+=1
        mode=it%6
        _,cnt,_=_contacts(cur, tol=2e-7)
        Y=cur.copy()
        if mode==0:
            sig=float(rng.uniform(0.02,0.06))
            Y=Y+sig*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y, steps=600, b0=30, b1=600, lr0=0.02, lr1=0.004)
        elif mode==1:
            order=np.argsort(-cnt)
            k=int(rng.integers(4,10))
            idx=order[:k]
            Y[idx]=Y[idx]+0.15*rng.standard_normal((k,6))
            Y=Y+0.008*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y, steps=800, b0=50, b1=1200, lr0=0.03, lr1=0.003)
        elif mode==2:
            G=cur@cur.T; np.fill_diagonal(G,-2.0)
            per_max=G.max(axis=1)
            worst_idx=np.argsort(-per_max)[:6]
            r=int(worst_idx[rng.integers(0,len(worst_idx))])
            others=np.delete(cur,r,axis=0)
            y=_hole(others,rng,nc=12000,beta=60,refine=200)
            Y[r]=y
            Y=Y+0.004*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y, steps=800, b0=40, b1=1000, lr0=0.02, lr1=0.003)
        elif mode==3:
            ratt=np.where(cnt<=1)[0]
            if len(ratt)==0:
                ratt=np.argsort(cnt)[:5]
            # replace 1-2 rattlers / random with holes
            for _ in range(min(2,len(ratt))):
                r=int(ratt[rng.integers(0,len(ratt))])
                others=np.delete(Y,r,axis=0)
                Y[r]=_hole(others,rng,nc=8000,beta=50,refine=120)
            Y=Y+0.006*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y, steps=500, b0=30, b1=800, lr0=0.02, lr1=0.004)
        elif mode==4:
            # double hole reinsert of two worst
            G=cur@cur.T; np.fill_diagonal(G,-2.0)
            per_max=G.max(axis=1)
            worst=np.argsort(-per_max)[:2]
            for r in worst:
                others=np.delete(Y,int(r),axis=0)
                Y[int(r)]=_hole(others,rng,nc=10000,beta=70,refine=150)
            Y=_norm(Y)
            Y=_smooth(Y, steps=900, b0=60, b1=1500, lr0=0.025, lr1=0.002)
        else:
            sig=float(rng.uniform(0.008,0.025))
            Y=Y+sig*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y, steps=700, b0=80, b1=2000, lr0=0.015, lr1=0.002)
        v=_max_cos(Y)
        # opportunistic LP if promising
        if v < best:
            # squeeze immediately
            remain=min(3.0, max(0.3, explore_end-time.time()))
            Y2,v2=_lp(Y, remain)
            if v2 < v:
                Y,v=Y2,v2
            if v < best:
                Xbest, best = Y.copy(), v
                cur, cur_v = Y.copy(), v
                continue
        # annealing acceptance for cur
        temp=0.0004*(1.0-(time.time()-t0)/explore_end) if time.time()<explore_end else 0.0
        if v < cur_v or (temp>0 and rng.random() < np.exp(-(v-cur_v)/max(temp,1e-9))):
            cur, cur_v = Y.copy(), v
        # occasional LP on cur to jump basin even if not best
        if it%12==0 and time.time()<explore_end-5:
            Y2,v2=_lp(cur, 1.2)
            if v2 < best:
                Xbest,best=Y2.copy(),v2
            if v2 < cur_v:
                cur,cur_v=Y2.copy(),v2
    # final long polish of best
    left=DEAD-(time.time()-t0)
    if left > 1.0:
        Xp,vp=_lp(Xbest, min(left-0.5, 35.0))
        if vp < best:
            Xbest,best=Xp,vp
    # extra smooth+polish cycle if time
    left=DEAD-(time.time()-t0)
    if left > 4.0:
        Y=_smooth(Xbest, steps=1200, b0=100, b1=2500, lr0=0.01, lr1=0.001)
        v=_max_cos(Y)
        if v < best+1e-6:
            Y2,v2=_lp(Y, left-0.5)
            if v2 < best:
                Xbest=X2copy=Y2 if False else Y2
    return _norm(Xbest)

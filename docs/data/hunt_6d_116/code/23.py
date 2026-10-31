import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(X, steps, b0, b1, lr0, lr1, t_dead=1e18):
    X = _norm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    p1 = 1.0; p2 = 1.0
    for s in range(1, steps+1):
        if s % 100 == 0 and time.time() > t_dead:
            break
        f = s / max(steps, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(G.max())
        W = np.exp(beta * (G - cm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        Gr = W @ X
        pr = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - pr * X
        p1 *= b1m; p2 *= b2m
        m *= b1m; m += (1-b1m)*Gt
        vv *= b2m; vv += (1-b2m)*Gt*Gt
        mh = m/(1-p1); vh = vv/(1-p2)
        X = X - lr * mh/(np.sqrt(vh)+eps)
        X = _norm(X)
    return X

def _hole(others, rng, nc=8000):
    d = others.shape[1]
    cand = rng.standard_normal((nc, d))
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True)+1e-15)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    y = cand[bi].copy()
    # few projected gradient steps to deepen hole
    st = 0.05
    for _ in range(60):
        dd = others @ y
        j = int(np.argmax(dd))
        g = others[j] - (others[j]@y)*y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - st*g
        y /= (np.linalg.norm(y)+1e-15)
        st *= 0.97
    return y

def _lp(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    rad = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-t0 < seconds and rad > 1e-14:
        if time.time()-t0 > seconds-0.15:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20*rad, 1e-9)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            break
        if len(idx) > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        I, J = iu[0][idx], iu[1][idx]
        cs = c[idx]
        p = len(idx)
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
        v = _maxcos(y)
        if v < best:
            best=v; bx=y.copy(); X=y; rad=min(rad*1.5,0.03)
        else:
            rad*=0.5
    return bx, best

def solve():
    t0=time.time()
    DEAD=t0+174.0
    EXPLORE=t0+140.0
    rng=np.random.default_rng(12345)
    def _clean(a):
        try:
            a=np.asarray(a,float)
        except Exception:
            return None
        if a.shape!=(116,6) or not np.all(np.isfinite(a)):
            return None
        n=np.linalg.norm(a,axis=1)
        if np.any(n<1e-12):
            return None
        return a/n[:,None]
    cands=[]
    try:
        cands.append(_clean(np.load("best.npy")))
    except Exception:
        pass
    try:
        cands.append(_clean(np.load("record.npy")))
    except Exception:
        pass
    cands=[c for c in cands if c is not None]
    if not cands:
        cands.append(_norm(rng.standard_normal((116,6))))
    vals=[_maxcos(c) for c in cands]
    bi=int(np.argmin(vals))
    Xbest=cands[bi].copy(); best=float(vals[bi])
    cur=Xbest.copy(); curv=best
    it=0
    while time.time()<EXPLORE and time.time()<DEAD-30:
        it+=1
        mode=it%6
        G=cur@cur.T; np.fill_diagonal(G,-2.0)
        per=G.max(axis=1)
        cnt=(G>float(G.max())-2e-7).sum(axis=1)
        Y=cur.copy()
        dl=min(DEAD-5.0, time.time()+5.0)
        if mode==0:
            Y=Y+rng.uniform(0.02,0.055)*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y,600,30,700,0.02,0.004,t_dead=dl)
        elif mode==1:
            order=np.argsort(-cnt)
            k=int(rng.integers(4,9))
            idx=order[:k]
            Y[idx]+=0.14*rng.standard_normal((k,6))
            Y+=0.008*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y,700,50,1200,0.028,0.003,t_dead=dl)
        elif mode==2:
            worst=np.argsort(-per)[:6]
            r=int(worst[rng.integers(0,6)])
            others=np.delete(cur,r,axis=0)
            Y[r]=_hole(others,rng,nc=9000)
            Y+=0.004*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y,700,40,1100,0.02,0.003,t_dead=dl)
        elif mode==3:
            ratt=np.where(cnt<=1)[0]
            if len(ratt)==0:
                ratt=np.argsort(cnt)[:6]
            nr=min(2,len(ratt))
            picks=rng.choice(len(ratt),size=nr,replace=False)
            for pp in picks:
                r=int(ratt[pp])
                others=np.delete(Y,r,axis=0)
                Y[r]=_hole(others,rng,nc=6000)
            Y+=0.006*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y,500,30,800,0.02,0.004,t_dead=dl)
        elif mode==4:
            worst2=np.argsort(-per)[:2]
            for r in worst2:
                rr=int(r)
                others=np.delete(Y,rr,axis=0)
                Y[rr]=_hole(others,rng,nc=8000)
            Y=_norm(Y)
            Y=_smooth(Y,800,60,1500,0.024,0.002,t_dead=dl)
        else:
            Y=Y+rng.uniform(0.008,0.025)*rng.standard_normal(Y.shape)
            Y=_norm(Y)
            Y=_smooth(Y,600,80,2000,0.014,0.002,t_dead=dl)
        v=_maxcos(Y)
        if v<best and time.time()<DEAD-25:
            Y2,v2=_lp(Y,min(2.0,max(0.4,EXPLORE-time.time())))
            if v2<v:
                Y,v=Y2,v2
            if v<best:
                Xbest,best=Y.copy(),v
                cur,curv=Y.copy(),v
                continue
        frac=max(0.0,(EXPLORE-time.time())/max(1.0,EXPLORE-t0))
        temp=0.0004*frac
        if v<curv or (temp>1e-12 and rng.random()<np.exp(-(v-curv)/max(temp,1e-12))):
            cur,curv=Y.copy(),v
        if it%15==0 and time.time()<EXPLORE-3 and time.time()<DEAD-30:
            Y2,v2=_lp(cur,0.9)
            if v2<best:
                Xbest,best=Y2.copy(),v2
            if v2<curv:
                cur,curv=Y2.copy(),v2
        if time.time()>EXPLORE:
            break
    left=DEAD-time.time()
    if left>2.0:
        Xp,vp=_lp(Xbest,min(left-0.6,22.0))
        if vp<best:
            Xbest,best=Xp,vp
    left=DEAD-time.time()
    if left>5.0:
        Y=_smooth(Xbest,900,100,2500,0.008,0.001,t_dead=DEAD-3.0)
        v=_maxcos(Y)
        Y2=None
        if v<best+2e-6:
            Y2,v2=_lp(Y,left-1.0)
            if v2<best:
                Xbest=X2copy=Y2
                best=v2
            elif v<best:
                Xbest=Y; best=v
    return _norm(Xbest)

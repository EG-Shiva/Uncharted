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
    return float(np.max(G))

def _tangent_grad(X, T):
    # gradient of softmax L=T*logsumexp(G/T) w.r.t X, projected to tangent
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(np.max(G))
    E = np.exp((G - m) / T)
    np.fill_diagonal(E, 0.0)
    s = float(np.sum(E))
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(X)
    E /= s
    Gr = E @ X
    proj = np.sum(Gr * X, axis=1, keepdims=True)
    return Gr - proj * X

def _expand_antipodal(Y):
    return np.vstack([Y, -Y])

def _expand_orbit(R, w):
    # R (9,6) unit rows, w (3,) unit
    # fixed equal vector
    v0 = np.full(6, 1.0/np.sqrt(6))
    # period-3 base
    b = np.array([w[0], w[1], w[2], w[0], w[1], w[2]], dtype=np.float64) / np.sqrt(2.0) * np.linalg.norm(w) / (np.linalg.norm(w)+1e-15)
    # w is unit, so b = tile(w)/sqrt2 ; ensure unit
    # recompute exactly:
    wn = w / (np.linalg.norm(w)+1e-15)
    b = np.array([wn[0], wn[1], wn[2], wn[0], wn[1], wn[2]]) / np.sqrt(2.0)
    # b has norm 1
    parts = []
    for i in range(R.shape[0]):
        r = R[i]
        for s in range(6):
            parts.append(np.roll(r, s))
        for s in range(6):
            parts.append(-np.roll(r, s))
    # special 6
    for s in range(3):
        parts.append(np.roll(b, s))
    for s in range(3):
        parts.append(-np.roll(b, s))
    parts.append(v0)
    parts.append(-v0)
    return np.vstack(parts)

def _orbit_pullback(Gt, Rshape=(9,6)):
    # Gt (116,6) tangent grad w.r.t expanded X
    # returns grad_R (9,6), grad_bvec (6,)
    nR = Rshape[0]
    grad_R = np.zeros(Rshape)
    for i in range(nR):
        base = i*12
        g = np.zeros(6)
        for s in range(6):
            gp = Gt[base+s]
            g += np.roll(gp, -s)
        for s in range(6):
            gn = Gt[base+6+s]
            g += -np.roll(gn, -s)
        grad_R[i] = g
    # special bvec
    gb = np.zeros(6)
    for s in range(3):
        gp = Gt[nR*12+s]
        gb += np.roll(gp, -s)
    for s in range(3):
        gn = Gt[nR*12+3+s]
        gb += -np.roll(gn, -s)
    return grad_R, gb

def _adam_descent_antipodal(Y0, steps, T0, T1, lr0, lr1):
    Y = _norm(Y0.copy())
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, steps+1):
        f = it/float(steps)
        T = T0*(T1/T0)**f
        lr = lr0+(lr1-lr0)*f
        X = _expand_antipodal(Y)
        GtX = _tangent_grad(X, T)
        Gt = GtX[:Y.shape[0]] - GtX[Y.shape[0]:]
        # project to tangent of Y
        proj = np.sum(Gt*Y, axis=1, keepdims=True)
        Gt = Gt - proj*Y
        b1p*=b1; b2p*=b2
        m*=b1; m+=(1.0-b1)*Gt
        v*=b2; v+=(1.0-b2)*Gt*Gt
        mh=m/(1.0-b1p); vh=v/(1.0-b2p)
        step = lr*mh/(np.sqrt(vh)+eps)
        # project step tangentially (remove radial)
        step = step - np.sum(step*Y,axis=1,keepdims=True)*Y
        Y = Y - step
        Y = _norm(Y)
    return Y

def _adam_descent_orbit(R0, w0, steps, T0, T1, lr0, lr1):
    R = _norm(R0.copy())
    w = w0.copy(); w/= (np.linalg.norm(w)+1e-15)
    mR=np.zeros_like(R); vR=np.zeros_like(R)
    mw=np.zeros_like(w); vw=np.zeros_like(w)
    b1,b2,eps=0.9,0.999,1e-8
    b1p,b2p=1.0,1.0
    for it in range(1, steps+1):
        f=it/float(steps)
        T=T0*(T1/T0)**f
        lr=lr0+(lr1-lr0)*f
        X=_expand_orbit(R,w)
        GtX=_tangent_grad(X,T)
        gR, gb=_orbit_pullback(GtX, R.shape)
        # map gb (6) to gw (3)
        gw=np.array([gb[0]+gb[3], gb[1]+gb[4], gb[2]+gb[5]])/np.sqrt(2.0)
        # tangent projections
        gR=gR-np.sum(gR*R,axis=1,keepdims=True)*R
        gw=gw-np.dot(gw,w)*w
        b1p*=b1; b2p*=b2
        mR*=b1; mR+=(1-b1)*gR
        vR*=b2; vR+=(1-b2)*gR*gR
        mw*=b1; mw+=(1-b1)*gw
        vw*=b2; vw+=(1-b2)*gw*gw
        mhR=mR/(1-b1p); vhR=vR/(1-b2p)
        mhw=mw/(1-b1p); vhw=vw/(1-b2p)
        sR=lr*mhR/(np.sqrt(vhR)+eps)
        sw=lr*mhw/(np.sqrt(vhw)+eps)
        sR=sR-np.sum(sR*R,axis=1,keepdims=True)*R
        sw=sw-np.dot(sw,w)*w
        R=R-sR; R=_norm(R)
        w=w-sw; w/=(np.linalg.norm(w)+1e-15)
    return R,w

def _adam_descent_free(X0, steps, T0, T1, lr0, lr1):
    X=_norm(X0.copy())
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1,b2,eps=0.9,0.999,1e-8
    b1p,b2p=1.0,1.0
    for it in range(1, steps+1):
        f=it/float(steps)
        T=T0*(T1/T0)**f
        lr=lr0+(lr1-lr0)*f
        Gt=_tangent_grad(X,T)
        b1p*=b1; b2p*=b2
        m*=b1; m+=(1-b1)*Gt
        v*=b2; v+=(1-b2)*Gt*Gt
        mh=m/(1-b1p); vh=v/(1-b2p)
        step=lr*mh/(np.sqrt(vh)+eps)
        step=step-np.sum(step*X,axis=1,keepdims=True)*X
        X=X-step
        X=_norm(X)
    return X

def _lp_polish(X, deadline):
    X=_norm(X.copy())
    n,d=X.shape
    best=_maxcos(X)
    bx=X.copy()
    radius=2.0e-3
    iu=np.triu_indices(n,1)
    while time.time()<deadline and radius>1e-14:
        if time.time()>deadline-0.4:
            break
        G=X@X.T
        c=G[iu]
        m=float(c.max())
        thr=m-max(20.0*radius,1e-9)
        idx=np.where(c>thr)[0]
        if idx.size==0:
            break
        if idx.size>1400:
            o=np.argsort(c[idx])[-1400:]
            idx=idx[o]
        sel=np.zeros_like(c,dtype=bool); sel[idx]=True
        I=iu[0][sel]; J=iu[1][sel]; cs=c[sel]
        p=len(I); nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],axis=1).ravel()
        vals=np.concatenate([X[J],X[I],-np.ones((p,1))],axis=1).ravel()
        try:
            from scipy.sparse import coo_matrix as _coo
            A_ub=_coo((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=_coo((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-cs,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=X+res.x[:-1].reshape(n,d)
        y=_norm(y)
        vv=_maxcos(y)
        if vv<best:
            best=vv; bx=y.copy(); X=y
            radius=min(radius*1.5,0.05)
        else:
            if vv<=best+1e-15:
                X=y
            radius*=0.5
    return bx,best

def solve():
    t0=time.time()
    HARD=t0+171.0
    rng=np.random.default_rng(0)
    bestX=None; bestV=1e9
    cand_pool=[]  # symmetric optima expanded

    # Phase A: antipodal 58 reps
    tA_end=t0+38.0
    bestAY=None; bestAV=1e9
    secondAY=None; secondAV=1e9
    a_iter=0
    while time.time()<tA_end-2.0:
        a_iter+=1
        Y0=rng.normal(size=(58,6))
        Y0=_norm(Y0)
        # vary steps with iteration
        steps=1600 if (a_iter%2==0) else 2200
        # beta 50->1500 => T 0.02->0.00067; use T0 0.02 T1 0.001
        Y1=_adam_descent_antipodal(Y0, steps=steps, T0=0.02, T1=0.0009, lr0=0.015, lr1=0.002)
        X1=_expand_antipodal(Y1)
        v=_maxcos(X1)
        if v<bestAV:
            secondAV,secondAY=bestAV,bestAY
            bestAV,bestAY=v,Y1.copy()
        elif v<secondAV:
            secondAV,secondAY=v,Y1.copy()
        if time.time()>tA_end-2.0:
            break
    if bestAY is not None:
        cand_pool.append((_expand_antipodal(bestAY),bestAV))
        if secondAY is not None and secondAV<1e8:
            cand_pool.append((_expand_antipodal(secondAY),secondAV))

    # Phase B: orbit C6 x {+-1}, 9 generic + period3 + fixed
    tB_end=t0+68.0
    bestR=None; bestw=None; bestBV=1e9
    secondR=None; secondw=None; secondBV=1e9
    b_iter=0
    while time.time()<tB_end-2.0:
        b_iter+=1
        R0=_norm(rng.normal(size=(9,6)))
        w0=rng.normal(size=(3,)); w0/=np.linalg.norm(w0)
        steps=1600 if (b_iter%2==0) else 2200
        R1,w1=_adam_descent_orbit(R0,w0,steps=steps,T0=0.02,T1=0.0009,lr0=0.015,lr1=0.002)
        X1=_expand_orbit(R1,w1)
        v=_maxcos(X1)
        if v<bestBV:
            secondBV,secondR,secondw=bestBV,bestR,bestw
            bestBV,bestR,bestw=v,R1.copy(),w1.copy()
        elif v<secondBV:
            secondBV,secondR,secondw=v,R1.copy(),w1.copy()
        if time.time()>tB_end-2.0:
            break
    if bestR is not None:
        cand_pool.append((_expand_orbit(bestR,bestw),bestBV))
        if secondR is not None and secondBV<1e8:
            cand_pool.append((_expand_orbit(secondR,secondw),secondBV))

    # contact variance check implicitly: keep all; Phase C releases
    # Phase C: release + perturb + unconstrained quench
    # build released candidates
    released=[]
    sigmas=[0.002,0.005,0.01,0.003]
    for ci,(Xc,vc) in enumerate(cand_pool):
        for si,s in enumerate(sigmas[:2] if ci>=2 else sigmas):
            if time.time()>t0+150.0:
                break
            Xn=Xc+rng.normal(scale=s,size=Xc.shape)
            Xn=_norm(Xn)
            # annealed free quench, shorter for small sigma
            steps=900 if s<0.004 else 1300
            Xq=_adam_descent_free(Xn,steps=steps,T0=0.015,T1=0.0015,lr0=0.012,lr1=0.002)
            vq=_maxcos(Xq)
            released.append((Xq,vq))
            if vq<bestV:
                bestV, bestX=vq,Xq.copy()
        if time.time()>t0+150.0:
            break
    # if nothing yet, use best symmetric directly
    for (Xc,vc) in cand_pool:
        if bestX is None or vc<bestV:
            bestV, bestX=vc,Xc.copy()

    # sort released by value, LP-polish top 3
    released.sort(key=lambda t:t[1])
    # also consider bestX
    lp_cands=[]
    for k in range(min(3,len(released))):
        lp_cands.append(released[k][0])
    # deduplicate: ensure bestX included
    if bestX is not None:
        # add if not already close
        lp_cands.append(bestX)

    # LP polish best two/three released within budget
    # keep unique top 3 by value among lp_cands
    scored=[(X,_maxcos(X)) for X in lp_cands]
    scored.sort(key=lambda t:t[1])
    seen=[]
    for X,v in scored:
        seen.append(X)
        if len(seen)>=3:
            break
    for X in seen:
        if time.time()>HARD-8.0:
            break
        # give each ~6s or remaining/...
        budget=min(7.0, (HARD-6.0-time.time())/ (len(seen)) )
        if budget<1.0:
            budget=1.0
        deadline=min(time.time()+budget, HARD-5.0)
        Xp,vp=_lp_polish(X,deadline)
        if vp<bestV:
            bestV, bestX=vp,Xp.copy()

    # final comparison with incumbent files (ignore until now)
    try:
        xb=np.load("best.npy").astype(np.float64)
        xb=_norm(xb)
        if xb.shape==(116,6) and np.all(np.isfinite(xb)):
            vb=_maxcos(xb)
            if vb<bestV:
                bestV, bestX=vb,xb
    except Exception:
        pass
    try:
        xr=np.load("record.npy").astype(np.float64)
        xr=_norm(xr)
        if xr.shape==(116,6) and np.all(np.isfinite(xr)):
            vr=_maxcos(xr)
            if vr<bestV:
                bestV, bestX=vr,xr
    except Exception:
        pass

    # final safety polish if time remains and best is from file? quick LP
    if time.time()<HARD-4.0 and bestX is not None:
        # tiny extra polish 3s
        Xp,vp=_lp_polish(bestX, min(time.time()+3.0,HARD-2.0))
        if vp<=bestV:
            bestX,bestV=Xp,vp

    bestX=_norm(bestX)
    return bestX

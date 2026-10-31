import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

_TRIU = None
def _get_triu(n):
    global _TRIU
    if _TRIU is None or _TRIU[0].shape[0] != n*(n-1)//2:
        _TRIU = np.triu_indices(n, 1)
    return _TRIU

def lp_polish(x, t_end, r0=0.003):
    x = _norm_rows(x.copy())
    n, d = x.shape
    iu = _get_triu(n)
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = float(r0)
    it = 0
    while time.time() < t_end and radius > 1e-14 and it < 80:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0*radius, 1e-9)
        sel = c > thr
        p_all = int(np.sum(sel))
        if p_all == 0:
            radius *= 0.5
            continue
        idx_all = np.where(sel)[0]
        if p_all > 2500:
            part = np.argpartition(c[idx_all], -2500)[-2500:]
            idx = idx_all[part]
        else:
            idx = idx_all
        p = idx.shape[0]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        try:
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,n*d+1)).tocsr()
            obj = np.zeros(n*d+1); obj[-1]=1.0
            bnds = [(-radius,radius)]*(n*d)+[(None,None)]
            res = linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),bounds=bnds,method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            dx = res.x[:-1].reshape(n,d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm_rows(x+dx)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, steps, beta, lr):
    x = _norm_rows(x.copy())
    for _ in range(int(steps)):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(beta*(G-m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        proj = np.sum(grad*x,axis=1,keepdims=True)
        gt = grad - proj*x
        step = lr*gt
        sn = np.linalg.norm(step,axis=1,keepdims=True)
        over = sn > 0.02
        if np.any(over):
            scale = np.ones_like(sn)
            scale[over] = 0.02/(sn[over]+1e-15)
            step = step*scale
        x = _norm_rows(x-step)
    return x

def lbfgs_smooth_full(x_init, tau=0.008, maxiter=80):
    x_init = _norm_rows(x_init.copy())
    n,d = x_init.shape
    iu = _get_triu(n)
    I = iu[0]; J = iu[1]
    y0 = x_init.ravel().copy()
    def fun(yflat):
        Y = yflat.reshape(n,d)
        nr = np.sqrt(np.sum(Y*Y,axis=1,keepdims=True))
        nr[nr<1e-12]=1e-12
        X = Y/nr
        G = X@X.T
        dots = G[iu]
        m = float(np.max(dots))
        e = np.exp((dots-m)/tau)
        s = float(np.sum(e))
        if s<=0: s=1e-300
        f = m+tau*np.log(s)
        w = e/s
        gx = np.zeros((n,d))
        np.add.at(gx,I,w[:,None]*X[J])
        np.add.at(gx,J,w[:,None]*X[I])
        dot = np.sum(gx*X,axis=1,keepdims=True)
        gy = (gx-dot*X)/nr
        return float(f), gy.ravel()
    try:
        res = minimize(fun,y0,jac=True,method="L-BFGS-B",options={"maxiter":int(maxiter),"ftol":1e-12,"gtol":1e-7,"maxfun":20000})
        Y = res.x.reshape(n,d)
        return _norm_rows(Y)
    except Exception:
        return x_init

def build_orbit(P, m, q):
    # P (K,3): alpha,t1,t2
    K = P.shape[0]
    alph = P[:,0]; t1=P[:,1]; t2=P[:,2]
    r1 = np.cos(alph); r2=np.sin(alph)
    ks = np.arange(m)
    c1 = 2*np.pi*ks/float(m)
    c2 = 2*np.pi*float(q)*ks/float(m)
    N = K*m
    X = np.empty((N,4),dtype=np.float64)
    for j in range(K):
        a1 = t1[j]+c1; a2=t2[j]+c2
        rr1=r1[j]; rr2=r2[j]
        X[j*m:(j+1)*m,0]=rr1*np.cos(a1)
        X[j*m:(j+1)*m,1]=rr1*np.sin(a1)
        X[j*m:(j+1)*m,2]=rr2*np.cos(a2)
        X[j*m:(j+1)*m,3]=rr2*np.sin(a2)
    return _norm_rows(X)

def orbit_fun_factory(m,q,K,tau,iu):
    ks = np.arange(m)
    off1 = 2*np.pi*ks/float(m)
    off2 = 2*np.pi*float(q)*ks/float(m)
    cos1_base = None  # computed inside
    def fun(pflat):
        P = pflat.reshape(K,3)
        alph=P[:,0]; t1=P[:,1]; t2=P[:,2]
        r1=np.cos(alph); r2=np.sin(alph)
        N=K*m
        X=np.empty((N,4),dtype=np.float64)
        # store per-orbit angles for grad
        A1=np.empty((K,m)); A2=np.empty((K,m))
        for j in range(K):
            a1=t1[j]+off1; a2=t2[j]+off2
            A1[j]=a1; A2[j]=a2
            X[j*m:(j+1)*m,0]=r1[j]*np.cos(a1)
            X[j*m:(j+1)*m,1]=r1[j]*np.sin(a1)
            X[j*m:(j+1)*m,2]=r2[j]*np.cos(a2)
            X[j*m:(j+1)*m,3]=r2[j]*np.sin(a2)
        G=X@X.T
        dots=G[iu]
        mx=float(np.max(dots))
        e=np.exp((dots-mx)/tau)
        s=float(np.sum(e))
        f=mx+tau*np.log(s+1e-300)
        w=e/(s+1e-300)
        # symmetric weight matrix action without forming full matrix
        # grad_X = Wsym @ X
        # use accumulation
        Ws=np.zeros((N,N),dtype=np.float64)
        Ws[iu]=w
        Ws=Ws+Ws.T
        gradX=Ws@X
        g=np.zeros((K,3),dtype=np.float64)
        for j in range(K):
            gx=gradX[j*m:(j+1)*m]  # (m,4)
            a1=A1[j]; a2=A2[j]
            c1=np.cos(a1); s1=np.sin(a1); c2=np.cos(a2); s2=np.sin(a2)
            rr1=r1[j]; rr2=r2[j]
            # dx/dalpha
            dxa=np.empty((m,4))
            dxa[:,0]=-rr2*c1; dxa[:,1]=-rr2*s1; dxa[:,2]=rr1*c2; dxa[:,3]=rr1*s2
            dxt1=np.empty((m,4))
            dxt1[:,0]=-rr1*s1; dxt1[:,1]=rr1*c1; dxt1[:,2]=0; dxt1[:,3]=0
            dxt2=np.empty((m,4))
            dxt2[:,0]=0; dxt2[:,1]=0; dxt2[:,2]=-rr2*s2; dxt2[:,3]=rr2*c2
            g[j,0]=float(np.sum(gx*dxa))
            g[j,1]=float(np.sum(gx*dxt1))
            g[j,2]=float(np.sum(gx*dxt2))
        return float(f), g.ravel()
    return fun

def solve():
    t0=time.time()
    T_END=t0+172.0
    SYM_END=t0+60.0
    rng=np.random.default_rng(20261031)
    n,d=153,4
    iu=_get_triu(n)
    best_x=None; best_v=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,dtype=np.float64)
            if a.shape==(n,d):
                a=_norm_rows(a)
                v=_max_cos(a)
                if v<best_v:
                    best_v=v; best_x=a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x=_norm_rows(rng.standard_normal(size=(n,d)))
        best_v=_max_cos(best_x)
    global_best=best_x.copy(); global_val=best_v

    # ---- Stage 1: symmetry-constrained search ----
    configs=[]
    for m,K,qs in [(51,3,[1,2,4,7]),(17,9,[1,2,3,5]),(9,17,[1,2])]:
        for q in qs:
            import math
            if math.gcd(m,q)!=1:
                continue
            configs.append((m,K,q))
    # restarts per config
    orbit_cands=[]  # (val, X)
    # shuffle configs deterministically
    order=rng.permutation(len(configs))
    for oi in order:
        m,K,q=configs[oi]
        if time.time()>SYM_END-2.0:
            break
        if K==3: nrep=5
        elif K==9: nrep=3
        else: nrep=2
        for rep in range(nrep):
            if time.time()>SYM_END-1.0:
                break
            alph0=rng.uniform(0.25,1.32,size=K)
            t10=rng.uniform(0,2*np.pi,size=K)
            t20=rng.uniform(0,2*np.pi,size=K)
            p0=np.stack([alph0,t10,t20],axis=1).ravel()
            lb=np.tile(np.array([0.06,0.0,0.0]),K)
            ub=np.tile(np.array([1.51,2*np.pi,2*np.pi]),K)
            bounds=list(zip(lb,ub))
            try:
                fun1=orbit_fun_factory(m,q,K,0.015,iu)
                res1=minimize(fun1,p0,jac=True,method="L-BFGS-B",bounds=bounds,options={"maxiter":90,"ftol":1e-10,"gtol":1e-6})
                p1=res1.x if res1 is not None and hasattr(res1,'x') else p0
            except Exception:
                p1=p0
            try:
                fun2=orbit_fun_factory(m,q,K,0.006,iu)
                res2=minimize(fun2,p1,jac=True,method="L-BFGS-B",bounds=bounds,options={"maxiter":90,"ftol":1e-10,"gtol":1e-6})
                p2=res2.x if res2 is not None and hasattr(res2,'x') else p1
            except Exception:
                p2=p1
            try:
                P=p2.reshape(K,3)
                X=build_orbit(P,m,q)
                v=_max_cos(X)
                # duplicate-orbit rejection: need distinct points
                if v>0.999:
                    continue
                # also reject duplicate orbits: check orbit-orbit min distance too small means same orbit
                orbit_cands.append((v,X.copy(),(m,q)))
            except Exception:
                continue
    orbit_cands.sort(key=lambda t:t[0])
    # keep top few
    top=[]
    for v,X,meta in orbit_cands[:6]:
        top.append((v,X,meta))
    # if nothing found, create random orbit seed
    if len(top)==0:
        m,K,q=51,3,2
        P=np.stack([rng.uniform(0.3,1.27,size=K),rng.uniform(0,2*np.pi,size=K),rng.uniform(0,2*np.pi,size=K)],axis=1)
        top=[(0.99,build_orbit(P,m,q),(m,q))]

    # ---- Stage 2: gradual symmetry breaking ----
    scales=[0.001,0.003,0.01,0.03]
    broken_best=None; broken_val=1e9
    # evaluate at most 3 seeds to save time
    nseeds=min(3,len(top))
    # time per seed budget
    for si in range(nseeds):
        if time.time()>T_END-25.0:
            break
        v0,X0,meta=top[si]
        X=X0.copy()
        # quick polish of symmetric point first
        try:
            X=lbfgs_smooth_full(X,tau=0.008,maxiter=40)
            X=_norm_rows(X)
        except Exception:
            pass
        for sc in scales:
            if time.time()>T_END-20.0:
                break
            try:
                noise=rng.standard_normal(size=(n,d))
                dot=np.sum(noise*X,axis=1,keepdims=True)
                noise=noise-dot*X
                Y=_norm_rows(X+float(sc)*noise)
                # soft-max quench: two taus
                Y=lbfgs_smooth_full(Y,tau=0.010,maxiter=50)
                Y=smooth_relax(Y,steps=60,beta=70.0,lr=0.008)
                Y=lbfgs_smooth_full(Y,tau=0.005,maxiter=40)
                t_end=min(time.time()+2.0,T_END-15.0)
                if t_end-time.time()>0.3:
                    Y=lp_polish(Y,t_end,r0=0.004)
                X=Y
            except Exception:
                continue
        try:
            v=_max_cos(X)
        except Exception:
            continue
        if v<broken_val:
            broken_val=v; broken_best=X.copy()
        # also update global if better
        if v<global_val:
            global_val=v; global_best=X.copy()
        # extra kicks on best broken so far within remaining budget
        # small additional annealed quench
        if time.time()<T_END-20.0:
            try:
                Y2=lbfgs_smooth_full(X,tau=0.004,maxiter=60)
                t_end=min(time.time()+1.5,T_END-12.0)
                Y2=lp_polish(Y2,t_end,r0=0.003)
                v2=_max_cos(Y2)
                if v2<broken_val:
                    broken_val=v2; broken_best=Y2.copy()
                if v2<global_val:
                    global_val=v2; global_best=Y2.copy()
            except Exception:
                pass

    # ---- Stage 3: spend remaining time polishing best ----
    # combine broken_best and global_best
    cur = global_best.copy()
    # a few random-restart quenches around cur with small sigma
    while time.time()<T_END-4.0:
        try:
            r=rng.random()
            if r<0.6 and broken_best is not None:
                base=broken_best if rng.random()<0.5 else cur
                sigma=float(10.0**rng.uniform(-3.0,-1.5))
                noise=rng.standard_normal(size=(n,d))
                dot=np.sum(noise*base,axis=1,keepdims=True)
                noise=noise-dot*base
                Y=_norm_rows(base+sigma*noise)
                Y=lbfgs_smooth_full(Y,tau=0.006,maxiter=60)
                Y=smooth_relax(Y,steps=40,beta=80.0,lr=0.006)
            else:
                # subset re-randomize worst rows
                G=cur@cur.T; np.fill_diagonal(G,-2.0)
                mrow=np.max(G,axis=1)
                order=np.argsort(-mrow)
                k=int(rng.integers(1,5))
                pool=order[:14]
                idx=rng.choice(pool,size=k,replace=False)
                Y=cur.copy()
                Y[idx]=rng.standard_normal(size=(k,d))
                Y=_norm_rows(Y)
                Y=smooth_relax(Y,steps=60,beta=65.0,lr=0.008)
                Y=lbfgs_smooth_full(Y,tau=0.007,maxiter=50)
            t_end=min(time.time()+1.8,T_END-2.0)
            if t_end-time.time()<0.3:
                break
            Y2=lp_polish(Y,t_end,r0=0.004)
            v2=_max_cos(Y2)
            if v2<global_val-1e-13:
                global_val=v2; global_best=Y2.copy(); cur=Y2.copy()
                if broken_best is None or v2<broken_val:
                    broken_val=v2; broken_best=Y2.copy()
            elif v2< _max_cos(cur) and rng.random()<0.15:
                cur=Y2.copy()
        except Exception:
            continue
    # final tight polish
    try:
        t_end=min(time.time()+3.0,T_END-0.5)
        global_best=lp_polish(global_best,t_end,r0=0.002)
        global_val=_max_cos(global_best)
    except Exception:
        pass
    global_best=_norm_rows(global_best)
    # safety: if somehow worse than loaded best, return loaded best polished
    # (already tracked) just return global_best
    return global_best

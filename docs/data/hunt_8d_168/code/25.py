import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def balanced_quench(X, steps, b0, b1, lr0, lr1, mu=0.85, lam=0.12, bl_frac=0.25, seed_jit=0.0):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    one_mu = 1.0 - mu
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps-1, 1)
        beta = np.exp(lb0 + (lb1-lb0)*frac)
        lr = np.exp(ll0 + (ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        # global weights
        E = np.exp(beta*(G-m))
        np.fill_diagonal(E, 0.0)
        Sg = E.sum()/max(n,1) + 1e-30
        Grad_g = (E @ X)/Sg  # row magnitude ~ stress
        # local spread weights with lower beta
        bl = max(beta*bl_frac, 8.0)
        # per-row max for stability
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl*(G-mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True)+1e-30
        Grad_l = (W @ X)/Rs
        Grad = Grad_g + lam*Grad_l
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        # tame outliers
        nn = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True))+1e-12
        # clip huge grads
        scale = np.minimum(1.0, 3.0/(nn+1e-12))
        Gt = Gt*scale
        V = mu*V + one_mu*Gt
        X = X - lr*V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    # return best
    return best

def hard_polish(X, steps=300, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=1e-6):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(steps):
        frac = s/max(steps-1,1)
        tol = tol0+(tol1-tol0)*frac
        lr = lr0*(lr1/lr0)**frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m-tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True)+1e-12
        active = (cnt[:,0]>0.5)
        if not np.any(active):
            continue
        Grad = (M @ X)/cnt
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad-dot*X
        nn = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True))+1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active]/nn[active]
        X = X - lr*step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def lp_polish(x, seconds):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n,d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n,1)
    while time.time()-start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p==0:
            radius*=0.5
            continue
        # cap constraints to avoid huge LP
        if p>2500:
            # keep tightest 2500
            idx = np.argsort(c[sel])[-2500:]
            I, J = I[idx], J[idx]
            p = 2500
            # need corresponding c values
            cc = c[sel][idx]
        else:
            cc = c[sel]
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius*=0.5
    return x

def build_E8():
    vecs=[]
    for i in range(8):
        for j in range(i+1,8):
            for si in (1.0,-1.0):
                for sj in (1.0,-1.0):
                    v=np.zeros(8); v[i]=si; v[j]=sj; vecs.append(v)
    for mask in range(256):
        cnt=0; s=np.ones(8)
        for k in range(8):
            if (mask>>k)&1:
                s[k]=-1.0; cnt+=1
        if cnt%2==0:
            vecs.append(0.5*s)
    E=np.array(vecs,float); E/=np.sqrt(2.0)
    return E

def refine_hole(Y, p, iters=160, beta=70.0, lr0=0.15, lr1=0.005):
    p=p/np.linalg.norm(p)
    for it in range(iters):
        frac=it/max(iters-1,1)
        lr=lr0*(lr1/lr0)**frac
        g=Y@p
        m=g.max()
        w=np.exp(beta*(g-m))
        s=w.sum()+1e-30
        A=(w@Y)/s
        At=A-float(A@p)*p
        p=p-lr*At
        nr=np.linalg.norm(p)
        if nr==0: break
        p/=nr
    return p

def farthest_reinsert(keep, k, n_probe=6000):
    cur=keep.copy()
    outs=[]
    d=keep.shape[1]
    for _ in range(k):
        P=np.random.randn(n_probe,d); P/=np.linalg.norm(P,axis=1,keepdims=True)
        # batched max
        CH=2000
        bs=np.empty(n_probe)
        for a in range(0,n_probe,CH):
            b=min(a+CH,n_probe)
            bs[a:b]=(P[a:b]@cur.T).max(axis=1)
        order=np.argsort(bs)
        best_s=np.inf; best_c=None
        for cand_idx in order[:10]:
            cand=P[cand_idx]
            cand=refine_hole(cur,cand,iters=120,beta=80.0)
            s=float(np.max(cand@cur.T))
            if s<best_s:
                best_s=s; best_c=cand
        outs.append(best_c)
        cur=np.vstack([cur,best_c.reshape(1,d)])
    return np.array(outs)

def rattler_centering(X, thresh=5e-8):
    X=X.copy()
    n=X.shape[0]
    G=X@X.T; np.fill_diagonal(G,-2.0)
    m=float(G.max()); mx=G.max(axis=1)
    ratt=np.where(mx < m-thresh)[0]
    if len(ratt)==0:
        return X
    # order by worst hole first (lowest mx = deepest hole? actually most free)
    # center those with smallest mx last? do ascending
    order=np.argsort(mx[ratt])
    Y_fixed=X.copy()
    for idx in ratt[order[:24]]:
        # hole w.r.t all other points
        mask=np.ones(n,bool); mask[idx]=False
        Y=Y_fixed[mask]
        # probe best of random
        P=np.random.randn(3000,8); P/=np.linalg.norm(P,axis=1,keepdims=True)
        bs=(P@Y.T).max(axis=1)
        p0=P[np.argmin(bs)]
        pnew=refine_hole(Y,p0,iters=160,beta=60.0)
        # only accept if not worsening global max
        old_max=float(np.max(Y_fixed[idx]@Y.T))
        new_max=float(np.max(pnew@Y.T))
        # allow if new_max <= max(old_max, m-1e-9)
        if new_max <= max(old_max, m-1e-9)+1e-12:
            Y_fixed[idx]=pnew
    return Y_fixed

def solve():
    np.random.seed(12345)
    t0=time.time()
    TOTAL=172.0
    try:
        Xb=np.load("best.npy").astype(float).reshape(168,8)
    except Exception:
        try:
            Xb=np.load("record.npy").astype(float).reshape(168,8)
        except Exception:
            Xb=np.random.randn(168,8)
    Xb/=np.linalg.norm(Xb,axis=1,keepdims=True)
    try:
        Xr=np.load("record.npy").astype(float).reshape(168,8)
        Xr/=np.linalg.norm(Xr,axis=1,keepdims=True)
        if _max_cos(Xr)<_max_cos(Xb):
            Xb=Xr.copy()
    except Exception:
        pass
    best_val=_max_cos(Xb)
    X_best=Xb.copy()
    X_cur=Xb.copy(); cur_val=best_val
    E8=build_E8()
    T=0.002
    it=0
    # quick rattler center of start
    try:
        Xr2=rattler_centering(X_best)
        v2=_max_cos(Xr2)
        if v2<=best_val:
            X_best=Xr2; best_val=v2; X_cur=Xr2.copy(); cur_val=v2
    except Exception:
        pass
    while time.time()-t0 < TOTAL-34.0:
        it+=1
        r=np.random.rand()
        if r<0.07 and it>2:
            X_init=np.random.randn(168,8); X_init/=np.linalg.norm(X_init,axis=1,keepdims=True)
            steps,a0,a1,lr0,lr1,mu,lam=1800,8.0,350.0,0.06,0.003,0.85,0.10
        elif r<0.18:
            idx=np.random.choice(240,168,replace=False)
            X_init=E8[idx]+0.02*np.random.randn(168,8)
            if np.random.rand()<0.3:
                Q,_=np.linalg.qr(np.random.randn(8,8))
                if np.linalg.det(Q)<0: Q[:,0]=-Q[:,0]
                X_init=X_init@Q.T
            X_init/=np.linalg.norm(X_init,axis=1,keepdims=True)
            steps,a0,a1,lr0,lr1,mu,lam=1500,10.0,400.0,0.05,0.0025,0.85,0.12
        elif r<0.50:
            base=X_cur if np.random.rand()<0.65 else X_best
            G=base@base.T; np.fill_diagonal(G,-2.0)
            m=float(G.max()); mx=G.max(axis=1); cnt=(G>m-1e-7).sum(axis=1)
            score=mx+0.0015*cnt
            worst=np.argsort(score)[-6:]
            sigma=float(np.random.choice([0.004,0.008,0.015,0.03,0.06,0.12]))
            X_init=base+sigma*np.random.randn(168,8)
            if np.random.rand()<0.65:
                X_init[worst]+=np.random.randn(6,8)*sigma*2.0
            # occasional cap shake
            if np.random.rand()<0.25:
                u=np.random.randn(8); u/=np.linalg.norm(u)
                proj=base@u
                cap=np.argsort(proj)[-12:]
                X_init[cap]+=0.03*np.random.randn(12,8)
            X_init/=np.linalg.norm(X_init,axis=1,keepdims=True)
            if sigma>=0.06:
                steps,a0,a1,lr0,lr1,mu,lam=1900,8.0,420.0,0.05,0.002,0.88,0.10
            else:
                steps,a0,a1,lr0,lr1,mu,lam=1300,15.0,520.0,0.045,0.002,0.85,0.15
        else:
            base=X_cur if np.random.rand()<0.6 else X_best
            G=base@base.T; np.fill_diagonal(G,-2.0)
            m=float(G.max()); mx=G.max(axis=1); cnt=(G>m-1e-7).sum(axis=1)
            score=mx+0.0015*cnt
            k=int(np.random.choice([1,2,3,4,5],p=[0.3,0.28,0.2,0.12,0.1]))
            worst=np.argsort(score)[-k:]
            # sometimes eject random rattler instead to reposition free points
            if np.random.rand()<0.3:
                ratt=np.where(mx<m-1e-7)[0]
                if len(ratt)>0:
                    worst=np.random.choice(ratt,min(k,len(ratt)),replace=False)
                    mask=np.ones(168,bool); mask[worst]=False
                    keep=base[mask]
                    try:
                        newp=farthest_reinsert(keep,len(worst),n_probe=5000)
                    except Exception:
                        newp=np.random.randn(len(worst),8); newp/=np.linalg.norm(newp,axis=1,keepdims=True)
                    X_init=np.vstack([keep,newp])
                    X_init+=0.003*np.random.randn(168,8)
                    X_init/=np.linalg.norm(X_init,axis=1,keepdims=True)
                    steps,a0,a1,lr0,lr1,mu,lam=1500,12.0,480.0,0.05,0.002,0.86,0.15
                else:
                    mask=np.ones(168,bool); mask[worst]=False
                    keep=base[mask]
                    try:
                        newp=farthest_reinsert(keep,k,n_probe=5000)
                    except Exception:
                        newp=np.random.randn(k,8); newp/=np.linalg.norm(newp,axis=1,keepdims=True)
                    X_init=np.vstack([keep,newp])
                    X_init+=0.004*np.random.randn(168,8)
                    X_init/=np.linalg.norm(X_init,axis=1,keepdims=True)
                    steps,a0,a1,lr0,lr1,mu,lam=1600,12.0,480.0,0.05,0.002,0.86,0.12
            else:
                mask=np.ones(168,bool); mask[worst]=False
                keep=base[mask]
                try:
                    newp=farthest_reinsert(keep,k,n_probe=5000)
                except Exception:
                    newp=np.random.randn(k,8); newp/=np.linalg.norm(newp,axis=1,keepdims=True)
                X_init=np.vstack([keep,newp])
                X_init+=0.004*np.random.randn(168,8)
                X_init/=np.linalg.norm(X_init,axis=1,keepdims=True)
                steps,a0,a1,lr0,lr1,mu,lam=1600,12.0,480.0,0.05,0.002,0.86,0.12
        # anneal lam slightly randomly
        lam = lam*np.random.uniform(0.6,1.4)
        blf = float(np.random.uniform(0.15,0.4))
        try:
            X_opt=balanced_quench(X_init,steps,a0,a1,lr0,lr1,mu=mu,lam=lam,bl_frac=blf)
        except Exception:
            continue
        # cheap hard polish + rattler center on promising
        v=_max_cos(X_opt)
        if v < cur_val+0.005:
            try:
                X_opt=hard_polish(X_opt,steps=200,lr0=0.002,lr1=0.0001)
                v=_max_cos(X_opt)
                if np.random.rand()<0.5:
                    X_opt2=rattler_centering(X_opt)
                    v2=_max_cos(X_opt2)
                    if v2<=v+1e-12:
                        X_opt=X_opt2; v=v2
            except Exception:
                pass
        if v < best_val:
            best_val=v; X_best=X_opt.copy(); X_cur=X_opt.copy(); cur_val=v
        else:
            if v<cur_val or np.random.rand()<np.exp(-(v-cur_val)/max(T,1e-7)):
                X_cur=X_opt.copy(); cur_val=v
        T=max(T*0.997,0.0002)
        if time.time()-t0>TOTAL-36:
            break
    # final squeeze
    try:
        if time.time()-t0<TOTAL-30:
            X_best=balanced_quench(X_best,900,70.0,900.0,0.012,0.001,mu=0.7,lam=0.05,bl_frac=0.2)
            X_best=hard_polish(X_best,steps=300,lr0=0.002,lr1=0.00008)
            v=_max_cos(X_best)
            if v<best_val: best_val=v
    except Exception:
        pass
    try:
        X_best=rattler_centering(X_best)
    except Exception:
        pass
    remain=TOTAL-(time.time()-t0)-1.0
    if remain>4:
        try:
            X_best=lp_polish(X_best,seconds=min(remain,26.0))
        except Exception:
            pass
        # one more high-beta touch after LP
        try:
            if TOTAL-(time.time()-t0)>6:
                X2=balanced_quench(X_best,400,200.0,2000.0,0.006,0.0005,mu=0.6,lam=0.02,bl_frac=0.15)
                if _max_cos(X2)<_max_cos(X_best):
                    X_best=X2
                    X_best=lp_polish(X_best,seconds=max(2.0,TOTAL-(time.time()-t0)-1.0))
        except Exception:
            pass
    X_best/=np.linalg.norm(X_best,axis=1,keepdims=True)
    return X_best

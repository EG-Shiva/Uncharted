import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def build_full(reps, ms, ab):
    R = reps.shape[0]
    N = int(np.sum(ms))
    X = np.empty((N, 4), dtype=float)
    pos = 0
    for i in range(R):
        m = int(ms[i]); a = int(ab[i,0]); b = int(ab[i,1])
        x1, y1, x2, y2 = reps[i]
        k = np.arange(m)
        th1 = 2.0*np.pi*a*k/max(1,m)
        th2 = 2.0*np.pi*b*k/max(1,m)
        c1 = np.cos(th1); s1 = np.sin(th1)
        c2 = np.cos(th2); s2 = np.sin(th2)
        X[pos:pos+m,0] = x1*c1 - y1*s1
        X[pos:pos+m,1] = x1*s1 + y1*c1
        X[pos:pos+m,2] = x2*c2 - y2*s2
        X[pos:pos+m,3] = x2*s2 + y2*c2
        pos += m
    return X

def pullback(grad_full, ms, ab):
    R = len(ms)
    rep = np.zeros((R,4), dtype=float)
    pos = 0
    for i in range(R):
        m = int(ms[i]); a = int(ab[i,0]); b = int(ab[i,1])
        G = grad_full[pos:pos+m]
        k = np.arange(m)
        th1 = 2.0*np.pi*a*k/max(1,m)
        th2 = 2.0*np.pi*b*k/max(1,m)
        c1 = np.cos(th1); s1 = np.sin(th1)
        c2 = np.cos(th2); s2 = np.sin(th2)
        gx1 = G[:,0]; gy1 = G[:,1]; gx2 = G[:,2]; gy2 = G[:,3]
        rx1 = c1*gx1 + s1*gy1
        ry1 = -s1*gx1 + c1*gy1
        rx2 = c2*gx2 + s2*gy2
        ry2 = -s2*gx2 + c2*gy2
        rep[i,0] = np.sum(rx1); rep[i,1] = np.sum(ry1)
        rep[i,2] = np.sum(rx2); rep[i,3] = np.sum(ry2)
        pos += m
    return rep

def orbit_order(a,b,m):
    if a % m == 0:
        oa = 1
    else:
        oa = m // math.gcd(int(a)%m, m)
    if b % m == 0:
        ob = 1
    else:
        ob = m // math.gcd(int(b)%m, m)
    g = math.gcd(oa, ob)
    return oa*ob//g

def relax_reps(reps, ms, ab, iters, k0, k1, lr0, lr1, mu):
    reps = reps.copy()
    R = reps.shape[0]
    vel = np.zeros_like(reps)
    for t in range(iters):
        frac = t/max(1,iters-1)
        kk = k0*(k1/k0)**frac
        lr = lr0*(lr1/lr0)**frac
        X = build_full(reps, ms, ab)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        moff = float(np.max(G))
        W = np.exp(kk*(G-moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)+1e-12
        grad_full = (W @ X)/rs
        rg = pullback(grad_full, ms, ab)
        dot = np.sum(rg*reps, axis=1, keepdims=True)
        rg = rg - dot*reps
        # normalize scale by orbit size to keep lr stable
        # divide by ms
        for i in range(R):
            rg[i] /= float(ms[i])
        vel = mu*vel + rg
        denom = 1.0 - mu**(t+1) if mu>0 else 1.0
        vc = vel/denom if denom!=0 else vel
        reps = reps - lr*vc*float(np.mean(ms))
        # renormalize
        reps = reps/np.linalg.norm(reps, axis=1, keepdims=True)
    return reps

def smooth_relax(X, iters, k0, k1, lr0, lr1, mu, noise, rng):
    x = X.copy()
    n = x.shape[0]
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t/max(1,iters-1)
        k = k0*(k1/k0)**frac
        lr = lr0*(lr1/lr0)**frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(np.max(G))
        W = np.exp(k*(G-moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)+1e-12
        grad = (W @ x)/rs
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        v = mu*v + gt
        vc = v/(1.0-mu**(t+1)) if mu>0 else v
        x = x - lr*vc
        if noise>0 and frac<0.6:
            amp = noise*(1.0-frac/0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert*x, axis=1, keepdims=True)
            pert -= pd*x
            x = x + amp*pert
        x = x/np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n,d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n,1)
    while time.time()-start < seconds and radius>1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20*radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cc = c[sel]
        p = len(I); nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = y/np.linalg.norm(y,axis=1,keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x,best,radius = y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def random_ab_set(ms, rng):
    R=len(ms)
    ab=np.zeros((R,2),dtype=int)
    for i in range(R):
        m=int(ms[i])
        # try to get full order
        for _ in range(50):
            a=int(rng.integers(0,m)); b=int(rng.integers(0,m))
            if a==0 and b==0:
                continue
            if orbit_order(a,b,m)==m:
                ab[i,0]=a; ab[i,1]=b
                break
        else:
            ab[i,0]=1; ab[i,1]=int(rng.integers(1,m))
            if ab[i,1]==0:
                ab[i,1]=1
    return ab

def solve():
    t0=time.time()
    rng=np.random.default_rng(0)
    n,d=153,4
    try:
        best=np.load("best.npy")
    except Exception:
        try:
            best=np.load("record.npy")
        except Exception:
            best=rng.standard_normal((n,d))
    best=np.asarray(best,dtype=float)
    if best.shape!=(n,d):
        best=rng.standard_normal((n,d))
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    best_val=max_cos(best)
    # symmetric search
    sym_cands=[]  # list of (val, reps, ms, ab, X)
    partitions=[[17]*9,[51]*3,[9]*17,[3]*51,[51,51,17,17,17],[17]*6+[51]]
    # last: 6*17=102+51=153
    DEAD1=t0+50
    trial=0
    while time.time()<DEAD1:
        trial+=1
        # choose partition
        r=rng.random()
        if r<0.55:
            ms=np.array([17]*9,dtype=int)
        elif r<0.70:
            ms=np.array([51]*3,dtype=int)
        elif r<0.80:
            ms=np.array([9]*17,dtype=int)
        elif r<0.90:
            ms=np.array([51,51,17,17,17],dtype=int)
        else:
            ms=np.array(partitions[int(rng.integers(len(partitions)))],dtype=int)
        R=len(ms)
        ab=random_ab_set(ms,rng)
        reps=rng.standard_normal((R,4))
        reps=reps/np.linalg.norm(reps,axis=1,keepdims=True)
        # enforce not too small radii
        for i in range(R):
            r1=np.linalg.norm(reps[i,0:2]); r2=np.linalg.norm(reps[i,2:4])
            if min(r1,r2)<0.25:
                # rebalance toward 0.707
                reps[i,0:2]*= (0.5/max(1e-6,r1**2+0.25))**0.25 if False else 1.0
                # simple: redraw if too small
                if min(r1,r2)<0.12:
                    v=rng.standard_normal(4); v/=np.linalg.norm(v)
                    # mix to ensure both comps sizeable
                    reps[i]=v
        try:
            reps2=relax_reps(reps,ms,ab,iters=350,k0=12.0,k1=220.0,lr0=0.06,lr1=0.004,mu=0.85)
            X=build_full(reps2,ms,ab)
            # check distinctness
            G=X@X.T
            np.fill_diagonal(G,-2.0)
            v=float(np.max(G))
        except Exception:
            continue
        if v>0.99:
            continue
        # keep top
        sym_cands.append((v,reps2.copy(),ms.copy(),ab.copy(),X.copy()))
        sym_cands=sorted(sym_cands,key=lambda z:z[0])[:4]
        if time.time()>DEAD1-1:
            break
    # fully relax best few symmetric
    DEAD2=t0+72
    new_cands=[]
    for (v,reps,ms,ab,X) in sym_cands[:3]:
        if time.time()>DEAD2-2:
            break
        try:
            repsR=relax_reps(reps,ms,ab,iters=700,k0=15.0,k1=350.0,lr0=0.04,lr1=0.002,mu=0.9)
            XR=build_full(repsR,ms,ab)
            vR=max_cos(XR)
        except Exception:
            continue
        new_cands.append((vR,repsR,ms,ab,XR))
    if new_cands:
        sym_cands=sorted(new_cands+sym_cands,key=lambda z:z[0])[:4]
    # symmetry breaking + smooth relaxation
    break_cands=[]
    if sym_cands:
        for idx,(v,reps,ms,ab,Xs) in enumerate(sym_cands[:3]):
            # several amplitudes 0.05-0.15
            for sig in ([0.05,0.10] if idx==0 else [0.07]):
                if time.time()>t0+125:
                    break
                try:
                    Xp=Xs+sig*rng.standard_normal(Xs.shape)
                    Xp=Xp/np.linalg.norm(Xp,axis=1,keepdims=True)
                    Xo=smooth_relax(Xp,iters=800,k0=15.0,k1=300.0,lr0=0.04,lr1=0.002,mu=0.85,noise=0.0,rng=rng)
                    vo=max_cos(Xo)
                    break_cands.append((vo,Xo.copy()))
                except Exception:
                    continue
    # also direct smooth from symmetric with gradual ramp: two-stage perturb
    if sym_cands and time.time()<t0+130:
        try:
            _,_,_,_,Xb0=sym_cands[0]
            Xg=Xb0.copy()
            for sig,its in [(0.03,300),(0.06,400),(0.10,500)]:
                if time.time()>t0+135:
                    break
                Xg=Xg+sig*0.5*rng.standard_normal(Xg.shape)
                Xg/=np.linalg.norm(Xg,axis=1,keepdims=True)
                Xg=smooth_relax(Xg,iters=its,k0=12.0,k1=250.0,lr0=0.03,lr1=0.002,mu=0.85,noise=0.0,rng=rng)
            vg=max_cos(Xg)
            break_cands.append((vg,Xg.copy()))
        except Exception:
            pass
    # collect all candidates including file best
    allc=[(best_val,best.copy())]
    for v,X in break_cands:
        allc.append((v,X))
    for (v,_,_,_,X) in sym_cands[:2]:
        allc.append((v,X))
    allc=sorted(allc,key=lambda z:z[0])[:4]
    # final polish with remaining time
    time_left=172-(time.time()-t0)
    if time_left<8:
        time_left=8
    per=time_left/min(3,len(allc))
    polished=None; pval=1e9
    for i,(v,xc) in enumerate(allc[:3]):
        budget=per*(1.6 if i==0 else 1.0)
        budget=min(budget,55)
        # first smooth a bit if coming from symmetric (exact symmetry hard for LP)
        try:
            if i>0 or v>0.86:
                xc2=smooth_relax(xc,iters=400,k0=20.0,k1=400.0,lr0=0.02,lr1=0.001,mu=0.8,noise=0.0,rng=rng)
            else:
                xc2=xc
            xp=lp_polish(xc2,budget)
        except Exception:
            xp=xc
        try:
            vp=max_cos(xp)
        except Exception:
            continue
        if vp<pval:
            pval=vp; polished=xp.copy()
        if time.time()-t0>172:
            break
    if polished is None:
        polished=best
    polished=polished/np.linalg.norm(polished,axis=1,keepdims=True)
    # return best among polished and file best
    try:
        if max_cos(polished)>best_val:
            return best
    except Exception:
        return best
    return polished

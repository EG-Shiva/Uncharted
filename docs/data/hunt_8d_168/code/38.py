import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _group_mats(m, freqs):
    # freqs length 4 for 4 planes (8D). freq 0 = fixed plane.
    mats = np.zeros((m, 8, 8))
    for k in range(m):
        M = np.zeros((8, 8))
        for p, f in enumerate(freqs):
            th = 2.0 * np.pi * k * f / m
            c = np.cos(th); s = np.sin(th)
            M[2*p, 2*p] = c; M[2*p, 2*p+1] = -s
            M[2*p+1, 2*p] = s; M[2*p+1, 2*p+1] = c
        mats[k] = M
    return mats

def _gen_full(seeds, mats):
    ns = seeds.shape[0]; m = mats.shape[0]
    N = ns * m
    X = np.empty((N, 8))
    for k in range(m):
        Mk = mats[k]
        # seeds (ns,8) @ Mk.T
        X[k::m] if False else None
    # ordering s*m+k
    for s in range(ns):
        ss = seeds[s]
        for k in range(m):
            X[s*m+k] = mats[k] @ ss
    return X

def _sym_quench(seeds0, mats, steps, beta0, beta1, lr0, lr1, lam, rng_seed=0):
    seeds = seeds0.copy()
    ns, d = seeds.shape
    m = mats.shape[0]
    N = ns * m
    Mt = mats.transpose(0, 2, 1)
    mad = np.zeros_like(seeds); vad = np.zeros_like(seeds)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    best = None; bestv = 1e9
    for t in range(1, steps+1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1-lb0)*frac)
        lr = np.exp(ll0 + (ll1-ll0)*frac)
        X = _gen_full(seeds, mats)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        # track best
        if gmax < bestv:
            bestv = gmax; best = seeds.copy()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-18
        Grs = (W @ X) / S
        if lam > 0:
            Gb = X @ X.T
            np.fill_diagonal(Gb, 0.0)
            np.maximum(Gb, 0.0, out=Gb)
            Grb = (Gb @ X) * (2.0 / N)
            Grad = Grs + lam * Grb
        else:
            Grad = Grs
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        Gtr = Gt.reshape(ns, m, d)
        # pullback: sum_k Mt[k] @ Gtr[s,k]
        sg = np.einsum('kij,skj->si', Mt, Gtr)
        # tangent project on seeds
        ds = (sg*seeds).sum(axis=1, keepdims=True)
        sgt = sg - ds*seeds
        mad *= b1; mad += (1.0-b1)*sgt
        vad *= b2; vad += (1.0-b2)*(sgt*sgt)
        bc1 = 1.0 - b1**t; bc2 = 1.0 - b2**t
        step = (mad/bc1)/(np.sqrt(vad/bc2)+eps)
        seeds = seeds - lr*step
        seeds /= np.sqrt((seeds*seeds).sum(axis=1, keepdims=True)) + 1e-18
    return best, bestv

def _free_quench(X, steps, beta0, beta1, lr0, lr1, lam, t_lim=None):
    n, d = X.shape
    X = X.copy()
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    best = X.copy(); bestv = _max_cos(X)
    for t in range(1, steps+1):
        if t_lim is not None and (t & 63)==0 and time.time() > t_lim:
            break
        frac = t/steps
        beta = np.exp(lb0+(lb1-lb0)*frac)
        lr = np.exp(ll0+(ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < bestv:
            bestv = gmax; best = X.copy()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)+1e-18
        Grs = (W @ X)/S
        if lam>0:
            Gb = X @ X.T
            np.fill_diagonal(Gb, 0.0)
            np.maximum(Gb, 0.0, out=Gb)
            Grb = (Gb @ X)*(2.0/n)
            Grad = Grs + lam*Grb
        else:
            Grad = Grs
        dot=(Grad*X).sum(axis=1, keepdims=True)
        Gt=Grad-dot*X
        m*=b1; m+=(1.0-b1)*Gt
        v*=b2; v+=(1.0-b2)*(Gt*Gt)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(m/bc1)/(np.sqrt(v/bc2)+eps)
        X=X-lr*step
        X/=np.sqrt((X*X).sum(axis=1, keepdims=True))
    return best

def _hard_polish(X, steps=250, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=1e-6):
    X=X.copy()
    n,d=X.shape
    for s in range(steps):
        frac=s/max(steps-1,1)
        tol=tol0+(tol1-tol0)*frac
        lr=lr0*(lr1/lr0)**frac
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        mm=float(G.max())
        M=(G>(mm-tol)).astype(np.float64)
        np.fill_diagonal(M,0.0)
        cnt=M.sum(axis=1,keepdims=True)+1e-12
        act=(cnt[:,0]>0.5)
        if not np.any(act):
            continue
        Grad=(M@X)/cnt
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        nn=np.sqrt((Gt*Gt).sum(axis=1,keepdims=True))+1e-12
        st=np.zeros_like(Gt); st[act]=Gt[act]/nn[act]
        X=X-lr*st
        X/=np.linalg.norm(X,axis=1,keepdims=True)
    return X

def _lp_polish(x, seconds):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds and radius>1e-14:
        c=(x@x.T)[iu]
        mm=float(c.max())
        sel=c>mm-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            radius*=0.5; continue
        if p>2500:
            idx=np.argsort(c[sel])[-2500:]
            I,J=I[idx],J[idx]; p=2500
            cc=c[sel][idx]
        else:
            cc=c[sel]
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv); obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cc,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=_max_cos(y)
        if vv<best:
            x,best, radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _e8_roots():
    pts=[]
    for i in range(8):
        for j in range(i+1,8):
            for si in (1.0,-1.0):
                for sj in (1.0,-1.0):
                    v=np.zeros(8); v[i]=si; v[j]=sj; pts.append(v)
    for mask in range(256):
        v=np.ones(8)*0.5; nm=0
        for k in range(8):
            if (mask>>k)&1:
                v[k]=-0.5; nm+=1
        if nm%2==0:
            pts.append(v.copy())
    A=np.array(pts,float)
    A/=np.linalg.norm(A,axis=1,keepdims=True)
    return A

def _greedy_e8(rng):
    A=_e8_roots()  # 240
    keep=np.ones(240,bool)
    # delete 72 worst by count of cos>0.499
    G=A@A.T
    for _ in range(72):
        idx=np.where(keep)[0]
        sub=G[np.ix_(idx,idx)]
        # count neighbours with >0.499 (excluding diag)
        cnt=(sub>0.499).sum(axis=1)-1
        # break ties randomly
        worst=cnt.max()
        cand=np.where(cnt==worst)[0]
        pick=cand[rng.randint(len(cand))]
        keep[idx[pick]]=False
    return A[keep]

def solve():
    t0=time.time(); BUDGET=175.0
    n,d=168,8
    rng=np.random.RandomState(20261103)
    best=None; bestv=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(float)
            if a.shape!=(n,d): continue
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            if not np.all(np.isfinite(a)): continue
            va=_max_cos(a)
            if va<bestv: bestv=va; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=rng.randn(n,d); best/=np.linalg.norm(best,axis=1,keepdims=True); bestv=_max_cos(best)
    # symmetry configs: (m, freqs, ns)
    configs=[(7,[1,2,3,0],24),(14,[1,3,5,7],12),(21,[1,2,4,8],8),(28,[1,3,5,7],6)]
    sym_cands=[]
    # phase sym: limited time ~55s
    t_sym_end=t0+55.0
    for ci,(m,freqs,ns) in enumerate(configs):
        if time.time()>t_sym_end: break
        mats=_group_mats(m,freqs)
        nrep=3 if ci<2 else 2
        for r in range(nrep):
            if time.time()>t_sym_end: break
            if time.time()>t0+BUDGET-60: break
            s0=rng.randn(ns,d); s0/=np.linalg.norm(s0,axis=1,keepdims=True)
            # vary beta/lr slightly per rep
            b0=18.0+r*6.0; b1=300.0+r*60.0
            l0=0.012-0.002*r; l1=0.0012
            lam=0.02 if (r%2==0) else 0.05
            bs,bv=_sym_quench(s0,mats,steps=850,beta0=b0,beta1=b1,lr0=l0,lr1=l1,lam=lam)
            X=_gen_full(bs,mats)
            vv=_max_cos(X)
            sym_cands.append((vv,X.copy()))
            if vv<bestv:
                bestv=vv; best=X.copy()
    # E8 greedy subset candidates
    try:
        if time.time()<t0+BUDGET-60:
            for e in range(2):
                E=_greedy_e8(rng)
                # random rotation to avoid axis alignment
                Q,_=np.linalg.qr(rng.randn(d,d))
                if np.linalg.det(Q)<0: Q[:,0]*=-1
                Eq=E@Q.T
                vv=_max_cos(Eq)
                sym_cands.append((vv,Eq.copy()))
                if vv<bestv:
                    bestv=vv; best=Eq.copy()
    except Exception:
        pass
    sym_cands=sorted(sym_cands,key=lambda x:x[0])
    # staged release: take top 3 sym candidates + current best
    pool=[]
    for vv,X in sym_cands[:3]:
        pool.append(X.copy())
    if len(pool)==0:
        pool=[best.copy()]
    # ensure best included
    pool.append(best.copy())
    # dedup pool to max 3 by value spread: keep first 3
    pool=pool[:4]
    cur_best=best.copy(); cur_bestv=bestv
    amps=[1e-3,8e-3,3e-2,5e-2]
    t_release_end=t0+BUDGET-18.0
    for pi,Xs in enumerate(pool):
        if time.time()>t_release_end: break
        Xcur=Xs.copy()
        # quick free quench of symmetric point itself (amp 0)
        if time.time()<t_release_end:
            Xq=_free_quench(Xcur,steps=500,beta0=25.0,beta1=380.0,lr0=0.007,lr1=0.0006,lam=0.04,t_lim=t_release_end)
            Xq=_hard_polish(Xq,steps=180)
            vq=_max_cos(Xq)
            if vq<cur_bestv: cur_bestv=vq; cur_best=Xq.copy()
            # continue chain from improved point
            if vq<_max_cos(Xcur): Xcur=Xq.copy()
        for ai,amp in enumerate(amps):
            if time.time()>t_release_end: break
            # orbit-cluster perturbation: half symmetric (rotated common) + half free
            # need mats for this? use generic free perturbation to save time:
            # common per-chunk of size N/len? approximate by block of 7
            Xp=Xcur.copy()
            noise_free=rng.randn(n,d)*amp
            # orbit-correlated part: repeat same noise within blocks of 7
            blk=7
            common=rng.randn((n+blk-1)//blk,d)*amp*0.7
            for i in range(n):
                noise_free[i]+=common[i//blk]
            Xp=Xp+noise_free
            Xp/=np.linalg.norm(Xp,axis=1,keepdims=True)
            steps=550 if amp<0.02 else 700
            Xq=_free_quench(Xp,steps=steps,beta0=22.0+ai*5.0,beta1=350.0+ai*60.0,lr0=0.008,lr1=0.0005,lam=0.05,t_lim=t_release_end)
            Xq=_hard_polish(Xq,steps=200)
            vq=_max_cos(Xq)
            if vq<cur_bestv:
                cur_bestv=vq; cur_best=Xq.copy(); Xcur=Xq.copy()
            elif vq< _max_cos(Xcur)+1e-4 and rng.rand()<0.3:
                # accept move to different basin occasionally
                Xcur=Xq.copy()
    best=cur_best; bestv=cur_bestv
    # final LP polish with remaining time
    remain=(t0+BUDGET-2.0)-time.time()
    if remain>4.0:
        try:
            best=_lp_polish(best,seconds=min(remain,14.0))
            bestv=_max_cos(best)
        except Exception:
            pass
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best

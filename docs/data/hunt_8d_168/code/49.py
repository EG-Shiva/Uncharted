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

def _build_E8():
    v = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    x = np.zeros(8); x[i]=si; x[j]=sj; v.append(x)
    for mask in range(256):
        s = np.ones(8); c = 0
        for k in range(8):
            if (mask>>k)&1:
                s[k]=-1.0; c+=1
        if c%2==0:
            v.append(0.5*s)
    E = np.array(v, float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E

def _quench(X, steps, b0, b1, lr0, lr1, mu, lam, t_lim):
    n, d = X.shape
    X = X.copy()
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    om = 1.0 - mu
    best_m = _max_cos(X)
    best = X.copy()
    for s in range(steps):
        if (s & 255)==0 and time.time()>t_lim:
            break
        f = s/max(steps-1,1)
        beta = np.exp(lb0+(lb1-lb0)*f)
        lr = np.exp(ll0+(ll1-ll0)*f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m; best = X.copy()
        E = np.exp(beta*(G-m))
        np.fill_diagonal(E, 0.0)
        Sg = E.sum(axis=1, keepdims=True)+1e-30
        Gg = (E @ X)/Sg
        # local spread with lower beta to keep rattlers moving
        bl = max(beta*0.25, 10.0)
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl*(G-mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True)+1e-30
        Gl = (W @ X)/Rs
        # weak positive background
        Gb = G.copy()
        np.maximum(Gb, 0.0, out=Gb)
        Gbg = (Gb @ X)*(2.0/max(n,1))
        Grad = Gg + 0.35*Gl*0.0 + lam*Gbg + 0.0*Gl
        # mix: global + lam*local? use lam for local too
        # actually blend local with fixed small weight
        Grad = Gg + 0.12*Gl + lam*Gbg
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        nn = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True))+1e-12
        sc = np.minimum(1.0, 3.0/nn)
        Gt = Gt*sc
        V = mu*V + om*Gt
        X = X - lr*V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best, best_m

def _hard_polish(X, steps, lr0, lr1, t0_, t1_, t_lim):
    X = X.copy()
    for s in range(steps):
        if (s & 127)==0 and time.time()>t_lim:
            break
        f = s/max(steps-1,1)
        tol = t0_+(t1_-t0_)*f
        lr = lr0*(lr1/lr0)**f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m-tol)).astype(float)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True)+1e-12
        Gg = (M @ X)/cnt
        dot = (Gg*X).sum(axis=1, keepdims=True)
        Gt = Gg-dot*X
        nn = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True))+1e-12
        act = (cnt[:,0]>0.5)
        St = np.zeros_like(Gt)
        St[act] = Gt[act]/nn[act]
        X = X - lr*St
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds, t_end):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n,1)
    itc = 0
    while time.time()-start < seconds and time.time() < t_end:
        if radius < 1e-14 or itc>40:
            break
        itc+=1
        Gf = x @ x.T
        cv = Gf[iu]
        m = cv.max()
        sel = cv > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p==0:
            radius*=0.5; continue
        if p>2400:
            idx = np.argsort(cv[sel])[-2400:]
            I, J = I[idx], J[idx]
            cc = cv[sel][idx]
        else:
            cc = cv[sel]
            p = len(I)
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
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius*=0.5
    return x, best

def _farthest_fill(kept, need, rng, n_try=2500):
    d = kept.shape[1]
    C = kept.copy()
    outs = []
    for _ in range(need):
        cand = rng.normal(size=(n_try,d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ C.T
        md = D.max(axis=1)
        bi = int(np.argmin(md))
        b = cand[bi]
        outs.append(b)
        C = np.vstack([C, b.reshape(1,-1)])
    if not outs:
        return kept
    return np.vstack([kept, np.array(outs)])

def _refine_hole(Y, p, iters=160, beta=110.0):
    p = p/np.linalg.norm(p)
    for it in range(iters):
        f = it/max(iters-1,1)
        lr = 0.12*(0.004/0.12)**f
        g = Y@p
        gm = g.max()
        w = np.exp(beta*(g-gm))
        s = w.sum()+1e-18
        A = (w@Y)/s
        At = A-float(A@p)*p
        p = p-lr*At
        nrm = np.linalg.norm(p)
        if nrm==0:
            break
        p/=nrm
    return p

def _crossover(A, B, rng):
    n, d = A.shape
    Q,_ = np.linalg.qr(rng.normal(size=(d,d)))
    if np.linalg.det(Q)<0:
        Q[:,0]*=-1
    Br = B @ Q.T
    u = rng.normal(size=d); u/=np.linalg.norm(u)
    r = rng.random()
    if r<0.45: thr=0.0
    elif r<0.75: thr=float(rng.uniform(-0.4,0.6))
    else: thr=float(rng.uniform(0.1,0.7))
    swap = rng.random()<0.5
    pa = A@u
    mask = pa>thr if not swap else pa<=thr
    child = np.empty_like(A)
    child[mask]=A[mask]
    # for complement take Br rows greedily far from already taken?
    # simple: take Br complement indices
    child[~mask]=Br[~mask]
    # dedup cos>0.88
    order = rng.permutation(n)
    kept=[]
    keepmask=np.zeros(n,bool)
    cur=[]
    for idx in order:
        v=child[idx]
        if not cur:
            keepmask[idx]=True; cur.append(v)
        else:
            md=float((np.array(cur)@v).max())
            if md>0.88:
                pass
            else:
                keepmask[idx]=True; cur.append(v)
    keptm=child[keepmask]
    need=n-keptm.shape[0]
    if need>0:
        child2=_farthest_fill(keptm,need,rng,n_try=2500)
    else:
        child2=keptm
    child2=child2+rng.normal(size=child2.shape)*1e-4
    child2/=np.linalg.norm(child2,axis=1,keepdims=True)
    return child2

def solve():
    t0=time.time()
    BUDGET=172.0
    rng=np.random.default_rng(7)
    n,d=168,8
    cands=[]
    try:
        b=np.load("best.npy").astype(float)
        if b.shape==(n,d):
            b/=np.linalg.norm(b,axis=1,keepdims=True)
            cands.append(b)
    except Exception:
        pass
    try:
        r=np.load("record.npy").astype(float)
        if r.shape==(n,d):
            r/=np.linalg.norm(r,axis=1,keepdims=True)
            cands.append(r)
    except Exception:
        pass
    if not cands:
        c=rng.normal(size=(n,d)); c/=np.linalg.norm(c,axis=1,keepdims=True)
        cands.append(c)
    best_x=min(cands,key=_max_cos).copy()
    best_val=_max_cos(best_x)
    E8=None
    try:
        E8=_build_E8()
    except Exception:
        E8=None
    # quick LP tighten baseline
    if time.time()-t0 < BUDGET-60:
        try:
            bx,bv=_lp_polish(best_x,2.0,t_end=t0+BUDGET-20)
            if bv<best_val:
                best_x,best_val=bx,bv
        except Exception:
            pass
    archive=[(best_x.copy(),best_val)]
    def push(x,v):
        nonlocal best_x,best_val
        if v<best_val:
            best_x=x.copy(); best_val=v
            archive.append((x.copy(),v))
        else:
            if len(archive)<10 and v<best_val+0.008:
                # diversity: min dot distance
                archive.append((x.copy(),v))
        if len(archive)>12:
            # drop worst
            wi=max(range(len(archive)),key=lambda i: archive[i][1])
            if archive[wi][1]>best_val+1e-12:
                archive.pop(wi)
    it=0
    while time.time()-t0 < BUDGET-26:
        it+=1
        mode=rng.random()
        base=best_x
        if len(archive)>=2 and rng.random()<0.45:
            ai,bi=rng.choice(len(archive),size=2,replace=False)
            A=archive[ai][0]; B=archive[bi][0]
            try:
                x0=_crossover(A,B,rng)
            except Exception:
                x0=base+rng.normal(size=base.shape)*0.15
                x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        elif mode<0.35:
            sig=float(np.exp(rng.uniform(np.log(0.05),np.log(0.35))))
            x0=base+rng.normal(size=base.shape)*sig
            x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        elif mode<0.60:
            # eject rattlers to holes
            try:
                pm=_per_max(base)
                k=int(rng.integers(6,20))
                kill=np.argsort(pm)[:k]  # smallest per-max = freest
                m=np.ones(n,bool); m[kill]=False
                kept=base[m]
                # find holes w.r.t kept
                npr=6000
                P=rng.normal(size=(npr,d)); P/=np.linalg.norm(P,axis=1,keepdims=True)
                D=P@kept.T
                md=D.max(axis=1)
                order=np.argsort(md)
                holes=[]
                for idx in order:
                    p=P[idx]
                    ok=True
                    for q in holes:
                        if abs(float(p@q))>0.85:
                            ok=False; break
                    if ok:
                        Ye=kept if not holes else np.vstack([kept,np.array(holes)])
                        pr=_refine_hole(Ye,p,iters=140,beta=90+rng.random()*60)
                        holes.append(pr)
                        if len(holes)>=k:
                            break
                if len(holes)<k:
                    # fill randomly
                    while len(holes)<k:
                        p=rng.normal(size=d); p/=np.linalg.norm(p)
                        holes.append(p)
                x0=np.vstack([kept,np.array(holes)])
                # shuffle order
                x0=x0[rng.permutation(n)]
                x0/=np.linalg.norm(x0,axis=1,keepdims=True)
            except Exception:
                x0=base+rng.normal(size=base.shape)*0.12
                x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        elif mode<0.75 and E8 is not None:
            idx=rng.choice(240,size=n,replace=False)
            x0=E8[idx]+rng.normal(size=(n,d))*float(rng.uniform(0.02,0.12))
            x0/=np.linalg.norm(x0,axis=1,keepdims=True)
        else:
            c=rng.normal(size=(n,d)); c/=np.linalg.norm(c,axis=1,keepdims=True)
            # mix a bit of best to guide
            if rng.random()<0.5:
                k=int(rng.integers(40,120))
                sel=rng.choice(n,size=k,replace=False)
                c[sel]=base[sel]+rng.normal(size=(k,d))*0.05
                c/=np.linalg.norm(c,axis=1,keepdims=True)
            x0=c
        steps=int(rng.choice([1200,1600,2200]))
        b0=float(rng.uniform(18,55)); b1=float(rng.uniform(150,340))
        lr0=float(np.exp(rng.uniform(np.log(0.03),np.log(0.12))))
        lr1=float(np.exp(rng.uniform(np.log(0.001),np.log(0.012))))
        mu=float(rng.uniform(0.75,0.90))
        lam=float(rng.uniform(0.0,0.06))
        t_lim=t0+BUDGET-24
        try:
            xq,vq0=_quench(x0,steps,b0,b1,lr0,lr1,mu,lam,t_lim)
        except Exception:
            continue
        # hard polish tail
        try:
            xq2=_hard_polish(xq,steps=220,lr0=0.0025,lr1=0.00012,t0_=2e-4,t1_=1e-6,t_lim=t0+BUDGET-24)
            vq=_max_cos(xq2)
            if vq<vq0:
                xq=xq2; vq0=vq
        except Exception:
            pass
        vq=vq0
        if vq<best_val+0.002 or rng.random()<0.08:
            try:
                xp,vp=_lp_polish(xq,1.0,t_end=t0+BUDGET-22)
            except Exception:
                xp,vp=xq,vq
            push(xp,vp)
        else:
            push(xq,vq)
        if time.time()-t0>BUDGET-26:
            break
    # final squeeze best
    try:
        bx,bv=_lp_polish(best_x,12.0,t_end=t0+BUDGET-1.5)
        if bv<best_val:
            best_x,best_val=bx,bv
    except Exception:
        pass
    try:
        best_x=best_x/np.linalg.norm(best_x,axis=1,keepdims=True)
        # tiny shrink guard against fp
        G=best_x@best_x.T
        np.fill_diagonal(G,-2.0)
    except Exception:
        pass
    return best_x

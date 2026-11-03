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

def _degs(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > (m - tol)
    deg = C.sum(axis=1).astype(int)
    return deg, m, G

def _refine_hole(Y, p, iters=160, beta=90.0, lr0=0.12, lr1=0.004):
    p = p / (np.linalg.norm(p)+1e-18)
    for it in range(iters):
        frac = it/max(iters-1,1)
        lr = lr0*(lr1/lr0)**frac
        g = Y @ p
        gm = g.max()
        w = np.exp(beta*(g-gm))
        s = w.sum()+1e-18
        A = (w @ Y)/s
        At = A - float(A @ p)*p
        p = p - lr*At
        n = np.linalg.norm(p)
        if n == 0:
            break
        p /= n
    return p

def _masked_adam(X, free, steps, beta0, beta1, lr0, lr1, lam_bg=0.0):
    n,d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1=0.9; b2=0.999; eps=1e-8
    lb0=np.log(beta0); lb1=np.log(beta1)
    ll0=np.log(lr0); ll1=np.log(lr1)
    free = free.astype(bool)
    for t in range(1, steps+1):
        frac = t/steps
        beta = np.exp(lb0+(lb1-lb0)*frac)
        lr = np.exp(ll0+(ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)+1e-18
        Grad = (W @ X)/S
        if lam_bg>0:
            Gb = X @ X.T
            np.fill_diagonal(Gb, 0.0)
            np.maximum(Gb,0.0,out=Gb)
            Grad = Grad + lam_bg*((Gb @ X)*(2.0/n))
        dot = (Grad*X).sum(axis=1,keepdims=True)
        Gt = Grad - dot*X
        Gt[~free,:]=0.0
        m *= b1; m += (1.0-b1)*Gt
        v *= b2; v += (1.0-b2)*(Gt*Gt)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(m/bc1)/(np.sqrt(v/bc2)+eps)
        step[~free,:]=0.0
        X = X - lr*step
        X /= np.sqrt((X*X).sum(axis=1,keepdims=True)).reshape(-1,1)
    return X

def _adam_free(X, steps, beta0, beta1, lr0, lr1, lam_bg=0.0):
    n,d=X.shape
    m=np.zeros_like(X); v=np.zeros_like(X)
    b1=0.9;b2=0.999;eps=1e-8
    lb0=np.log(beta0);lb1=np.log(beta1)
    ll0=np.log(lr0);ll1=np.log(lr1)
    for t in range(1,steps+1):
        frac=t/steps
        beta=np.exp(lb0+(lb1-lb0)*frac)
        lr=np.exp(ll0+(ll1-ll0)*frac)
        G=X@X.T
        np.fill_diagonal(G,-10.0)
        gmax=G.max()
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        S=W.sum(axis=1,keepdims=True)+1e-18
        Grad=(W@X)/S
        if lam_bg>0:
            Gb=X@X.T
            np.fill_diagonal(Gb,0.0)
            np.maximum(Gb,0.0,out=Gb)
            Grad=Grad+lam_bg*((Gb@X)*(2.0/n))
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        m*=b1; m+=(1.0-b1)*Gt
        v*=b2; v+=(1.0-b2)*(Gt*Gt)
        bc1=1.0-b1**t; bc2=1.0-b2**t
        step=(m/bc1)/(np.sqrt(v/bc2)+eps)
        X=X-lr*step
        X/=np.sqrt((X*X).sum(axis=1,keepdims=True)).reshape(-1,1)
    return X

def _hard_polish(X, steps=200, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=1e-6):
    for s in range(steps):
        frac=s/max(steps-1,1)
        tol=tol0+(tol1-tol0)*frac
        lr=lr0*(lr1/lr0)**frac
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        m=float(G.max())
        M=(G>(m-tol)).astype(np.float64)
        np.fill_diagonal(M,0.0)
        cnt=M.sum(axis=1,keepdims=True)+1e-12
        act=(cnt[:,0]>0.5)
        if not np.any(act):
            continue
        Grad=(M@X)/cnt
        dot=(Grad*X).sum(axis=1,keepdims=True)
        Gt=Grad-dot*X
        nn=np.sqrt((Gt*Gt).sum(axis=1,keepdims=True))+1e-12
        step=np.zeros_like(Gt)
        step[act]=Gt[act]/nn[act]
        X=X-lr*step
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
        m=float(c.max())
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            radius*=0.5
            continue
        if p>2500:
            idx=np.argsort(c[sel])[-2500:]
            I,J=I[idx],J[idx]
            cc=c[sel][idx]
        else:
            cc=c[sel]
            p=len(I)
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj=np.zeros(nv);obj[-1]=1.0
        try:
            res=linprog(obj,A_ub=A_ub,b_ub=-cc,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        vv=_max_cos(y)
        if vv<best:
            x,best, radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    BUDGET=175.0
    n,d=168,8
    rng=np.random.RandomState(12345)
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape!=(n,d):
                continue
            a=a/np.linalg.norm(a,axis=1,keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va=_max_cos(a)
            if va<best_val:
                best_val=va; best=a.copy()
        except Exception:
            pass
    if best is None:
        best=rng.randn(n,d); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    base_best=best.copy()

    def build_block(X, k, rng):
        deg,m,G=_degs(X, tol=1e-9)
        ratt=np.where(deg==0)[0]
        # hub candidates deg>=8 else top
        order=np.argsort(-deg)
        hubs=np.where(deg>=8)[0]
        if len(hubs)<8:
            hubs=order[:max(16,len(order))]
        nhub=int(8+rng.randint(9))  # 8..16
        nhub=min(nhub, len(hubs), max(8,k//3))
        sel_hubs=rng.choice(hubs, size=nhub, replace=False) if len(hubs)>=nhub else hubs
        neigh=set()
        for h in sel_hubs:
            row=G[h].copy()
            row[h]=-2
            nn=np.argsort(-row)[:2+rng.randint(2)]  # 2..3 neighbours
            for q in nn:
                neigh.add(int(q))
        neigh=np.array(list(neigh),dtype=int) if len(neigh)>0 else np.array([],dtype=int)
        # assemble
        block_set=set([int(x) for x in sel_hubs]+[int(x) for x in neigh])
        # add rattlers
        ratt_list=[int(x) for x in ratt if int(x) not in block_set]
        rng.shuffle(ratt_list)
        for x in ratt_list:
            if len(block_set)>=k:
                break
            block_set.add(x)
        # if still short, add most jammed (high per-max)
        if len(block_set)<k:
            pm=_per_max(X)
            ord2=np.argsort(-pm)
            for x in ord2:
                if int(x) not in block_set:
                    block_set.add(int(x))
                    if len(block_set)>=k:
                        break
        # if too big, drop excess (prefer keep hubs+neigh, drop random rattlers)
        blk=np.array(list(block_set),dtype=int)
        if len(blk)>k:
            # keep sel_hubs+neigh mandatory, subsample rest
            mandatory=set([int(x) for x in sel_hubs]+[int(x) for x in neigh])
            mand=np.array(list(mandatory),dtype=int)
            rest=np.array([x for x in blk if x not in mandatory],dtype=int)
            need=k-len(mand)
            if need<0:
                # too many mandatory, random subset
                mand=rng.choice(mand,size=k,replace=False)
                blk=mand
            else:
                if len(rest)>need:
                    rest=rng.choice(rest,size=need,replace=False)
                blk=np.concatenate([mand,rest])
        # final adjust to exactly k
        if len(blk)>k:
            blk=rng.choice(blk,size=k,replace=False)
        elif len(blk)<k:
            pool=np.setdiff1d(np.arange(n),blk)
            extra=rng.choice(pool,size=k-len(blk),replace=False)
            blk=np.concatenate([blk,extra])
        return blk

    trial=0
    ks=[32,48,64]
    # precompute probe budget per trial adaptive to remaining time
    while time.time()-t0 < BUDGET-12:
        trial+=1
        k=ks[(trial-1)%len(ks)]
        # vary tol slightly every few trials to change stratum
        # use current best as source (occasionally base_best)
        src=best.copy()
        try:
            blk=build_block(src, k, rng)
        except Exception:
            continue
        core_idx=np.setdiff1d(np.arange(n),blk)
        core=src[core_idx].copy()
        # probe pool 50000
        try:
            P=rng.randn(50000,d)
            P/=np.linalg.norm(P,axis=1,keepdims=True)
        except Exception:
            continue
        Y=core.copy()  # (n-k) x d growing
        inserted=[]
        # sequential insertion
        CH=5000
        ok=True
        for s_idx in range(k):
            if time.time()-t0>BUDGET-10:
                ok=False; break
            # score probes vs Y
            best_s=1e9; best_p=None
            # chunked
            for a in range(0,50000,CH):
                b=min(a+CH,50000)
                D=P[a:b]@Y.T
                mx=D.max(axis=1)
                j=np.argmin(mx)
                if float(mx[j])<best_s:
                    best_s=float(mx[j]); best_p=P[a+j].copy()
            # refine hole against Y
            beta_r=80.0+rng.rand()*60.0
            best_p=_refine_hole(Y,best_p,iters=160,beta=beta_r)
            inserted.append(best_p)
            Y=np.vstack([Y,best_p[None,:]])
            # optional: replace probe to avoid reuse (not needed)
        if not ok:
            break
        inserted=np.array(inserted)  # k x d
        Xnew=np.empty_like(src)
        Xnew[core_idx]=core
        Xnew[blk]=inserted
        # boundary: core points with dot>0.3 to any inserted
        try:
            Dcb=core@inserted.T
            bmask=(Dcb.max(axis=1)>0.3)
        except Exception:
            bmask=np.zeros(len(core_idx),dtype=bool)
        free=np.zeros(n,dtype=bool)
        free[blk]=True
        # map bmask to global
        for li,gi in enumerate(core_idx):
            if bmask[li]:
                free[gi]=True
        # phase 1: masked anneal 1000 steps deep core fixed
        beta0=25.0+rng.rand()*25.0
        beta1=260.0+rng.rand()*340.0
        lr0=0.004+rng.rand()*0.004
        lr1=lr0*(0.05+rng.rand()*0.08)
        lam=0.0 if rng.rand()<0.6 else 0.05+rng.rand()*0.1
        # rownorm variant via masked_adam (row-normalized)
        Xnew=_masked_adam(Xnew, free, steps=1000, beta0=beta0, beta1=beta1, lr0=lr0, lr1=lr1, lam_bg=lam)
        if time.time()-t0>BUDGET-10:
            v=_max_cos(Xnew)
            if v<best_val:
                best=Xnew.copy(); best_val=v
            break
        # full thaw
        beta0b=30.0+rng.rand()*30.0
        beta1b=300.0+rng.rand()*300.0
        lr0b=0.002+rng.rand()*0.003
        lr1b=lr0b*(0.05+rng.rand()*0.1)
        Xnew=_adam_free(Xnew, steps=600, beta0=beta0b, beta1=beta1b, lr0=lr0b, lr1=lr1b, lam_bg=lam*0.5)
        Xnew=_hard_polish(Xnew, steps=200, lr0=0.003, lr1=0.00012)
        # quick check before expensive LP
        vq=_max_cos(Xnew)
        # LP polish with remaining-aware budget
        remain=BUDGET-(time.time()-t0)
        lp_sec=min(4.0, max(1.0, remain/8.0))
        # only LP if promising (within 3e-4 of best) to save time
        if vq < best_val+3e-4:
            Xnew=_lp_polish(Xnew, seconds=lp_sec)
            vq=_max_cos(Xnew)
        if vq<best_val and np.all(np.isfinite(Xnew)):
            best=Xnew.copy(); best_val=vq
        # early exit if big win? continue to use new best
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    return best

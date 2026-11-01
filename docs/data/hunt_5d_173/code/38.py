import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[-3500:]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[sel]; J = iu1[sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _repel(x, alphas, iters_list, lrs, mu, decay, deadline=None):
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for it in range(int(iters)):
            if deadline is not None and (it % 50 == 0) and time.time() > deadline:
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                return x
            G = x @ x.T
            # shift for stability
            W = np.exp(alpha*(G-0.758))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True)+1e-300
            A = (W @ x)/rs
            dot = (A*x).sum(axis=1, keepdims=True)
            Ft = A - dot*x
            v = mu*v + (1.0-mu)*Ft
            x = x - lr*v
            nrm = np.linalg.norm(x, axis=1, keepdims=True)
            nrm[nrm==0]=1.0
            x = x/nrm
            lr *= decay
    return x

def _void_fill(rem, k, rng, n_cand=4000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand,d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    smax = (cand @ rem.T).max(axis=1)
    out=[]
    for _ in range(k):
        idx=int(np.argmin(smax))
        pt=cand[idx].copy()
        out.append(pt)
        upd=cand @ pt
        np.maximum(smax, upd, out=smax)
        smax[idx]=2.0
    return np.stack(out,axis=0)

def _contact_counts(X, m=None):
    G=X@X.T
    np.fill_diagonal(G,-2.0)
    if m is None:
        m=float(G.max())
    cnt=(G>=m-1e-9).sum(axis=1)
    mi=G.max(axis=1)
    return m,cnt,mi,G

def solve():
    t0=time.time()
    BUDGET=172.0
    deadline=t0+BUDGET
    rng=np.random.default_rng(12345)
    n,d=173,5
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape==(n,d):
                an=a/np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(an)
                if v<best_val:
                    best_val=v; best=an.copy()
        except Exception:
            pass
    if best is None:
        best=rng.standard_normal((n,d)); best/=np.linalg.norm(best,axis=1,keepdims=True); best_val=_max_cos(best)
    trial=0
    # quick diverse smooth restarts interleaved with targeted hops
    while True:
        rem=deadline-time.time()
        if rem<8.0:
            break
        pt=min(1.5, max(0.5, rem-7.0))
        # decide mutation
        m,cnt,mi,_=_contact_counts(best)
        order_desc=np.argsort(-mi)  # per-point worst cosine
        order_cnt=np.argsort(-cnt)
        y=best.copy()
        mode=trial%8
        if mode==0:
            # small global shake
            y=y+0.012*rng.standard_normal(y.shape)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[40,120],[150,200],[0.05,0.02],0.6,0.998,deadline-6)
        elif mode==1:
            # medium shake focused on jammed + neighbours
            k=int(4+rng.integers(0,4))
            pool=order_cnt[:25]
            sel=rng.choice(pool,size=min(k,len(pool)),replace=False)
            # add randoms
            extra=rng.choice(n,size=3,replace=False)
            sel=np.unique(np.concatenate([sel,extra]))
            y[sel]+=0.08*rng.standard_normal((len(sel),d))
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[30,90,200],[120,150,150],[0.07,0.035,0.015],0.65,0.998,deadline-6)
        elif mode==2:
            # overload removal + void reinsert
            k=int(2+rng.integers(0,5))  # 2..6
            victims=np.unique(order_desc[:max(10,k*3)])
            # choose k among top
            sel=rng.choice(victims,size=k,replace=False)
            mask=np.ones(n,bool); mask[sel]=False
            rem_pts=y[mask]
            try:
                newp=_void_fill(rem_pts,k,rng,n_cand=4000)
            except Exception:
                newp=rng.standard_normal((k,d)); newp/=np.linalg.norm(newp,axis=1,keepdims=True)
            y[sel]=newp
            y=y+0.008*rng.standard_normal(y.shape)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[50,150,250],[120,150,200],[0.06,0.03,0.015],0.7,0.998,deadline-6)
        elif mode==3:
            # spherical cap removal
            u=rng.standard_normal(d); u/=np.linalg.norm(u)
            proj=y@u
            # cap threshold to get 4..12 points
            thr=np.sort(proj)[-int(4+rng.integers(0,9))]
            sel=np.where(proj>=thr)[0]
            if len(sel)>14:
                sel=rng.choice(sel,size=14,replace=False)
            mask=np.ones(n,bool); mask[sel]=False
            rem_pts=y[mask]
            try:
                newp=_void_fill(rem_pts,len(sel),rng,n_cand=4000)
            except Exception:
                newp=rng.standard_normal((len(sel),d)); newp/=np.linalg.norm(newp,axis=1,keepdims=True)
            y[sel]=newp
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[30,100,220],[150,180,220],[0.08,0.035,0.015],0.7,0.998,deadline-6)
        elif mode==4:
            # rattler injection: move rattlers/near-rattlers into densest void near jammed
            ratt=np.where(cnt<=1)[0]
            if len(ratt)==0:
                ratt=np.argsort(cnt)[:2]
            # also take worst points
            sel=np.unique(np.concatenate([ratt, order_desc[:2]]))[:5]
            mask=np.ones(n,bool); mask[sel]=False
            rem_pts=y[mask]
            try:
                newp=_void_fill(rem_pts,len(sel),rng,n_cand=5000)
            except Exception:
                newp=rng.standard_normal((len(sel),d)); newp/=np.linalg.norm(newp,axis=1,keepdims=True)
            y[sel]=newp+0.01*rng.standard_normal(newp.shape)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[60,160],[150,250],[0.05,0.018],0.65,0.998,deadline-6)
        elif mode==5:
            # large escape shake
            y=y+0.28*rng.standard_normal(y.shape)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[25,80,180],[150,200,250],[0.09,0.04,0.018],0.75,0.998,deadline-6)
        elif mode==6:
            # fresh random + anneal (different basin)
            y=rng.standard_normal((n,d)); y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[20,60,140,260],[200,250,250,250],[0.10,0.05,0.025,0.012],0.7,0.9985,deadline-6)
        else:
            # antipodal-ish perturbation of random subset
            k=int(8+rng.integers(0,8))
            sel=rng.choice(n,size=k,replace=False)
            y[sel]=-y[sel]+0.1*rng.standard_normal((k,d))
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            y=_repel(y,[40,130],[150,220],[0.06,0.02],0.65,0.998,deadline-6)
        # LP polish
        if time.time()>deadline-7:
            trial+=1
            continue
        y=_lp_polish(y, pt)
        v=_max_cos(y)
        if v<best_val-1e-12:
            best=y; best_val=v
        trial+=1
    # final long polish of incumbent with remaining time
    rem=deadline-time.time()
    if rem>1.0:
        best=_lp_polish(best, min(rem-1.0, 10.0))
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best

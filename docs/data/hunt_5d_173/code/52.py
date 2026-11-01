import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if seconds <= 0.03:
        return x
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt_sel = int(sel.sum())
        if cnt_sel == 0:
            radius *= 0.5
            continue
        if cnt_sel > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]; cs = c[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _repel(x, alphas, iters_list, lrs, mu=0.7, decay=0.998):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for _ in range(int(iters)):
            G = x @ x.T
            W = np.exp(alpha*(G-0.75))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True)+1e-300
            A = (W @ x)/rs
            dot = (A*x).sum(axis=1, keepdims=True)
            Ft = A - dot*x
            v = mu*v + (1.0-mu)*Ft
            x = x - lr*v
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            lr *= decay
    return x

def _quick_repel(x, rng):
    a1 = 30.0+rng.random()*20.0
    a2 = 80.0+rng.random()*60.0
    a3 = 170.0+rng.random()*130.0
    l1 = 0.06+rng.random()*0.08
    l2 = 0.03+rng.random()*0.04
    l3 = 0.012+rng.random()*0.02
    mu = 0.5+rng.random()*0.35
    dec = 0.997+rng.random()*0.002
    if dec>0.9995: dec=0.9995
    return _repel(x,[a1,a2,a3],[120,150,200],[l1,l2,l3],mu,dec)

def _void_candidates(existing, k, rng, n_cand=20000):
    d = existing.shape[1]
    cand = rng.standard_normal((n_cand,d))
    cand /= np.linalg.norm(cand,axis=1,keepdims=True)
    s = (cand @ existing.T).max(axis=1)
    chosen=[]
    for _ in range(k):
        idx=int(np.argmin(s))
        pt=cand[idx].copy()
        chosen.append(pt)
        upd=cand@pt
        np.maximum(s,upd,out=s)
        s[idx]=2.0
    return np.stack(chosen,axis=0)

def _contact_sig(x):
    n=x.shape[0]
    G=x@x.T
    np.fill_diagonal(G,-2.0)
    m=float(G.max())
    mask=G>=m-1e-9
    cnt=mask.sum(axis=1)
    e=int(mask.sum()//2)
    return m,int(cnt.min()),int(cnt.max()),e,tuple(sorted(cnt.tolist()))

def _overload_idx(x):
    G=x@x.T
    np.fill_diagonal(G,-2.0)
    m=float(G.max())
    mask=G>=m-1e-9
    cnt=mask.sum(axis=1)
    mx=G.max(axis=1)
    order=np.lexsort((mx,-cnt))
    return int(order[-1])

def _recombine(A,B,rng):
    n,d=A.shape
    # init rotation from random orthogonal
    Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    R=Q
    # iterative Procrustes + assignment
    for _ in range(4):
        Br=B@R
        cost=-(A@Br.T)
        _,col=linear_sum_assignment(cost)
        # Kabsch: Q ordered, P=A
        Qm=Br[col]
        M=Qm.T@A
        U,_,Vt=np.linalg.svd(M)
        R=U@Vt
    Br=B@R
    cost=-(A@Br.T)
    _,col=linear_sum_assignment(cost)
    Bord=Br[col]
    # complementary subsets
    mask=rng.random(n)<0.5
    if mask.sum()==0: mask[rng.integers(0,n)]=True
    if mask.sum()==n: mask[rng.integers(0,n)]=False
    C=np.empty_like(A)
    C[mask]=A[mask]
    C[~mask]=Bord[~mask]
    C=C+0.008*rng.standard_normal(C.shape)
    C/=np.linalg.norm(C,axis=1,keepdims=True)
    return C

def solve():
    t0=time.time()
    BUDGET=172.0
    deadline=t0+BUDGET
    rng=np.random.default_rng(20261101)
    nT,dT=173,5
    best=None; best_val=1e9
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(np.float64)
            if a.shape==(nT,dT):
                an=a/np.linalg.norm(a,axis=1,keepdims=True)
                v=_max_cos(an)
                if v<best_val:
                    best_val=v; best=an.copy()
        except Exception:
            pass
    if best is None:
        best=rng.standard_normal((nT,dT)); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    archive=[(best_val,best.copy())]
    sigs=set()
    try:
        _,mn,mx,e,_=_contact_sig(best)
        sigs.add((mn,mx,e))
    except Exception:
        pass

    def add_to_archive(x):
        nonlocal archive
        v=_max_cos(x)
        try:
            _,mn,mx,e,pat=_contact_sig(x)
        except Exception:
            return v
        key=(mn,e,pat)
        # diversity: allow if new pattern or better than worst
        if key in sigs:
            # only keep if better than worst
            worst=max(z[0] for z in archive)
            if v>=worst-1e-12:
                return v
        sigs.add(key)
        archive.append((v,x.copy()))
        # keep size <=12, drop worst
        archive=sorted(archive,key=lambda z:z[0])
        if len(archive)>12:
            archive=archive[:12]
        return v

    def time_left():
        return deadline-time.time()

    # Phase A: particle-number homotopy N=168..172
    for N in (168,169,170,171,172):
        for rep in range(2):
            if time_left()<25: break
            # init small code
            if rng.random()<0.5 and len(archive)>0:
                # derive from random archive member by deletion
                src=archive[rng.integers(0,len(archive))][1]
                # if src is 173, delete to N
                idx=np.arange(src.shape[0])
                rng.shuffle(idx)
                # delete most overloaded to diversify sometimes
                if rng.random()<0.5:
                    x=src.copy()
                    while x.shape[0]>N:
                        k=_overload_idx(x)
                        x=np.delete(x,k,axis=0)
                else:
                    keep=np.sort(idx[:N])
                    x=src[keep].copy()
                x=x+rng.standard_normal(x.shape)*0.02
                x/=np.linalg.norm(x,axis=1,keepdims=True)
            else:
                x=rng.standard_normal((N,dT)); x/=np.linalg.norm(x,axis=1,keepdims=True)
            x=_quick_repel(x,rng)
            x=_lp_polish(x,0.7)
            # grow one at a time to 173 with void insertion + repel + LP
            ok=True
            while x.shape[0]<nT:
                if time_left()<18:
                    ok=False; break
                need=1
                pts=_void_candidates(x,need,rng,n_cand=20000)
                x=np.concatenate([x,pts],axis=0)
                # light repel + LP after each insertion
                x=_repel(x,[60.0,150.0],[80,120],[0.05,0.02],0.65,0.998)
                x=_lp_polish(x,0.5)
            if not ok: continue
            v=_max_cos(x)
            add_to_archive(x)
            if v<best_val:
                best_val=v; best=x.copy()

    # Phase B: N=174,175 then delete most overloaded
    for Nextra in (174,175):
        for rep in range(2):
            if time_left()<16: break
            src=best.copy()
            # add void points
            pts=_void_candidates(src,Nextra-nT,rng,n_cand=20000)
            x=np.concatenate([src,pts],axis=0)
            x=_quick_repel(x,rng)
            x=_lp_polish(x,0.9)
            # delete most overloaded one at a time down to 173
            while x.shape[0]>nT:
                k=_overload_idx(x)
                x=np.delete(x,k,axis=0)
            x=_repel(x,[80.0,180.0],[100,150],[0.04,0.015],0.7,0.998)
            x=_lp_polish(x,0.7)
            v=_max_cos(x)
            add_to_archive(x)
            if v<best_val:
                best_val=v; best=x.copy()

    # Phase C: recombination of distinct optima
    it=0
    while time_left()>9.0:
        if len(archive)<2:
            # mutate best by void reinsertion
            x=best.copy()
            G=x@x.T; np.fill_diagonal(G,-2.0)
            mi=G.max(axis=1); order=np.argsort(-mi)
            k=int(2+rng.integers(0,6))
            sel=order[:k] if rng.random()<0.7 else rng.choice(nT,k,replace=False)
            mask=np.ones(nT,bool); mask[sel]=False
            rem=x[mask]
            try:
                pts=_void_candidates(rem,k,rng,n_cand=20000)
            except Exception:
                pts=rng.standard_normal((k,dT)); pts/=np.linalg.norm(pts,axis=1,keepdims=True)
            x[sel]=pts
            x=_quick_repel(x,rng)
            x=_lp_polish(x,1.0)
        else:
            # pick two distinct (prefer different min-contact)
            archive_sorted=sorted(archive,key=lambda z:z[0])
            # non-greedy: sometimes pick not the best
            if rng.random()<0.7:
                i=rng.integers(0,min(4,len(archive_sorted)))
                j=rng.integers(0,len(archive_sorted))
                if j==i: j=(j+1)%len(archive_sorted)
            else:
                i=rng.integers(0,len(archive_sorted)); j=rng.integers(0,len(archive_sorted))
                if j==i: j=(j+1)%len(archive_sorted)
            A=archive_sorted[i][1]; Bc=archive_sorted[j][1]
            try:
                x=_recombine(A,Bc,rng)
            except Exception:
                x=A.copy()+0.02*rng.standard_normal(A.shape)
                x/=np.linalg.norm(x,axis=1,keepdims=True)
            # repair: anneal + repel + LP
            x=_repel(x,[50.0,140.0],[60,120],[0.05,0.02],0.6,0.997)
            x=_lp_polish(x,0.9)
            # occasional void repair of worst point
            if rng.random()<0.3:
                k=_overload_idx(x)
                # propose void replacement for that point
                mask=np.ones(nT,bool); mask[k]=False
                rem=x[mask]
                try:
                    pts=_void_candidates(rem,1,rng,n_cand=20000)
                    x2=x.copy(); x2[k]=pts[0]
                    x2=_lp_polish(x2,0.4)
                    if _max_cos(x2)<_max_cos(x):
                        x=x2
                except Exception:
                    pass
        v=_max_cos(x)
        add_to_archive(x)
        # refresh archive variable order (add_to_archive uses closure list; re-sort)
        # archive maintained inside function via nonlocal; ensure sorted trim
        archive_sorted2=sorted(archive,key=lambda z:z[0])
        archive=archive_sorted2[:12]
        if v<best_val-1e-12:
            best_val=v; best=x.copy()
        it+=1
        if it>60:
            break

    rem=time_left()
    if rem>1.5:
        best=_lp_polish(best,min(rem-0.7,6.0))
        best_val=_max_cos(best)
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    return best

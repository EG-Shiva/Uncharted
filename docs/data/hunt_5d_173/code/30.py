import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, radius_init=2e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0.03:
        return X, best
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt_sel = int(sel.sum())
        if cnt_sel == 0:
            radius *= 0.5
            continue
        if cnt_sel > 2800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[-2800:]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate([(I[:, None]*d+np.arange(d)),
                               (J[:, None]*d+np.arange(d)),
                               np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn==0]=1.0
        y = y/yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def _graph_vec(X, m, tol=1e-7):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    return (G[iu] >= m - tol)

def _hamming(a,b):
    return int(np.count_nonzero(a ^ b))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 173, 5
    POOL = 12
    # load seeds
    bases = []
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape==(n,d):
                a = a/np.linalg.norm(a,axis=1,keepdims=True)
                bases.append(a)
        except Exception:
            pass
    if not bases:
        a = rng.normal(size=(n,d)); a/=np.linalg.norm(a,axis=1,keepdims=True)
        bases.append(a)
    seed0 = bases[0].copy()
    v0 = _max_cos(seed0)
    pool_X=[]; pool_V=[]; pool_G=[]
    # first member = record polished very briefly (already optimal)
    pool_X.append(seed0.copy()); pool_V.append(v0)
    pool_G.append(_graph_vec(seed0,v0))
    if len(bases)>1:
        v1=_max_cos(bases[1])
        if abs(v1-v0)>1e-12:
            pool_X.append(bases[1].copy()); pool_V.append(v1)
            pool_G.append(_graph_vec(bases[1],v1))
    # init rest of pool with perturbations + short polish
    sigmas=[0.03,0.06,0.10,0.15,0.22,0.32,0.05,0.12,0.20,0.28,0.08]
    si=0
    while len(pool_X)<POOL and time.time()<deadline-120:
        sig=sigmas[si%len(sigmas)]; si+=1
        y=seed0.copy()
        if rng.random()<0.3:
            y=rng.normal(size=(n,d)); y/=np.linalg.norm(y,axis=1,keepdims=True)
            # partial keep: mix half record half random to be less bad
            k=n//2
            idx=rng.choice(n,k,replace=False)
            y[idx]=seed0[idx]
            y/=np.linalg.norm(y,axis=1,keepdims=True)
        else:
            y=y+sig*rng.normal(size=(n,d))
            y/=np.linalg.norm(y,axis=1,keepdims=True)
        y2,v2=_lp_polish(y,1.2,radius_init=5e-3)
        g2=_graph_vec(y2,v2)
        # diversity check
        ok=True
        for g in pool_G:
            if _hamming(g2,g)<20 and v2>=min(pool_V):
                # check if beats that member
                # find member index
                pass
        # allow anyway for init to fill pool, but avoid exact dup
        dup=False
        for g in pool_G:
            if _hamming(g2,g)<5:
                dup=True; break
        if dup:
            continue
        pool_X.append(y2); pool_V.append(v2); pool_G.append(g2)
    pool_V=np.array(pool_V,float)
    order=np.argsort(pool_V)
    pool_X=[pool_X[i] for i in order]; pool_V=[float(pool_V[i]) for i in order]; pool_G=[pool_G[i] for i in order]
    best_idx=int(np.argmin(pool_V))
    gbest_X=pool_X[best_idx].copy(); gbest_V=float(pool_V[best_idx])
    trial=0
    iu_full=np.triu_indices(n,1)
    while True:
        rem=deadline-time.time()
        if rem<8.0:
            break
        elapsed=time.time()-t0
        prog=min(1.0,max(0.0,elapsed/BUDGET))
        T=2e-4*((1e-6/2e-4)**prog)  # 2e-4 -> 1e-6
        # polish time per candidate annealed: larger early
        pt=1.1 if rem>40 else (0.7 if rem>15 else 0.45)
        pt=min(pt,rem-7.0)
        if pt<0.35:
            break
        do_recomb=(trial%10>=6) and len(pool_X)>=2
        if do_recomb:
            # pick two parents biased to good but diverse
            a_idx=int(rng.integers(len(pool_X)))
            b_idx=int(rng.integers(len(pool_X)))
            if b_idx==a_idx:
                b_idx=(b_idx+1)%len(pool_X)
            Xa=pool_X[a_idx]; Xb=pool_X[b_idx]
            # Procrustes align Xb to Xa
            M=Xa.T@Xb
            try:
                U,S,Vt=np.linalg.svd(M)
                R=Vt.T@U.T  # maps Xb -> Xa frame? Y@R
                Xb_al=Xb@R
            except Exception:
                Xb_al=Xb
            # cost + assignment
            try:
                D=np.linalg.norm(Xa[:,None,:]-Xb_al[None,:,:],axis=2)
                r,c=linear_sum_assignment(D)
                # r is 0..n-1, c maps Xa->Xb_al
                perm=np.empty(n,dtype=int); perm[r]=c
                # build child: half from each
                mask=rng.random(n)<0.5
                child=np.empty_like(Xa)
                for i in range(n):
                    if mask[i]:
                        child[i]=Xa[i]
                    else:
                        child[i]=Xb_al[perm[i]]
                child=child/np.linalg.norm(child,axis=1,keepdims=True)
                # small anneal noise at seams
                seam_sig=0.015*(1-prog)+0.005
                child=child+seam_sig*rng.normal(size=(n,d))
                child/=np.linalg.norm(child,axis=1,keepdims=True)
                parent_val=min(pool_V[a_idx],pool_V[b_idx])
            except Exception:
                continue
            y_child,v_parent=child,parent_val
        else:
            # mutation from random pool member (bias to best early, uniform late for diversity)
            if rng.random()<0.5:
                # pick among top 4
                top=min(4,len(pool_X))
                # pool sorted? maintain sorted; pick random top
                # re-sort indices
                sorder=np.argsort(np.array(pool_V))
                p_idx=sorder[int(rng.integers(top))]
            else:
                p_idx=int(rng.integers(len(pool_X)))
            base=pool_X[p_idx]; parent_val=float(pool_V[p_idx])
            y_child=base.copy()
            mode=trial%7
            # anneal kick size
            scale=(1-prog)*0.7+0.3
            if mode==0:
                # global shake small
                sig=0.02*scale+0.005
                y_child=y_child+sig*rng.normal(size=(n,d))
            elif mode==1:
                # medium kick on random subset
                k=int(rng.integers(4,12))
                idx=rng.choice(n,k,replace=False)
                sig=(0.08+0.05*rng.random())*scale+0.01
                y_child[idx]+=sig*rng.normal(size=(k,d))
            elif mode==2:
                # overload targeted: kick high-contact points
                G=base@base.T; np.fill_diagonal(G,-2.0)
                m=float(G.max())
                cnt=(G>=m-1e-7).sum(axis=1)
                od=np.argsort(-cnt)
                k=int(rng.integers(3,8))
                victims=od[:k].copy()
                # include rattler (min contacts) reinsert
                ratt=int(np.argmin(cnt))
                if cnt[ratt]<2 and rng.random()<0.7:
                    victims=np.unique(np.append(victims,ratt))
                sig=(0.10+0.10*rng.random())*scale+0.01
                y_child[victims]+=sig*rng.normal(size=(len(victims),d))
            elif mode==3:
                # void reinsertion: move worst point to largest void
                G=base@base.T; np.fill_diagonal(G,-2.0)
                m=float(G.max())
                cnt=(G>=m-1e-7).sum(axis=1)
                victim=int(np.argmax(cnt))
                if rng.random()<0.3:
                    victim=int(np.argmin(cnt))
                cand=rng.normal(size=(3000,d)); cand/=np.linalg.norm(cand,axis=1,keepdims=True)
                # score = max dot to others (excluding victim)
                dots=cand@base.T
                # ignore victim column
                dots[:,victim]=-2.0
                md=dots.max(axis=1)
                best_c=cand[int(np.argmin(md))]
                # jitter
                y_child[victim]=best_c+0.02*rng.normal(size=(d,))
                # shake neighbours slightly
                y_child+=0.008*rng.normal(size=(n,d))
            elif mode==4:
                # large kick to escape basin
                sig=(0.25+0.15*rng.random())*scale+0.02
                k=int(rng.integers(15,45))
                idx=rng.choice(n,k,replace=False)
                y_child[idx]+=sig*rng.normal(size=(k,d))
            elif mode==5:
                # spherical cap rebuild: pick random direction, replace points inside cap
                u=rng.normal(size=(d,)); u/=np.linalg.norm(u)
                dots=base@u
                thr=np.quantile(dots,0.9)
                idx=np.where(dots>=thr)[0]
                if len(idx)<5:
                    idx=rng.choice(n,8,replace=False)
                for j in idx:
                    y_child[j]=rng.normal(size=(d,))
                y_child[idx]+=0.03*rng.normal(size=(len(idx),d))
            else:
                sig=0.04*scale+0.005
                y_child=y_child+sig*rng.normal(size=(n,d))
            y_child/=np.linalg.norm(y_child,axis=1,keepdims=True)
            v_parent=parent_val
        # polish candidate
        try:
            yp,vp=_lp_polish(y_child,pt,radius_init=4e-3)
        except Exception:
            trial+=1
            continue
        gp=_graph_vec(yp,vp)
        # update global best
        if vp<gbest_V:
            gbest_V=float(vp); gbest_X=yp.copy()
        # diversity: check closeness
        closest=min((_hamming(gp,g) for g in pool_G), default=10**9)
        # find member with closest graph
        if closest<20:
            # only replace that member if beats it
            dists=[_hamming(gp,g) for g in pool_G]
            j=int(np.argmin(dists))
            if vp<float(pool_V[j]):
                pool_X[j]=yp; pool_V[j]=float(vp); pool_G[j]=gp
                if vp<gbest_V:
                    gbest_V=float(vp); gbest_X=yp.copy()
            # else discard (keeps diversity)
        else:
            worst=int(np.argmax(pool_V))
            wv=float(pool_V[worst])
            if vp<wv:
                pool_X[worst]=yp; pool_V[worst]=float(vp); pool_G[worst]=gp
            else:
                # Metropolis accept worse into pool
                if rng.random()<np.exp(-(vp-wv)/max(T,1e-12)):
                    pool_X[worst]=yp; pool_V[worst]=float(vp); pool_G[worst]=gp
        trial+=1
        # occasionally re-sort pool for bias (cheap)
        if trial%25==0:
            sorder=np.argsort(np.array(pool_V))
            pool_X=[pool_X[i] for i in sorder]; pool_V=[float(pool_V[i]) for i in sorder]; pool_G=[pool_G[i] for i in sorder]
    # final polish best
    frem=deadline-time.time()
    if frem>6.0:
        try:
            gbest_X,gbest_V=_lp_polish(gbest_X,min(frem-1.0,5.0),radius_init=1e-3)
        except Exception:
            pass
    gbest_X=gbest_X/np.linalg.norm(gbest_X,axis=1,keepdims=True)
    return gbest_X

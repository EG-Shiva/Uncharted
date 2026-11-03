import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lse_max(X, alpha=180.0):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    m = float(c.max())
    E = np.exp(np.clip(alpha*(c-m), -50, 0.0))
    S = float(E.sum())
    return m + np.log(S+1e-300)/alpha, m

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lp_polish(X, budget, t_deadline=1e18):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - t0 < budget and time.time() < t_deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c_full = G[iu0]
        m = c_full.max()
        thresh = m - max(20*radius, 1e-9)
        sel = c_full > thresh
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c_full[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-cc, 2500)[:2500]
            I, J, cc = I[idx], J[idx], cc[idx]
            p = 2500
        nv = n*d+1
        try:
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],axis=1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p,1))],axis=1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v; bestX = y.copy(); X = y
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best

def _smooth_opt(X, steps, a0, a1, lr0, t_deadline):
    n, d = X.shape
    M = np.zeros_like(X); V = np.zeros_like(X)
    b1,b2,eps = 0.9,0.999,1e-8
    G0 = X @ X.T; np.fill_diagonal(G0,-2.0)
    best = float(G0.max()); bestX = X.copy()
    for t in range(1, steps+1):
        if (t & 63)==0 and time.time() > t_deadline:
            break
        frac = t/steps
        alpha = a0*(a1/a0)**frac
        lr = lr0*0.5*(1.0+np.cos(np.pi*frac))
        lr = max(lr, lr0*0.05)
        G = X @ X.T; np.fill_diagonal(G,-2.0)
        gmax = float(G.max())
        if gmax < best:
            best = gmax; bestX = X.copy()
        E = np.clip(alpha*(G-gmax), -30, 0.0)
        W = np.exp(E); np.fill_diagonal(W,0.0)
        Grad = W @ X
        dot = np.sum(Grad*X,axis=1,keepdims=True)
        Gt = Grad - dot*X
        M = b1*M+(1-b1)*Gt
        V = b2*V+(1-b2)*(Gt*Gt)
        Mh = M/(1-b1**t); Vh = V/(1-b2**t)
        X = X - lr*(Mh/(np.sqrt(Vh)+eps))
        X /= np.linalg.norm(X,axis=1,keepdims=True)
    v = _max_cos(X)
    if v < best:
        best=v; bestX=X.copy()
    return bestX, best

def _spherical_kmeans(Xc, K, rng, restarts=20, iters=25):
    n = Xc.shape[0]
    best_labels=None; best_score=-1e100; best_C=None
    for _ in range(restarts):
        idx = rng.choice(n, size=K, replace=False)
        C = Xc[idx].copy()
        labels = np.zeros(n,dtype=int)
        for _it in range(iters):
            S = Xc @ C.T
            labels = np.argmax(S,axis=1)
            Cnew = np.zeros_like(C)
            for k in range(K):
                mem = Xc[labels==k]
                if len(mem)==0:
                    Cnew[k]=Xc[rng.integers(n)]
                else:
                    m = mem.mean(axis=0); nm=np.linalg.norm(m)
                    Cnew[k]=m/nm if nm>1e-12 else Xc[rng.integers(n)]
            if np.allclose(Cnew,C):
                C=Cnew; break
            C=Cnew
        S = Xc @ C.T
        score = float(np.max(S,axis=1).sum())
        if score>best_score:
            best_score=score; best_labels=labels.copy(); best_C=C.copy()
    return best_labels, best_C

def solve():
    t_start=time.time(); T_END=t_start+172.0; T_EXPL=t_start+146.0
    rng=np.random.default_rng(0)
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.asarray(np.load(fn),dtype=np.float64)
            if a.shape==(131,9):
                a=a/np.linalg.norm(a,axis=1,keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands)==0:
        X0=rng.standard_normal((131,9)); X0/=np.linalg.norm(X0,axis=1,keepdims=True)
        cands.append(X0)
    bestX=None; best=1e9
    for a in cands:
        v=_max_cos(a)
        if v<best: best=v; bestX=a.copy()
    n,d=bestX.shape
    # pool of 5 diverse elites: (lse, hard, X)
    pool=[]
    lse0,_=_lse_max(bestX)
    pool.append((lse0,best,bestX.copy()))
    # add noisy copies to seed diversity
    for s in [0.01,0.02,0.035,0.06]:
        Xn=bestX+s*rng.standard_normal(bestX.shape)
        Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)
        lse,h=_lse_max(Xn); h2=_max_cos(Xn)
        pool.append((lse,h2,Xn))
    # sort by hard and keep 5
    pool=sorted(pool,key=lambda t:t[1])[:5]
    # incumbent best from pool
    for _,h,_ in pool:
        if h<best:
            best=h
    # find bestX as the argmin hard
    for _,h,X in pool:
        if abs(h-best)<1e-15:
            bestX=X.copy(); break
    clusters=None; cluster_core=None; cluster_K=10
    trial=0
    thetas=np.array([0.15,0.3,0.6])
    while time.time()<T_EXPL:
        rem_expl=T_EXPL-time.time()
        if rem_expl<1.0: break
        # pick base elite (bias to best but allow diversity)
        # 50% best, else random
        if rng.random()<0.5:
            base=pool[0][2]
        else:
            base=pool[rng.integers(len(pool))][2]
        ppm,_=_per_point_max(base)
        gmax=float(ppm.max())
        ratt_thr=gmax-1e-4
        core_idx=np.where(ppm>=ratt_thr)[0]
        if len(core_idx)<20:
            core_idx=np.arange(n)
        Xc=base[core_idx]
        # recluster every 6 trials or if invalid
        if (trial%6==0) or (clusters is None) or (len(clusters)!=len(core_idx)):
            cluster_K=int(rng.integers(8,13))  # 8..12
            try:
                labs,_=_spherical_kmeans(Xc,cluster_K,rng,restarts=20,iters=25)
            except Exception:
                labs=rng.integers(0,cluster_K,size=len(core_idx))
            clusters=labs; cluster_core=core_idx.copy()
        else:
            # core set may have changed; if size mismatched, reassign by nearest center is skipped -> force recluster next time
            if len(cluster_core)!=len(core_idx) or not np.array_equal(cluster_core,core_idx):
                # quick assign using old clustering via nearest member? just recluster quickly with fewer restarts
                try:
                    labs,_=_spherical_kmeans(Xc,cluster_K,rng,restarts=5,iters=15)
                except Exception:
                    labs=rng.integers(0,cluster_K,size=len(core_idx))
                clusters=labs; cluster_core=core_idx.copy()
        # eligible clusters size 5..18
        uniq,counts=np.unique(clusters,return_counts=True)
        elig=[u for u,c in zip(uniq,counts) if 5<=c<=18]
        if len(elig)==0:
            # fallback: pick random subset of core of size 10
            ksz=min(12,max(5,len(core_idx)//10))
            chosen_core=rng.choice(len(core_idx),size=ksz,replace=False)
            glob_idx=core_idx[chosen_core]
        else:
            ch=int(rng.choice(elig))
            glob_idx=core_idx[clusters==ch]
        # build rotation / reflection
        X=base.copy()
        mlen=len(glob_idx)
        do_reflect=(rng.random()<0.25)
        if do_reflect:
            h=rng.standard_normal(d); h/=np.linalg.norm(h)
            Xc_cl=X[glob_idx]
            proj=(Xc_cl@h)[:,None]*h
            Xc_cl2=Xc_cl-2.0*proj
            Xc_cl2/=np.linalg.norm(Xc_cl2,axis=1,keepdims=True)
            X[glob_idx]=Xc_cl2
        else:
            th=float(rng.choice(thetas))
            cent=X[glob_idx].mean(axis=0); cn=np.linalg.norm(cent)
            if cn<1e-12:
                u=np.zeros(d); u[0]=1.0
            else:
                u=cent/cn
            r=rng.standard_normal(d); r=r-(r@u)*u
            rn=np.linalg.norm(r)
            if rn<1e-12:
                r=rng.standard_normal(d); r=r-(r@u)*u; rn=np.linalg.norm(r)+1e-12
            v=r/rn
            Xc_cl=X[glob_idx]
            a=Xc_cl@u; b=Xc_cl@v
            co=np.cos(th); si=np.sin(th)
            coeff_u=(a*(co-1.0)-b*si)[:,None]*u
            coeff_v=(a*si+b*(co-1.0))[:,None]*v
            Xc_cl2=Xc_cl+coeff_u+coeff_v
            Xc_cl2/=np.linalg.norm(Xc_cl2,axis=1,keepdims=True)
            X[glob_idx]=Xc_cl2
        steps=int(1200+rng.integers(800))  # 1200..1999
        # anneal params slight random
        a0=30.0; a1=280.0; lr0=0.0035
        Xs,vs=_smooth_opt(X,steps,a0,a1,lr0,T_EXPL)
        lse,_=_lse_max(Xs)
        # update incumbent / pool
        improved_hard=(vs<best-1e-12)
        # LP polish only if within 1e-4
        if vs<best+1e-4:
            rb=min(2.5,max(0.4,rem_expl-0.3))
            if rb>0.35 and time.time()<T_EXPL-0.4:
                Xp,vp=_lp_polish(Xs,rb,T_EXPL)
                if vp<best-1e-12:
                    best=vp; bestX=Xp.copy()
                    lsep,_=_lse_max(Xp)
                    pool.append((lsep,vp,Xp.copy()))
                    pool=sorted(pool,key=lambda t:t[1])[:5]
                    # keep pool sorted by hard; ensure best first
                    trial+=1
                    continue
                if improved_hard:
                    best=vs; bestX=Xs.copy()
                    pool.append((lse,vs,Xs.copy()))
                    pool=sorted(pool,key=lambda t:t[1])[:5]
                    trial+=1
                    continue
            else:
                if improved_hard:
                    best=vs; bestX=Xs.copy()
                    pool.append((lse,vs,Xs.copy()))
                    pool=sorted(pool,key=lambda t:t[1])[:5]
                    trial+=1
                    continue
        # not polished (far) -> pool insertion by LSE diversity
        # insert if better LSE than worst and reasonably close hard
        worst_lse=max(t[0] for t in pool); worst_hard=max(t[1] for t in pool)
        if (lse<worst_lse-1e-12 and vs<worst_hard+5e-3) or (vs<worst_hard-1e-12):
            # diversity check: far enough from existing
            pool.append((lse,vs,Xs.copy()))
            # keep 5: sort by hard primarily but keep one LSE-best for diversity
            pool_sorted_h=sorted(pool,key=lambda t:t[1])
            # ensure LSE best kept
            lse_best=min(pool,key=lambda t:t[0])
            newpool=pool_sorted_h[:4]
            if not any(np.shares_memory(a[2],lse_best[2]) or np.allclose(a[2],lse_best[2]) for a in newpool):
                # check identity by object
                found=False
                for a in newpool:
                    if a[2] is lse_best[2]:
                        found=True; break
                if not found:
                    newpool.append(lse_best)
                else:
                    newpool.append(pool_sorted_h[4])
            else:
                newpool.append(pool_sorted_h[4] if len(pool_sorted_h)>4 else lse_best)
            # dedupe to 5 best-unique by hard
            pool=sorted(newpool,key=lambda t:t[1])[:5]
        trial+=1
    rem=T_END-time.time()
    if rem>1.2:
        b=min(rem-0.8,25.0)
        if b>0.8:
            try:
                Xp,vp=_lp_polish(bestX,b,T_END)
                if vp<best: best=vp; bestX=Xp.copy()
            except Exception:
                pass
    bestX=bestX/np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX

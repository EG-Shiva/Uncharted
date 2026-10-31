import time
import itertools
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normr(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _rand_orth(rng):
    A = rng.standard_normal((4,4))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R))
    d[d==0]=1.0
    Q = Q * d[None,:]
    if np.linalg.det(Q) < 0:
        Q[:,0] = -Q[:,0]
    return Q

def _all_signed_perms():
    import itertools
    mats=[]
    for p in itertools.permutations(range(4)):
        P=np.zeros((4,4))
        for i,j in enumerate(p):
            P[i,j]=1.0
        for mask in range(16):
            S=np.ones(4)
            for k in range(4):
                if (mask>>k)&1:
                    S[k]=-1.0
            M = P * S[None,:]
            # right-action matrix: row x -> x @ M ; columns permuted? keep as is
            mats.append(M)
    return np.array(mats)

def _match_stat(YB, Y, thr=0.999):
    G = YB @ Y.T
    m = G.max(axis=1)
    cnt = int((m>thr).sum())
    meanmiss = float((1.0-m).mean())
    return cnt, meanmiss, m

def _anneal(x, steps, beta0, beta1, lr0, mu):
    v = np.zeros_like(x)
    lb = np.log(beta1/beta0) if beta1>beta0 else 0.0
    for t in range(steps):
        frac = t/max(1,steps-1)
        beta = beta0*np.exp(lb*frac)
        lr = lr0*(1.0-0.5*frac)
        G = x @ x.T
        np.fill_diagonal(G,-2.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        S = W.sum(axis=1,keepdims=True)+1e-300
        A = (W@x)/S
        dot=(A*x).sum(axis=1,keepdims=True)
        Gt=A-dot*x
        v=mu*v+Gt
        vd=(v*x).sum(axis=1,keepdims=True)
        v-=vd*x
        x=x-lr*v
        x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return x

def _lp_polish_deadline(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x=_normr(x)
    n,d=x.shape
    best=_max_cos(x)
    radius=1e-3
    iu=np.triu_indices(n,1)
    while time.time()<deadline and radius>1e-14:
        c=(x@x.T)[iu]
        m=c.max()
        sel=c>m-max(20*radius,1e-9)
        I,J=iu[0][sel],iu[1][sel]
        p=len(I)
        if p==0:
            break
        nv=n*d+1
        try:
            rows=np.repeat(np.arange(p),2*d+1)
            cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
            vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv);obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                        bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            break
        if getattr(res,"status",1)!=0:
            radius*=0.5
            continue
        try:
            y=x+res.x[:-1].reshape(n,d)
        except Exception:
            radius*=0.5
            continue
        y=_normr(y)
        vv=_max_cos(y)
        if vv<best:
            x,best,radius=y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def _contacts_stats(X, tol=1e-9):
    G=X@X.T
    iu=np.triu_indices(X.shape[0],1)
    c=G[iu]
    m=c.max()
    sel=c>m-tol
    # degrees
    I,J=np.triu_indices(X.shape[0],1)
    Is=I[sel];Js=J[sel]
    deg=np.zeros(X.shape[0],dtype=int)
    np.add.at(deg,Is,1);np.add.at(deg,Js,1)
    return int(sel.sum()), float(deg.std()), int((deg==0).sum()), deg

def _cyclic_group(T, tol=1e-6, maxk=292):
    Gs=[np.eye(4)]
    cur=np.eye(4)
    for k in range(1,maxk+1):
        cur=cur@T
        if np.linalg.norm(cur-np.eye(4))<tol:
            break
        Gs.append(cur.copy())
        if len(Gs)>64:
            break
    return np.array(Gs)

def _expand_group(S, Gs):
    # S (K,4) rows, action x->x@G
    K=Gs.shape[0]
    return np.einsum('kij,nj->nik',Gs,S).reshape(S.shape[0]*K,4)

def _symfree_fg(yflat, Gs, K, Rf, alpha):
    N=K*Gs.shape[0]+Rf
    Y=yflat.reshape(N if False else (K+Rf),4) if False else yflat.reshape(K+Rf,4)
    # Y: first K seeds, last Rf free
    S=Y[:K]; F=Y[K:]
    nrmS=np.maximum(np.linalg.norm(S,axis=1,keepdims=True),1e-12)
    Sn=S/nrmS
    if Rf>0:
        nrmF=np.maximum(np.linalg.norm(F,axis=1,keepdims=True),1e-12)
        Fn=F/nrmF
        Xsym=_expand_group(Sn,Gs)
        X=np.vstack([Xsym,Fn])
    else:
        X=_expand_group(Sn,Gs)
    Xn=_normr(X)
    G=Xn@Xn.T
    np.fill_diagonal(G,-10.0)
    m=float(G.max())
    E=np.exp(alpha*(G-m))
    np.fill_diagonal(E,0.0)
    Ssum=float(E.sum())*0.5
    if Ssum<=0 or not np.isfinite(Ssum):
        return 1e6, np.zeros_like(yflat)
    f=m+np.log(Ssum)/alpha
    W=E/Ssum
    gradX=W@Xn
    # split grad for sym part and free part
    mG=Gs.shape[0]
    gsym=gradX[:K*mG].reshape(K,mG,4)
    gradS=np.zeros_like(Sn)
    for k in range(K):
        acc=np.zeros(4)
        for gi in range(mG):
            acc+=gsym[k,gi]@Gs[gi]  # since x->x@G, pullback is g@G^T? check: (s@G); d/ds = g@G^T? G orth so @G ; careful
            # Actually y=s@G, dy/ds = G^T? row: dy_j/ds_i = G_ij, so grad_s = grad_y @ G^T? No: grad_s_i = sum_j grad_y_j G_ij = (grad_y @ G^T?) Let's use G: (grad_y (row) @ G^T)? For row convention grad row vector: gs = gy @ G^T? Since gy is row. Our gy is row vector gsym. So gs = gy @ G.T? But Gs stored as right-multiply matrix, so G^T = G.T. Use that:
        # redo with transpose:
    gradS2=np.zeros_like(Sn)
    for k in range(K):
        acc=np.zeros(4)
        for gi in range(mG):
            acc+=gsym[k,gi]@Gs[gi].T
        gradS2[k]=acc
    dot=(gradS2*Sn).sum(axis=1,keepdims=True)
    gradSeed=(gradS2-dot*Sn)/nrmS
    if Rf>0:
        gfree=gradX[K*mG:]
        dotF=(gfree*Fn).sum(axis=1,keepdims=True)
        gradFree=(gfree-dotF*Fn)/nrmF
        gradY=np.vstack([gradSeed,gradFree])
    else:
        gradY=gradSeed
    return float(f), gradY.ravel()

def _optim_symfree(Gs, K, Rf, rng, deadline, alphas=(60.0,250.0,800.0), iters=(80,80,100)):
    from scipy.optimize import minimize
    Y0=rng.standard_normal((K+Rf,4))
    y=Y0.ravel()
    for alpha,mi in zip(alphas,iters):
        if time.time()>deadline-1.0:
            break
        def fg(v):
            return _symfree_fg(v,Gs,K,Rf,float(alpha))
        try:
            res=minimize(fg,y,jac=True,method='L-BFGS-B',options={'maxiter':int(mi),'ftol':1e-12,'gtol':1e-7,'maxfun':80000})
            y=res.x
        except Exception:
            break
    Y=y.reshape(K+Rf,4)
    Y=_normr(Y)
    S=Y[:K]; F=Y[K:]
    Sn=_normr(S)
    Xsym=_expand_group(Sn,Gs)
    if Rf>0:
        Fn=_normr(F)
        X=np.vstack([Xsym,Fn])
    else:
        X=Xsym
    return _normr(X)

def _snap_average(X, Gs, thr=0.97):
    n=X.shape[0]; m=Gs.shape[0]
    unused=np.ones(n,bool)
    orbits=[]
    # precompute norms (unit)
    for _ in range(n):
        # find first unused
        idx=np.where(unused)[0]
        if len(idx)==0:
            break
        seed_i=int(idx[0])
        seed=X[seed_i]
        imgs=seed@Gs.transpose(0,2,1) if False else np.einsum('ij,kjl->kil',np.atleast_2d(seed),Gs).reshape(m,4)  # seed@G? Gs acts right: seed@G
        # Actually einsum: (seed_j * G_jk?) seed row @ G => (seed @ G)_k = sum_j seed_j G_jk
        imgs2=np.stack([seed@G for G in Gs],axis=0)
        imgs=imgs2
        # match each img to nearest unused
        used_this=[]
        back=[]
        for k in range(m):
            cand_idx=np.where(unused)[0]
            if len(cand_idx)==0:
                break
            dots=X[cand_idx]@imgs[k]
            jpos=int(np.argmax(dots))
            best=float(dots[jpos])
            if best>thr:
                j=int(cand_idx[jpos])
                used_this.append(j)
                # map back: x_j @ Gk.T
                back.append(X[j]@Gs[k].T)
            # else leave missing
        if len(used_this)==0:
            # isolated: mark seed used, create orbit from seed alone
            unused[seed_i]=False
            r=seed/np.linalg.norm(seed)
            orb=np.stack([r@G for G in Gs],axis=0)
            orbits.append(orb)
            continue
        for j in used_this:
            unused[j]=False
        if len(back)>0:
            r=np.mean(np.stack(back,axis=0),axis=0)
            nr=np.linalg.norm(r)
            if nr<1e-9:
                r=seed
                nr=np.linalg.norm(r)
            r/=nr
        else:
            r=seed
        orb=np.stack([r@G for G in Gs],axis=0)
        orbits.append(orb)
        if len(orbits)*m>=n+ m:
            break
    if len(orbits)==0:
        return _normr(X.copy())
    Out=np.vstack(orbits)
    # fix size to n by whole orbits
    if Out.shape[0]>n:
        # drop last orbits until <=n
        keep=n//m
        rem=n-keep*m
        Out2=np.vstack(orbits[:keep]) if keep>0 else np.zeros((0,4))
        if rem>0 and keep<len(orbits):
            Out2=np.vstack([Out2,orbits[keep][:rem]])
        Out=Out2
    elif Out.shape[0]<n:
        rng=np.random.default_rng(7)
        need=n-Out.shape[0]
        extra=[]
        while need>0:
            s=rng.standard_normal(4);s/=np.linalg.norm(s)
            orb=np.stack([s@G for G in Gs],axis=0)
            take=min(need,m)
            # take whole orbit if fits else partial
            extra.append(orb[:take])
            need-=take
        Out=np.vstack([Out]+extra)
    return _normr(Out[:n])

def _align_perm(A,B):
    # permute A to match B (max sum cos). return A_aligned
    try:
        from scipy.optimize import linear_sum_assignment
        C=-(A@B.T)
        r,c=linear_sum_assignment(C)
        # r should be 0..n-1, c maps A-index -> B-index? C[r,c]: r indexes A, c indexes B. We want for each B i, which A j: invert
        inv=np.empty_like(c)
        inv[c]=r  # not exactly; r is arange typically. Safer: build aligned
        # linear_sum_assignment returns row_ind, col_ind pairing. So A[row] matched to B[col].
        # To align A to B order: aligned[col]=A[row]
        Al=np.empty_like(A)
        Al[c]=A[r]
        return Al
    except Exception:
        # greedy
        n=A.shape[0]
        used=np.zeros(n,bool)
        Al=np.empty_like(A)
        G=A@B.T
        for i in range(n):
            # B i best unused A
            col=G[:,i].copy()
            col[used]=-2
            j=int(np.argmax(col))
            used[j]=True
            Al[i]=A[j]
        return Al

def solve():
    t0=time.time(); TLIM=170.0
    rng=np.random.default_rng(292292)
    n,d=292,4
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn);a=np.asarray(a,dtype=np.float64)
            if a.shape==(n,d) and np.all(np.isfinite(a)):
                nr=np.linalg.norm(a,axis=1,keepdims=True)
                if np.min(nr)>1e-9:
                    cands.append(a/nr)
        except Exception:
            pass
    if not cands:
        Xb=_normr(rng.standard_normal((n,d)))
    else:
        vals=[_max_cos(c) for c in cands]
        Xb=cands[int(np.argmin(vals))].copy()
    best_val=_max_cos(Xb); X_best=Xb.copy()
    # ---------- symmetry detection ----------
    det_end=t0+10.0
    groups=[]
    try:
        # frames
        Qs=[np.eye(4)]
        try:
            U,Sv,Vt=np.linalg.svd(Xb,full_matrices=False)
            Qs.append(Vt.T.copy())
        except Exception:
            pass
        for _ in range(10):
            if time.time()>det_end-2.0: break
            Qs.append(_rand_orth(rng))
        Bmats=_all_signed_perms()
        best_hits=[]
        for Q in Qs:
            if time.time()>det_end-1.0: break
            Y=Xb@Q
            # loop B
            bcnt_best=-1; bbest=None
            for B in Bmats:
                YB=Y@B
                G=YB@Y.T
                mx=G.max(axis=1)
                cnt=int((mx>0.999).sum())
                if cnt>bcnt_best:
                    bcnt_best=cnt; bbest=B.copy()
                    if cnt>=290:
                        break
            # overall T in original coords: T = Q@B@Q.T
            if bbest is not None:
                T=Q@bbest@Q.T
                # recount in original coords
                cnt,_ ,_= _match_stat(Xb@T,Xb,0.999)
                best_hits.append((cnt,T.copy(),Q.copy(),bbest.copy()))
            if time.time()>det_end-1.0: break
        best_hits.sort(key=lambda t:-t[0])
        # keep top 2 distinct B symmetries
        for cnt,T,Q,Bb in best_hits[:2]:
            if cnt>=150:  # only meaningful
                groups.append(_cyclic_group(T,1e-4,32))
        # Kabsch patch search
        try:
            G0=Xb@Xb.T
            np.fill_diagonal(G0,-2.0)
            neigh=np.argsort(-G0,axis=1)[:,:8]
            kab=[]
            for _ in range(120):
                if time.time()>det_end: break
                i0=int(rng.integers(n)); j0=int(rng.integers(n))
                if i0==j0: continue
                Pi=Xb[[i0]+neigh[i0,:6].tolist()]
                Qp=Xb[[j0]+neigh[j0,:6].tolist()]
                M=Pi.T@Qp
                Uu,ss,Vt=np.linalg.svd(M)
                R=Uu@Vt
                cnt,mm,_=_match_stat(Xb@R,Xb,0.999)
                kab.append((cnt,R.copy()))
            kab.sort(key=lambda t:-t[0])
            for cnt,R in kab[:2]:
                if cnt>=150 and all(np.linalg.norm(R-g)>1e-3 for G in groups for g in G):
                    groups.append(_cyclic_group(R,1e-4,24))
        except Exception:
            pass
    except Exception:
        pass
    # fallback groups if none found: C4 quaternion and C2
    if len(groups)==0:
        G4=np.array([[1,0,0,0],[0,1,0,0],[-1,0,0,0],[0,-1,0,0]],dtype=np.float64)
        # build 4-cycle in two planes? simple block rotation 90deg in xy + identity? use cyclic order 4
        T4=np.array([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]],dtype=np.float64)
        groups.append(_cyclic_group(T4,1e-6,8))
        groups.append(np.array([np.eye(4),-np.eye(4)]))
    # limit groups
    groups=groups[:3]
    # ---------- symmetric snap + orbit optimisation ----------
    sym_cands=[]
    # snapped averages (novelty kept even if worse)
    for Gs in groups:
        try:
            if time.time()>t0+55: break
            Xs=_snap_average(X_best,Gs,thr=0.97)
            v=_max_cos(Xs)
            _,std,rat,_=_contacts_stats(Xs)
            sym_cands.append((v,std,Xs.copy()))
            if v<best_val:
                X_best, best_val=Xs.copy(),v
        except Exception:
            continue
    for Gs in groups:
        try:
            if time.time()>t0+75: break
            m=len(Gs)
            if m<2 or m>146: continue
            K=n//m; Rf=n-K*m
            # keep dims small: if K>73 limit iters
            dl=min(time.time()+14.0,t0+75)
            Xo=_optim_symfree(Gs,K,Rf,rng,dl)
            # dedup check
            G=Xo@Xo.T; np.fill_diagonal(G,-2.0)
            if float(G.max())>1-1e-8:
                continue
            v=_max_cos(Xo)
            _,std,_,_=_contacts_stats(Xo)
            sym_cands.append((v,std,Xo.copy()))
            if v<best_val+0.004:
                Z=_lp_polish_deadline(Xo.copy(),min(time.time()+3.0,t0+80))
                vz=_max_cos(Z)
                _,std2,_,_=_contacts_stats(Z)
                sym_cands.append((vz,std2,Z.copy()))
                if vz<best_val:
                    X_best,best_val=Z.copy(),vz
        except Exception:
            continue
    sym_cands.sort(key=lambda t:t[0])
    # keep diverse: best by score + best by regularity (low std)
    bases=[X_best.copy()]
    for v,s,Xs in sym_cands[:4]:
        bases.append(Xs.copy())
    # ---------- gradual symmetry release X(lam) ----------
    try:
        lams=[0.0,0.15,0.35,0.55,0.75,0.9]
        for Xs in list(bases[1:4]):
            if time.time()>t0+110: break
            try:
                Xa=_align_perm(Xs,X_best)
            except Exception:
                Xa=Xs
            for lam in lams:
                if time.time()>t0+110: break
                rem=t0+110-time.time()
                if rem<2.0: break
                Xmix=_normr((1-lam)*Xa+lam*X_best)
                # short anneal
                try:
                    Xmix=_anneal(Xmix.copy(),180,25.0,350.0,0.05,0.85)
                except Exception:
                    pass
                v=_max_cos(Xmix)
                if v<best_val+5e-04:
                    try:
                        Y=_lp_polish_deadline(Xmix.copy(),min(time.time()+1.8,t0+112))
                        vy=_max_cos(Y)
                    except Exception:
                        Y,vy=Xmix,v
                    # novelty: keep if better std even if slightly worse
                    try:
                        _,stdY,_,_=_contacts_stats(Y)
                        _,stdB,_,_=_contacts_stats(X_best)
                    except Exception:
                        stdY=1e9;stdB=1e9
                    if vy<best_val or (vy<best_val+2e-04 and stdY<stdB-0.15):
                        if vy<best_val:
                            X_best,best_val=Y.copy(),vy
                        if len(bases)<10:
                            bases.append(Y.copy())
                        else:
                            bases[int(rng.integers(len(bases)))]=Y.copy()
    except Exception:
        pass
    # ---------- final basin hopping with topology kicks ----------
    explore_end=t0+TLIM-12.0
    sigmas=[0.003,0.008,0.02,0.05,0.12]
    it=0
    while time.time()<explore_end:
        it+=1
        rem=explore_end-time.time()
        if rem<1.2: break
        r=rng.random()
        base=X_best if r<0.7 else bases[int(rng.integers(len(bases)))]
        X=base.copy()
        if it%7==0:
            try:
                G=X@X.T; np.fill_diagonal(G,-2.0)
                rowmax=G.max(axis=1); idx=int(np.argmax(rowmax))
                best_try=None;bsc=1e9
                for _ in range(8):
                    cand=rng.standard_normal(4); cand/=np.linalg.norm(cand)
                    sc=float((X@cand).max())
                    if sc<bsc: bsc=sc; best_try=cand
                X[idx]=best_try
            except Exception:
                pass
        sigma=float(sigmas[int(rng.integers(len(sigmas)))])
        X=X+sigma*rng.standard_normal(X.shape); X=_normr(X)
        steps=340 if rem>12 else 200
        try:
            X=_anneal(X,steps,20.0,380.0,0.06,0.85)
        except Exception:
            continue
        try:
            v=_max_cos(X)
        except Exception:
            continue
        if v<best_val+8e-05:
            dl=min(time.time()+2.0,explore_end)
            if dl-time.time()<0.5:
                if v<best_val:
                    X_best,best_val=X.copy(),v; bases[0]=X_best.copy()
                continue
            try:
                Y=_lp_polish_deadline(X,dl)
                vy=_max_cos(Y)
            except Exception:
                continue
            if vy<best_val:
                X_best,best_val=Y.copy(),vy; bases[0]=X_best.copy()
            elif vy<best_val+5e-05 and rng.random()<0.15:
                if len(bases)<10: bases.append(Y.copy())
    # final polish
    try:
        X_best=_lp_polish_deadline(X_best.copy(),min(time.time()+8.0,t0+TLIM-1.0))
        best_val=_max_cos(X_best)
    except Exception:
        pass
    return _normr(X_best)

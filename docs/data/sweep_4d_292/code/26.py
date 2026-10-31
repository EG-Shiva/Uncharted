import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quat_left_mats(group):
    # group (G,4) w,x,y,z -> L s.t. (g*s) col = L @ s_col
    Ls = []
    for w,x,y,z in group:
        L = np.array([[w,-x,-y,-z],
                      [x, w,-z, y],
                      [y, z, w,-x],
                      [z,-y, x, w]], dtype=np.float64)
        Ls.append(L)
    return np.array(Ls)  # (G,4,4)

def _expand(S, Ls):
    # S (K,4) rows, Ls (G,4,4): X[(k,g)] = S_k @ L_g.T
    K = S.shape[0]; G = Ls.shape[0]
    X = np.empty((K*G, 4), dtype=np.float64)
    for g in range(G):
        X[g::G] = S @ Ls[g].T if False else None
    # above stride wrong; do block
    for g in range(G):
        pass
    # block version: k major
    for k in range(K):
        for g in range(G):
            X[k*G+g] = Ls[g] @ S[k]
    return X

def _sym_fun_grad(yflat, Ls, K, alpha):
    Y = yflat.reshape(K,4)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    S = Y / nrm
    G = Ls.shape[0]
    N = K*G
    X = np.empty((N,4), dtype=np.float64)
    for k in range(K):
        sk = S[k]
        base = k*G
        for g in range(G):
            X[base+g] = Ls[g] @ sk
    Gr = X @ X.T
    np.fill_diagonal(Gr, -2.0)
    m = float(Gr.max())
    E = np.exp(alpha*(Gr - m))
    np.fill_diagonal(E, 0.0)
    Ssum = E.sum()*0.5 + 1e-300
    lse = m + np.log(Ssum)/alpha + np.log(2.0)/alpha - np.log(2.0)/alpha  # = m+log(Ssum)/a
    # actually Ssum is sum_{i<j}; lse = m + log(Ssum)/alpha
    W = E / Ssum  # symmetric, diag 0, sums to 2 over all
    gradX = W @ X  # (N,4)
    # back to seeds
    gradS = np.zeros_like(S)
    for k in range(K):
        base = k*G
        acc = np.zeros(4)
        for g in range(G):
            acc += Ls[g].T @ gradX[base+g]
        gradS[k] = acc
    # through normalization
    dot = (gradS*S).sum(axis=1, keepdims=True)
    gradY = (gradS - dot*S)/nrm
    return float(lse), gradY.ravel()

def _optim_sym(K, Ls, rng, alphas=(50.0,200.0,1000.0), iters=(120,120,150), deadline=None):
    from scipy.optimize import minimize
    Y0 = rng.standard_normal((K,4))
    y = Y0.ravel()
    for alpha, mi in zip(alphas, iters):
        if deadline is not None and time.time() > deadline-2.0:
            break
        def fg(v):
            return _sym_fun_grad(v, Ls, K, alpha)
        try:
            res = minimize(fg, y, jac=True, method='L-BFGS-B',
                           options={'maxiter': mi, 'ftol':1e-12, 'gtol':1e-8, 'maxfun':100000})
            y = res.x
        except Exception:
            break
    Y = y.reshape(K,4)
    Y /= np.maximum(np.linalg.norm(Y,axis=1,keepdims=True),1e-12)
    X = np.empty((K*Ls.shape[0],4))
    for k in range(K):
        for g in range(Ls.shape[0]):
            X[k*Ls.shape[0]+g] = Ls[g] @ Y[k]
    X /= np.linalg.norm(X,axis=1,keepdims=True)
    return X, Y

def _has_dup(X, tol=1e-12):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return bool(G.max() > 1.0-tol)

def _anneal(x, steps, beta0, beta1, lr0, mu):
    n,d = x.shape
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
        Gt = A-dot*x
        v = mu*v+Gt
        vd=(v*x).sum(axis=1,keepdims=True)
        v-=vd*x
        x = x-lr*v
        x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return x

def _lp_polish_deadline(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d = x.shape
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
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=_max_cos(y)
        if v<best:
            x,best,radius=y,v,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0=time.time()
    TLIM=172.0
    rng=np.random.default_rng(292292)
    n,d=292,4
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,dtype=np.float64)
            if a.shape==(n,d) and np.all(np.isfinite(a)):
                nr=np.linalg.norm(a,axis=1,keepdims=True)
                if np.min(nr)>1e-9:
                    cands.append(a/nr)
        except Exception:
            pass
    if not cands:
        Xb=rng.standard_normal((n,d));Xb/=np.linalg.norm(Xb,axis=1,keepdims=True)
    else:
        vals=[_max_cos(c) for c in cands]
        Xb=cands[int(np.argmin(vals))].copy()
    best_val=_max_cos(Xb)
    X_best=Xb.copy()
    # --- symmetry groups (subgroups of binary tetrahedral / hyperoctahedral) ---
    # C4 = {1,i,-1,-i} left multiplication (free, order 4 -> 73 orbits =292)
    G4=np.array([[1,0,0,0],[0,1,0,0],[-1,0,0,0],[0,-1,0,0]],dtype=np.float64)
    # C2 antipodal {1,-1} -> 146 orbits =292
    G2=np.array([[1,0,0,0],[-1,0,0,0]],dtype=np.float64)
    # Q8 = {+-1,+-i,+-j,+-k} order 8 (for mixed check, not directly fittable)
    L4=_quat_left_mats(G4)
    L2=_quat_left_mats(G2)
    sym_cands=[]
    sym_deadline=t0+55.0
    # C4 runs: 3 inits
    for rep in range(3):
        if time.time()>sym_deadline:
            break
        try:
            Xs,_=_optim_sym(73,L4,rng,alphas=(50.0,200.0,800.0),iters=(100,100,120),deadline=sym_deadline)
            if Xs.shape!=(292,4) or _has_dup(Xs):
                continue
            v=_max_cos(Xs)
            sym_cands.append((v,Xs.copy()))
        except Exception:
            continue
    # C2 run: 1 init, coarser
    if time.time()<sym_deadline-8.0:
        try:
            Xs,_=_optim_sym(146,L2,rng,alphas=(50.0,200.0),iters=(60,60),deadline=sym_deadline)
            if Xs.shape==(292,4) and not _has_dup(Xs):
                sym_cands.append((_max_cos(Xs),Xs.copy()))
        except Exception:
            pass
    sym_cands.sort(key=lambda t:t[0])
    sym_cands=sym_cands[:2]
    # polish symmetric candidates with tiny symmetry break + LP
    for v,Xs in list(sym_cands):
        if time.time()>t0+TLIM-45.0:
            break
        # only polish if remotely competitive (within 0.005)
        if v>best_val+0.005:
            continue
        try:
            pert=1e-4*rng.standard_normal(Xs.shape)
            # tangentize
            pert-= (pert*Xs).sum(axis=1,keepdims=True)*Xs
            Y=Xs+pert
            Y/=np.linalg.norm(Y,axis=1,keepdims=True)
            Y=_lp_polish_deadline(Y,min(time.time()+6.0,t0+TLIM-35.0))
            vy=_max_cos(Y)
            if vy<best_val:
                X_best, best_val=Y.copy(),vy
            # keep raw sym too if better
            if v<best_val:
                # re-evaluate raw polished? also LP raw without break
                Z=_lp_polish_deadline(Xs.copy(),min(time.time()+4.0,t0+TLIM-35.0))
                vz=_max_cos(Z)
                if vz<best_val:
                    X_best,best_val=Z.copy(),vz
        except Exception:
            continue
    # --- basin-hopping improvement from best + sym bases ---
    explore_end=t0+TLIM-28.0
    bases=[X_best.copy()]
    for _,Xs in sym_cands:
        bases.append(Xs.copy())
    it=0
    while time.time()<explore_end:
        it+=1
        rem=explore_end-time.time()
        if rem<1.0:
            break
        base=bases[int(rng.integers(len(bases)))] if rng.random()<0.7 else X_best
        X=base.copy()
        # perturb
        sigma=float(rng.choice([0.005,0.012,0.03,0.07,0.15]))
        X=X+sigma*rng.standard_normal(X.shape)
        X/=np.linalg.norm(X,axis=1,keepdims=True)
        steps=400 if rem>10 else 200
        try:
            X=_anneal(X,steps,20.0,350.0,0.06,0.85)
        except Exception:
            continue
        try:
            v=_max_cos(X)
        except Exception:
            continue
        if v<best_val+5e-05:
            dl=min(time.time()+2.0,explore_end)
            if dl-time.time()<0.5:
                if v<best_val:
                    X_best,best_val=X.copy(),v
                continue
            try:
                Y=_lp_polish_deadline(X,dl)
                vy=_max_cos(Y)
            except Exception:
                continue
            if vy<best_val:
                X_best,best_val=Y.copy(),vy
        elif v<best_val:
            X_best,best_val=X.copy(),v
    # final reserve polish of champion
    try:
        if time.time()<t0+TLIM-8.0:
            X_best=_lp_polish_deadline(X_best,t0+TLIM-2.0)
    except Exception:
        pass
    X_best/=np.linalg.norm(X_best,axis=1,keepdims=True)
    return X_best

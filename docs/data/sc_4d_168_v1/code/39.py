import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _quat_left(q):
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w],
    ], dtype=float)

def _build_hurwitz():
    qs = []
    for i in range(4):
        for s in (-1.0, 1.0):
            q = np.zeros(4); q[i] = s
            qs.append(q)
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                for sw in (-0.5, 0.5):
                    qs.append(np.array([sw, sx, sy, sz]))
    # qs should be 8+16=24
    G = np.stack([_quat_left(q) for q in qs], axis=0)
    return G

def _build_cyclic(m, k=5):
    G = []
    for t in range(m):
        a = 2*np.pi*t/m
        b = 2*np.pi*k*t/m
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        M = np.array([[ca,-sa,0,0],[sa,ca,0,0],[0,0,cb,-sb],[0,0,sb,cb]])
        G.append(M)
    return np.stack(G, axis=0)

def _expand(S, G):
    # S (K,4), G (g,4,4) -> (g*K,4)
    return np.einsum('gij,kj->gki', G, S).reshape(-1, 4)

def _sym_refine(S0, G, steps, beta0, beta1, lr0, lr1, mu=0.9, rng=None):
    S = _norm(S0.copy())
    V = np.zeros_like(S)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    gnum = G.shape[0]
    K = S.shape[0]
    for s in range(steps):
        f = s / max(1, steps-1)
        beta = float(np.exp(lb0 + (lb1-lb0)*f))
        tau = 1.0/beta
        lr = lr0*(lr1/lr0)**f if lr0!=lr1 else lr0
        X = _expand(S, G)
        Gr = X @ X.T
        np.fill_diagonal(Gr, -2.0)
        m = float(np.max(Gr))
        # weights
        W = np.exp(np.clip((Gr-m)/tau, -25, 0.0))
        np.fill_diagonal(W, 0.0)
        # raw grad wrt X
        GX = W @ X
        # map to seeds: Grad_S[k,j]=sum_{g,i} GX[g,k,i]*G[g,i,j]
        GXr = GX.reshape(gnum, K, 4)
        GS = np.einsum('gki,gij->kj', GXr, G)
        # tangent project
        d = np.sum(GS*S, axis=1, keepdims=True)
        Gt = GS - d*S
        sc = np.mean(np.linalg.norm(Gt, axis=1)) + 1e-12
        Gt = Gt / sc
        V = mu*V + (1.0-mu)*Gt
        bc = 1.0 - mu**(s+1)
        Vn = V/bc
        S = S - lr*Vn
        S = _norm(S)
    return S

def _full_refine(X0, steps, beta0, beta1, lr0, lr1, mu=0.88):
    X = _norm(X0.copy())
    V = np.zeros_like(X)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        f = s / max(1, steps-1)
        beta = float(np.exp(lb0 + (lb1-lb0)*f))
        tau = 1.0/beta
        lr = lr0*(lr1/lr0)**f if lr0!=lr1 else lr0
        Gr = X @ X.T
        np.fill_diagonal(Gr, -2.0)
        m = float(np.max(Gr))
        W = np.exp(np.clip((Gr-m)/tau, -25, 0.0))
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True) + 1e-30
        Grad = (W @ X)/rs
        d = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - d*X
        sc = np.mean(np.linalg.norm(Gt, axis=1)) + 1e-12
        Gt = Gt/sc
        V = mu*V + (1.0-mu)*Gt
        bc = 1.0 - mu**(s+1)
        Vn = V/bc
        X = X - lr*Vn
        X = _norm(X)
    return X

def _tangent_noise(X, sigma, rng):
    N = rng.randn(*X.shape)
    d = np.sum(N*X, axis=1, keepdims=True)
    Nt = N - d*X
    Y = X + sigma*Nt
    return _norm(Y)

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    i0, i1 = iu0[0], iu0[1]
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        vals_all = Gfull[i0, i1]
        m = float(np.max(vals_all))
        sel = vals_all > m - max(20*radius, 1e-9)
        if np.sum(sel) < 1:
            break
        I = i0[sel]; J = i1[sel]
        c = vals_all[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vls = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vls,(rows,cols)),shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best=v; xb=y.copy(); x=y
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
        if _max_cos(x) > best+1e-12:
            x=xb.copy(); radius*=0.5
    return xb

def solve():
    t0=time.time()
    BUDGET=175.0
    t_end=t0+BUDGET-1.0
    SYM_END=t0+62.0
    SEARCH_END=t0+130.0
    rng=np.random.RandomState(7)
    n=168
    try:
        Xb=np.load("best.npy").astype(float)
    except Exception:
        Xb=rng.randn(n,4)
    try:
        Xr=np.load("record.npy").astype(float)
    except Exception:
        Xr=Xb.copy()
    Xb=_norm(Xb); Xr=_norm(Xr)
    vb=_max_cos(Xb); vr=_max_cos(Xr)
    if vr<vb:
        Xb=Xr.copy(); vb=vr
    best_val=vb
    best_X=Xb.copy()
    # build groups
    G24=_build_hurwitz()
    G12=_build_cyclic(12,k=5)
    G8=_build_cyclic(8,k=3)
    configs=[(G24,7,8),(G12,14,2),(G8,21,2)]
    # hole pool from incumbent
    try:
        S_samp=rng.randn(20000,4); S_samp/=np.linalg.norm(S_samp,axis=1,keepdims=True)
        D=S_samp@Xb.T
        mc=np.max(D,axis=1)
        idx=np.argpartition(mc,600)[:600]
        holes=S_samp[idx]
        # order by depth
        o=np.argsort(mc[idx])
        holes=holes[o]
    except Exception:
        holes=_norm(rng.randn(600,4))
    cand_pool=[(best_val,best_X.copy())]
    def push(v,X):
        cand_pool.append((v,X.copy()))
        cand_pool.sort(key=lambda z:z[0])
        del cand_pool[4:]
    # symmetric search + break
    for (G,K,ntrials) in configs:
        g=G.shape[0]
        assert g*K==168,(g,K)
        for t in range(ntrials):
            if time.time()>SYM_END:
                break
            # init seeds
            if t%2==0 or len(holes)<K:
                S0=_norm(rng.randn(K,4))
            else:
                # sample K holes far apart greedily
                perm=rng.permutation(len(holes))
                chosen=[]
                for ii in perm:
                    h=holes[ii]
                    ok=True
                    for c in chosen:
                        if float(h@c)>0.85:
                            ok=False; break
                    if ok:
                        chosen.append(h)
                    if len(chosen)>=K:
                        break
                while len(chosen)<K:
                    chosen.append(_norm(rng.randn(1,4))[0])
                S0=_norm(np.stack(chosen,axis=0)+0.002*rng.randn(K,4))
            # sym anneal beta 30->500
            steps=1100 if K<=7 else 900
            # shorten if time low
            if SYM_END-time.time()<12:
                steps=min(steps,500)
            Ssym=_sym_refine(S0,G,steps,30.0,500.0,0.06,0.002,mu=0.9)
            Xsym=_norm(_expand(Ssym,G))
            vsym=_max_cos(Xsym)
            # break symmetry at 3 noise levels
            for sigma in (0.005,0.015,0.03):
                if time.time()>SEARCH_END:
                    break
                Y=_tangent_noise(Xsym,sigma,rng)
                # full LSE refinement beta 30->500
                Y2=_full_refine(Y,500,30.0,500.0,0.05,0.002,mu=0.88)
                # quick extra low-temp squeeze
                Y3=_full_refine(Y2,200,120.0,800.0,0.008,0.001,mu=0.9)
                v=float(min(_max_cos(Y2),_max_cos(Y3)))
                Yb=Y2 if _max_cos(Y2)<_max_cos(Y3) else Y3
                if v<best_val-1e-12:
                    best_val=v; best_X=Yb.copy()
                push(v,Yb)
            if time.time()>SYM_END and (G is not G24):
                break
        if time.time()>SYM_END+25:
            break
    # also squeeze incumbent directly (ensure no regression vs parent)
    if time.time()<SEARCH_END:
        for _ in range(2):
            if time.time()>SEARCH_END-2: break
            Y=_tangent_noise(best_X,0.008,rng)
            Y=_full_refine(Y,600,25.0,600.0,0.05,0.002,mu=0.88)
            v=_max_cos(Y)
            if v<best_val:
                best_val=v; best_X=Y.copy()
            push(v,Y)
    # final LP polish on top candidates
    cand_pool=sorted(cand_pool,key=lambda z:z[0])[:3]
    total_lp=max(5.0,t_end-time.time()-1.0)
    if total_lp>=5.0 and len(cand_pool)>0:
        shares=np.array([0.6,0.25,0.15][:len(cand_pool)],dtype=float)
        shares=shares/np.sum(shares)*total_lp
        for (v0,X0),sec in zip(list(cand_pool),shares):
            if time.time()>=t_end-0.5: break
            sec=min(float(sec),t_end-time.time()-0.5)
            if sec<1.0: continue
            Xp=_lp_polish(X0,time.time()+sec)
            vp=_max_cos(Xp)
            if vp<best_val:
                best_val=vp; best_X=Xp.copy()
    return _norm(best_X)

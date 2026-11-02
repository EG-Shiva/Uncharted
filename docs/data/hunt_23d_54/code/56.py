import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_orbit_X(sA, sB, Cmat, Smat, sgn):
    # sA (23,), sB (23,) -> X (54,23): 46 full + 8 partial (k=0..7)
    nfull = 46
    X = np.empty((54, 23))
    a_pairs = sA[:22].reshape(11, 2)
    b_pairs = sB[:22].reshape(11, 2)
    # full
    for j in range(11):
        c = Cmat[:, j]  # 46
        s = Smat[:, j]
        a0, a1 = a_pairs[j, 0], a_pairs[j, 1]
        X[:nfull, 2*j] = c*a0 - s*a1
        X[:nfull, 2*j+1] = s*a0 + c*a1
    X[:nfull, 22] = sgn * sA[22]
    # partial 8
    for j in range(11):
        c = Cmat[:8, j]
        s = Smat[:8, j]
        b0, b1 = b_pairs[j, 0], b_pairs[j, 1]
        X[nfull:, 2*j] = c*b0 - s*b1
        X[nfull:, 2*j+1] = s*b0 + c*b1
    X[nfull:, 22] = sgn[:8] * sB[22]
    return X

def _seed_grads(Gr_full, Gr_part, Cmat, Smat, sgn):
    gA = np.zeros(23)
    gB = np.zeros(23)
    # planes
    for j in range(11):
        cF = Cmat[:, j]; sF = Smat[:, j]
        gf0 = Gr_full[:, 2*j]; gf1 = Gr_full[:, 2*j+1]
        # adjoint Q^{-k}: [c*s? ] inverse rotation [[c,s],[-s,c]]
        gA[2*j] = np.sum(cF*gf0 + sF*gf1)
        gA[2*j+1] = np.sum(-sF*gf0 + cF*gf1)
        cP = Cmat[:8, j]; sP = Smat[:8, j]
        gp0 = Gr_part[:, 2*j]; gp1 = Gr_part[:, 2*j+1]
        gB[2*j] = np.sum(cP*gp0 + sP*gp1)
        gB[2*j+1] = np.sum(-sP*gp0 + cP*gp1)
    gA[22] = np.sum(sgn * Gr_full[:, 22])
    gB[22] = np.sum(sgn[:8] * Gr_part[:, 22])
    return gA, gB

def _opt_seeds(sA0, sB0, Cmat, Smat, sgn, steps, lr=0.012):
    sA = sA0.copy(); sB = sB0.copy()
    mA = np.zeros(23); vA = np.zeros(23)
    mB = np.zeros(23); vB = np.zeros(23)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for st in range(steps):
        f = st / max(steps-1, 1)
        beta = 50.0 * (1000.0/50.0)**f
        X = _build_orbit_X(sA, sB, Cmat, Smat, sgn)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        E = np.exp(beta*(G-gm))
        np.fill_diagonal(E, 0.0)
        Es = float(np.sum(E))
        if Es <= 0: Es = 1.0
        Gr = (E @ X) * (2.0/Es)
        GrF = Gr[:46]; GrP = Gr[46:]
        gA, gB = _seed_grads(GrF, GrP, Cmat, Smat, sgn)
        for g_, s_, m_, v_ in ((gA, sA, mA, vA), (gB, sB, mB, vB)):
            pass
        # Riemannian project
        gA = gA - np.dot(gA, sA)*sA
        gB = gB - np.dot(gB, sB)*sB
        # Adam update (in-place on mA etc.)
        mA = 0  # placeholder to avoid confusion; use local vars
        # we keep explicit arrays via closure: use lists
        break
    # real loop with proper state (rewritten to avoid bug above)
    sA = sA0.copy(); sB = sB0.copy()
    mA = np.zeros(23); vA = np.zeros(23)
    mB = np.zeros(23); vB = np.zeros(23)
    for st in range(steps):
        f = st / max(steps-1, 1)
        beta = 50.0 * (1000.0/50.0)**f
        X = _build_orbit_X(sA, sB, Cmat, Smat, sgn)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        E = np.exp(beta*(G-gm))
        np.fill_diagonal(E, 0.0)
        Es = float(np.sum(E))
        if Es <= 0: Es = 1.0
        Gr = (E @ X) * (2.0/Es)
        gA, gB = _seed_grads(Gr[:46], Gr[46:], Cmat, Smat, sgn)
        gA = gA - np.dot(gA, sA)*sA
        gB = gB - np.dot(gB, sB)*sB
        mA = b1*mA + (1-b1)*gA
        vA = b2*vA + (1-b2)*(gA*gA)
        mB = b1*mB + (1-b1)*gB
        vB = b2*vB + (1-b2)*(gB*gB)
        bc1 = 1 - b1**(st+1); bc2 = 1 - b2**(st+1)
        stepA = lr * (mA/bc1) / (np.sqrt(vA/bc2)+eps)
        stepB = lr * (mB/bc1) / (np.sqrt(vB/bc2)+eps)
        stepA = stepA - np.dot(stepA, sA)*sA
        stepB = stepB - np.dot(stepB, sB)*sB
        sA = sA - stepA; sA /= np.linalg.norm(sA)
        sB = sB - stepB; sB /= np.linalg.norm(sB)
    Xf = _build_orbit_X(sA, sB, Cmat, Smat, sgn)
    return sA, sB, Xf

def _adam_full(X0, steps, beta0, beta1, lr0, lr1):
    X = X0.copy()
    n = X.shape[0]
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for st in range(steps):
        f = st/max(steps-1, 1) if steps > 1 else 1.0
        beta = beta0*(beta1/beta0)**f
        lr = lr0*(lr1/lr0)**f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        E = np.exp(beta*(G-gm))
        np.fill_diagonal(E, 0.0)
        Es = float(np.sum(E))
        if Es <= 0: Es = 1.0
        Gr = (E @ X)*(2.0/Es)
        dot = np.sum(Gr*X, axis=1, keepdims=True)
        Gt = Gr - dot*X
        m = b1*m + (1-b1)*Gt
        v = b2*v + (1-b2)*(Gt*Gt)
        bc1 = 1-b1**(st+1); bc2 = 1-b2**(st+1)
        step = lr*(m/bc1)/(np.sqrt(v/bc2)+eps)
        sdot = np.sum(step*X, axis=1, keepdims=True)
        step = step - sdot*X
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, t0, budget):
    X = X/np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time()-start < seconds:
        if time.time()-t0 > budget:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20*radius, 1e-9)
        if np.count_nonzero(sel) > 800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr() if False else coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(7)
    n, d = 54, 23
    # group Q: 11 planes angles odd/46, order 46 with Q^23=-I
    thetas = 2*np.pi*(2*np.arange(11)+1)/46.0
    K = np.arange(46)
    Cmat = np.cos(K[:,None]*thetas[None,:])
    Smat = np.sin(K[:,None]*thetas[None,:])
    sgn = np.where(K%2==0, 1.0, -1.0)
    # fallback
    try:
        xb0 = np.load("best.npy").astype(float)
        xb0 /= np.linalg.norm(xb0,axis=1,keepdims=True)
        if xb0.shape != (54,23):
            raise ValueError("shape")
    except Exception:
        xb0 = rng.randn(54,23); xb0/=np.linalg.norm(xb0,axis=1,keepdims=True)
    best_fb = _max_cos(xb0)
    try:
        xr = np.load("record.npy").astype(float)
        xr/=np.linalg.norm(xr,axis=1,keepdims=True)
        if xr.shape==(54,23):
            vr=_max_cos(xr)
            if vr<best_fb:
                best_fb=vr; xb0=xr.copy()
    except Exception:
        pass
    cand_X=[xb0.copy()]; cand_v=[best_fb]
    # Phase 1: symmetric seed search, <= ~60s
    STEPS=3000
    trials=0
    while time.time()-t0 < 60.0 and trials < 8:
        if time.time()-t0 > 58.0:
            break
        sA0 = rng.randn(23); sA0/=np.linalg.norm(sA0)
        sB0 = rng.randn(23); sB0/=np.linalg.norm(sB0)
        try:
            _,_,Xf=_opt_seeds(sA0,sB0,Cmat,Smat,sgn,STEPS,lr=0.012)
            Xf/=np.linalg.norm(Xf,axis=1,keepdims=True)
            vf=_max_cos(Xf)
            cand_X.append(Xf); cand_v.append(vf)
        except Exception:
            pass
        trials+=1
    # pick best symmetric (exclude fallback idx0 if possible)
    order=np.argsort(np.array(cand_v))
    sym_best=None; sym_val=1e9
    for idx in order:
        if idx==0: continue
        sym_best=cand_X[idx]; sym_val=cand_v[idx]; break
    if sym_best is None:
        sym_best=xb0.copy(); sym_val=best_fb
    # Phase 2: symmetry breaking + unconstrained Adam, until 140s
    pool_X=[sym_best]; pool_v=[sym_val]
    # also try second best symmetric
    cnt=0
    for idx in order:
        if idx==0: continue
        if cand_X[idx] is sym_best: continue
        # add second distinct
        pool_X.append(cand_X[idx]); pool_v.append(cand_v[idx]); break
    noises=[0.01,0.03,0.05]
    for scale in noises:
        if time.time()-t0 > 140.0: break
        # apply to current best in pool
        bi=int(np.argmin(np.array(pool_v)))
        Xb=pool_X[bi]
        for rep in range(2):
            if time.time()-t0 > 140.0: break
            Xn=Xb+rng.randn(n,d)*scale
            Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)
            try:
                Xs=_adam_full(Xn,steps=1200,beta0=80,beta1=3000,lr0=0.015,lr1=0.001)
                vs=_max_cos(Xs)
                pool_X.append(Xs); pool_v.append(vs)
                # quick second anneal from same start with different schedule
            except Exception:
                continue
        # prune to 3 best
        o=np.argsort(np.array(pool_v))
        pool_X=[pool_X[i] for i in o[:3]]
        pool_v=[pool_v[i] for i in o[:3]]
    # extra unconstrained refinement of champion until 145s
    if time.time()-t0 < 145.0:
        bi=int(np.argmin(np.array(pool_v)))
        try:
            Xc=_adam_full(pool_X[bi],steps=800,beta0=200,beta1=5000,lr0=0.006,lr1=0.0005)
            vc=_max_cos(Xc)
            pool_X.append(Xc); pool_v.append(vc)
        except Exception:
            pass
    all_X=pool_X+[xb0]; all_v=pool_v+[best_fb]
    bi=int(np.argmin(np.array(all_v)))
    x_best=all_X[bi].copy()
    left=BUDGET-(time.time()-t0)
    lp_time=min(max(left-1.0,2.0),28.0)
    if left>2.5:
        try:
            x_best=_lp_polish(x_best,lp_time,t0,BUDGET)
            x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
        except Exception:
            pass
        bv=_max_cos(x_best)
        if bv>best_fb+1e-12:
            rem=BUDGET-(time.time()-t0)
            if rem>2:
                try:
                    xf=_lp_polish(xb0,min(rem-0.5,5.0),t0,BUDGET)
                    if _max_cos(xf)<bv:
                        x_best=xf
                except Exception:
                    x_best=xb0.copy()
    x_best/=np.linalg.norm(x_best,axis=1,keepdims=True)
    return x_best

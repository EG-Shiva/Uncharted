import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

SQ15 = np.sqrt(15.0)
IDX_ROLL = (np.arange(15)[None, :] - np.arange(15)[:, None]) % 15
SGN = ((-1.0) ** np.arange(15))[:, None]
FIXED = np.ones(15) / SQ15

def build_A(S):
    X = np.empty((76, 15), dtype=np.float64)
    for k in range(5):
        X[k*15:(k+1)*15] = S[k][IDX_ROLL]
    X[75] = FIXED
    return X

def build_B(S):
    X = np.empty((76, 15), dtype=np.float64)
    for k in range(5):
        X[k*15:(k+1)*15] = SGN * S[k][IDX_ROLL]
    X[75] = FIXED
    return X

def build_C(Y):
    return np.vstack([Y, -Y])

def _anneal_full(X, steps, T0, T1, lr0, lr1):
    X = X.copy()
    n = X.shape[0]
    for k in range(steps):
        f = k / max(1, steps-1)
        T = T0 * (T1/T0) ** f
        lr = lr0 * (lr1/lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        X = X - lr * Gt
        X = _norm(X)
    return X

def _refine_hole(v, Y, iters=120):
    v = v / np.linalg.norm(v)
    for k in range(iters):
        f = k / max(1, iters-1)
        Tk = 0.008 * (0.2 ** f)
        lrk = 0.08 * (0.2 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / Tk)
        ws = float(w.sum())
        if not np.isfinite(ws) or ws == 0:
            break
        w /= ws
        g = w @ Y
        g = g - float(g @ v) * v
        v = v - lrk * g
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            break
        v /= nv
    return v

def _lp_polish(x, t_end):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.6 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20*radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) > 1500:
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj,A_ub=A_ub,b_ub=-c[sel],A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    t_end = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = xb
    def ok(a):
        return a.shape==(76,15) and np.all(np.isfinite(a))
    cands=[]
    if ok(xb): cands.append(_norm(xb))
    if ok(xr): cands.append(_norm(xr))
    bv=np.inf; Xbest=None
    for c in cands:
        v=_maxcos(c)
        if v<bv: bv,Xbest=v,c.copy()
    if Xbest is None:
        Xbest=_norm(rng.normal(size=(76,15))); bv=_maxcos(Xbest)

    # ---- Phase 1: symmetry-constrained stochastic search ----
    best_sym = None; best_sym_v = np.inf
    # budget ~12s for this phase
    p1_end = t0 + 14.0
    def try_family(build, nseed, restarts, evals_per):
        nonlocal best_sym, best_sym_v
        for r in range(restarts):
            if time.time() > p1_end: break
            S = _norm(rng.normal(size=(nseed,15)))
            X = build(S); v=_maxcos(X)
            if v<best_sym_v: best_sym_v, best_sym=v.copy(), X.copy()
            curS=S.copy(); curv=v
            # greedy hill climb with decaying sigma
            for it in range(evals_per):
                if time.time()>p1_end: break
                f=it/max(1,evals_per-1)
                sigma=0.35*(0.008/0.35)**f
                Sp=_norm(curS+sigma*rng.normal(size=(nseed,15)))
                Xp=build(Sp); vp=_maxcos(Xp)
                if vp<curv:
                    curS,curv=Sp,vp
                    if vp<best_sym_v:
                        best_sym_v, best_sym=vp, Xp.copy()
                # occasional random restart of current walk if stuck
                if (it+1)%500==0 and time.time()<p1_end:
                    # shake current a bit larger to escape
                    curS=_norm(curS+0.05*rng.normal(size=(nseed,15)))
                    curv=_maxcos(build(curS))
            # end evals
    try_family(build_A,5,restarts=3,evals_per=900)
    try_family(build_B,5,restarts=3,evals_per=900)
    try_family(build_C,38,restarts=2,evals_per=700)

    # ---- Phase 2: unconstrained release of best symmetric point ----
    if best_sym is not None and time.time()<t_end-60:
        Xr=_anneal_full(best_sym,steps=2500,T0=0.015,T1=0.0008,lr0=4.0,lr1=0.1)
        vr=_maxcos(Xr)
        # extra fine polish
        if time.time()<t_end-55:
            Xr2=_anneal_full(Xr,steps=1500,T0=0.001,T1=0.0002,lr0=0.6,lr1=0.05)
            vr2=_maxcos(Xr2)
            if vr2<vr: Xr,vr=Xr2,vr2
        if vr<bv:
            Xbest,bv=Xr.copy(),vr

    # ---- Phase 3: hole-reinsertion basin hopping from global best ----
    # identify worst points adaptively; fast anneal only (no LP in loop)
    n,d=76,15
    trial=0
    while time.time() < t_end-32 and trial<14:
        trial+=1
        rm=_rowmax(Xbest)
        # choose k points: worst + random
        k_re=2 if trial%2==0 else 3
        order=np.argsort(-rm)
        idxs=[int(order[(trial-1)%n])]
        while len(idxs)<k_re:
            j=int(rng.integers(0,n))
            if j not in idxs: idxs.append(j)
        Xt=Xbest.copy()
        for idx in idxs:
            if time.time()>t_end-32: break
            mask=np.ones(n,bool); mask[idx]=False
            Y=Xt[mask]
            C=_norm(rng.normal(size=(2000,d)))
            sc=(C@Y.T).max(axis=1)
            v0=C[int(np.argmin(sc))]
            v0=_refine_hole(v0,Y,iters=100)
            Xt[idx]=v0
        sigma=[0.02,0.05,0.01,0.08][(trial-1)%4]
        Xt=_norm(Xt+sigma*rng.normal(size=(n,d)))
        if sigma<0.03:
            Xt=_anneal_full(Xt,steps=1800,T0=0.010,T1=0.0008,lr0=3.0,lr1=0.1)
        else:
            Xt=_anneal_full(Xt,steps=2200,T0=0.016,T1=0.0008,lr0=4.5,lr1=0.1)
        v=_maxcos(Xt)
        if v<bv-1e-12:
            Xbest,bv=Xt.copy(),v

    # ---- Phase 4: final LP polish on best ----
    if time.time()<t_end-1.5:
        Xp=_lp_polish(Xbest,t_end)
        vp=_maxcos(Xp)
        if vp<bv and np.all(np.isfinite(Xp)):
            Xbest,bv=Xp,vp
    Xbest=_norm(Xbest)
    return Xbest

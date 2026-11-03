import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0; b2p = 1.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    if steps > 1:
        logr = np.log(max(T1,1e-12)/max(T0,1e-12))/(steps-1)
    else:
        logr = 0.0
    for t in range(1, steps+1):
        T = T0*np.exp(logr*(t-1))
        if T < 5e-5: T = 5e-5
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        E = np.exp((G-mv)/T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()*0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x)/S
        dot = (Grad*x).sum(axis=1, keepdims=True)
        Gt = Grad - dot*x
        b1p *= b1; b2p *= b2
        m_ad = b1*m_ad + (1.0-b1)*Gt
        v_ad = b2*v_ad + (1.0-b2)*(Gt*Gt)
        mhat = m_ad/(1.0-b1p); vhat = v_ad/(1.0-b2p)
        step = mhat/(np.sqrt(vhat)+eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)+1e-12
        step = step/np.maximum(sn/1.0, 1.0)
        x = x - lr*step
        x = _norm(x)
    return x

def _lp_short(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time()-start < seconds and time.time() < deadline:
        if radius < 1e-13: break
        it += 1
        if it > 6: break
        G = x @ x.T
        cfull = G[iu0]
        m = cfull.max()
        sel = cfull > m - max(20*radius, 5e-4)
        idx = np.where(sel)[0]
        if len(idx) == 0: break
        # cap to tightest 2500 to keep HiGHS fast
        if len(idx) > 2500:
            part = np.argpartition(cfull[idx], -2500)[-2500:]
            idx = idx[part]
        I, J = iu0[0][idx], iu0[1][idx]
        p = len(I)
        csel = cfull[idx]
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": max(0.2, seconds-(time.time()-start))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; x = y; bx = y.copy()
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
    return bx, best

def _tangent_bases(x):
    n, d = x.shape
    dm = d-1
    T = np.zeros((n,d,dm))
    E = np.eye(d)
    for i in range(n):
        xi = x[i]
        k0 = int(np.argmax(np.abs(xi)))
        keep = [k for k in range(d) if k != k0]
        B0 = E[:, keep]  # d x dm
        coeff = xi[keep]  # dm
        B = B0 - np.outer(xi, coeff)
        Q, _ = np.linalg.qr(B)
        T[i] = Q
    return T

def _soft_modes(x, T, delta, K=30):
    n, d = x.shape
    dm = d-1
    N = n*dm
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n,1)
    c = G[iu]
    sel = c > m - delta
    I, J = iu[0][sel], iu[1][sel]
    p = len(I)
    if p < 50 or p > 6000:
        return None, None, None, m
    # build rigidity matrix m x N
    R = np.zeros((p, N), dtype=float)
    # precompute projections T_i^T x_j : need per pair
    # loop over pairs (python loop ~ few thousand, each small dot -> ok)
    for r in range(p):
        i, j = int(I[r]), int(J[r])
        ai = T[i].T @ x[j]  # dm
        aj = T[j].T @ x[i]
        R[r, i*dm:(i+1)*dm] = ai
        R[r, j*dm:(j+1)*dm] = aj
    # Gram
    Gram = R.T @ R
    # deflate global rotations: build W N x 45
    pairs = [(a,b) for a in range(d) for b in range(a+1,d)]
    W = np.zeros((N, len(pairs)))
    for k,(a,b) in enumerate(pairs):
        # W block for point i: T[i,a,:]*x[i,b]-T[i,b,:]*x[i,a]
        W[:,k] = (T[:,a,:]*x[:,b:b+1] - T[:,b,:]*x[:,a:a+1]).reshape(N)
    try:
        Q,_ = np.linalg.qr(W)
        Gram = Gram + 10.0*(Q @ Q.T)
    except Exception:
        pass
    try:
        w, V = np.linalg.eigh(Gram)
    except Exception:
        return None, None, None, m
    idx = np.argsort(w)[:K]
    Vs = V[:, idx]  # N x K
    ws = w[idx]
    return Vs, ws, (I,J), m

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    EXPL_END = t0 + 128.0
    rng = np.random.default_rng(20261103)
    n, d = 154, 10
    best = None; best_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn)
            if a.shape==(n,d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val=v; best=a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n,d)))
        best_val = _max_cos(best)
    cur = best.copy(); cur_val = best_val
    # cache tangent + modes, refresh periodically
    Vs = None; IJ = None; mcur = cur_val
    Tcur = None
    outer = 0
    # quick short LP to get true basin value cheap (2s max)
    try:
        if time.time() < EXPL_END-10:
            b2,v2 = _lp_short(cur, 1.5, EXPL_END)
            if v2 < best_val:
                best=b2.copy(); best_val=v2; cur=b2.copy(); cur_val=v2
    except Exception:
        pass
    while time.time() < EXPL_END:
        outer += 1
        if time.time() > DEAD-25: break
        # recompute soft modes every outer iter (each covers several trials)
        try:
            Tcur = _tangent_bases(cur)
        except Exception:
            Tcur = None
        Vs = None
        if Tcur is not None:
            # random realistic threshold
            delta = float(rng.choice([1e-6, 3e-6, 1e-5, 2e-5, 3e-5]))
            try:
                Vs, ws, IJ, mcur = _soft_modes(cur, Tcur, delta, K=30)
            except Exception:
                Vs=None
        if Vs is None:
            # fallback: small gaussian + relax (rare)
            prop = _norm(cur + rng.normal(scale=0.03, size=cur.shape))
            prop = _relax(prop, steps=350, lr=0.012, T0=0.010, T1=0.001)
            v = _max_cos(prop)
            if v < best_val-2e-7:
                try:
                    pp,vp = _lp_short(prop, 0.8, EXPL_END)
                    if vp < best_val:
                        best=pp.copy(); best_val=vp; cur=pp.copy(); cur_val=vp
                except Exception:
                    pass
            continue
        N = n*(d-1)
        dm = d-1
        # top pairs for sign flip
        try:
            Gc = cur @ cur.T
            np.fill_diagonal(Gc,-2.0)
            iu = np.triu_indices(n,1)
            cc = Gc[iu]
            k = min(40, len(cc))
            top = np.argpartition(cc,-k)[-k:]
            TI = iu[0][top]; TJ = iu[1][top]
            # rows for top pairs in same tangent coords
            Rt = np.zeros((k, N))
            for r in range(k):
                i,j = int(TI[r]), int(TJ[r])
                Rt[r, i*dm:(i+1)*dm] = Tcur[i].T @ cur[j]
                Rt[r, j*dm:(j+1)*dm] = Tcur[j].T @ cur[i]
        except Exception:
            Rt=None
        ntrials = 3
        for tt in range(ntrials):
            if time.time() > EXPL_END: break
            Kuse = int(rng.integers(10,31))  # 10-30
            Kuse = min(Kuse, Vs.shape[1])
            cols = rng.choice(Vs.shape[1], size=Kuse, replace=False)
            coeff = rng.normal(size=Kuse)
            u = Vs[:,cols] @ coeff
            # sign flip to reduce current highest pairs
            if Rt is not None:
                try:
                    s = float((Rt @ u).sum())
                    if s > 0: u = -u
                    # random extra flip 20% to explore opposite
                    if rng.random() < 0.2: u = -u
                except Exception:
                    pass
            # scale to amplitude 0.03-0.25 rad (log uniform)
            amp = float(np.exp(rng.uniform(np.log(0.03), np.log(0.25))))
            # per-point tangent norms
            U = u.reshape(n,dm)
            # t_i = T_i @ u_i
            # compute norms fast
            # (n,d) = sum_q T[n,d,q]*U[n,q]
            Tv = (Tcur * U[:,None,:]).sum(axis=2)
            pn = np.linalg.norm(Tv, axis=1).max()+1e-12
            u = u*(amp/pn)
            U = u.reshape(n,dm)
            Tv = (Tcur * U[:,None,:]).sum(axis=2)
            pn2 = np.linalg.norm(Tv,axis=1)
            # exponential map
            prop = np.zeros_like(cur)
            for i in range(n):
                ti = Tv[i]; ni = pn2[i]
                if ni < 1e-14:
                    prop[i]=cur[i]
                else:
                    prop[i]=np.cos(ni)*cur[i]+np.sin(ni)*(ti/ni)
            prop = _norm(prop)
            steps = int(rng.integers(300,601))
            lr = float(rng.uniform(0.008,0.016))
            T0a = float(rng.uniform(0.008,0.015)); T1a = float(rng.uniform(0.0007,0.002))
            prop = _relax(prop, steps=steps, lr=lr, T0=T0a, T1=T1a)
            v = _max_cos(prop)
            # only spend LP if promising: within 3e-3 of best or better
            if v > best_val + 0.003:
                continue
            try:
                pp,vp = _lp_short(prop, 0.9, EXPL_END)
            except Exception:
                continue
            if vp < best_val - 2e-7:
                best=pp.copy(); best_val=vp; cur=pp.copy(); cur_val=vp
                break  # refresh modes after improvement
    # final polish with remaining time
    remain = DEAD - time.time() - 1.5
    if remain > 3:
        try:
            lp_t = min(max(5.0, remain-1.0), 30.0)
            best,_ = _lp_short(best, lp_t, DEAD-1.0)
        except Exception:
            pass
    best = _norm(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best,axis=1,keepdims=True)
    bad = (nr.squeeze()<1e-12)
    if np.any(bad):
        rb = rng.normal(size=(int(np.sum(bad)),d))
        best[bad]=_norm(rb)
    best=_norm(best)
    return best

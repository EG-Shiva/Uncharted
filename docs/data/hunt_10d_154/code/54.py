import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X, rng=None):
    n, d = X.shape
    k = d - 1
    B = np.empty((n, d, k))
    for i in range(n):
        x = X[i]
        if rng is None:
            R = np.random.randn(d, k)
        else:
            R = rng.standard_normal((d, k))
        R = R - np.outer(x, (x @ R))
        Q, _ = np.linalg.qr(R)
        # ensure tangent (qr sign may flip but still orthogonal to x up to num err)
        # re-orthogonalize against x once
        Q = Q - np.outer(x, (x @ Q))
        # renormalize columns
        nr = np.linalg.norm(Q, axis=0, keepdims=True)
        nr[nr == 0] = 1.0
        Q = Q / nr
        B[i] = Q
    return B

def _build_J(X, B, thr=2e-5, cap=2500):
    n, d = X.shape
    k = d - 1
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = float(np.max(G))
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    slack = M - c
    # thresholds 1e-6,1e-5,1e-4 diagnostics: adapt if too few
    sel_thr = thr
    idx = np.where(slack < sel_thr)[0]
    if len(idx) < 200:
        sel_thr = 1e-4
        idx = np.where(slack < sel_thr)[0]
    if len(idx) < 200:
        sel_thr = 5e-4
        idx = np.where(slack < sel_thr)[0]
    if len(idx) < 100:
        # take tightest 300
        order = np.argsort(slack)[:300]
        idx = order
    if len(idx) > cap:
        order = np.argsort(slack[idx])[:cap]
        idx = idx[order]
    I = iu0[0][idx]
    J = iu0[1][idx]
    p = len(I)
    N = n * k
    Jmat = np.zeros((p, N), dtype=float)
    # row r: B_I^T x_J at offset I*k, B_J^T x_I at offset J*k
    for r in range(p):
        i = int(I[r]); j = int(J[r])
        Jmat[r, i*k:(i+1)*k] = B[i].T @ X[j]
        Jmat[r, j*k:(j+1)*k] = B[j].T @ X[i]
    return Jmat, I, J, M, slack

def _soft_modes(Jmat, kwant=24, rng=None):
    p, N = Jmat.shape
    k = int(min(kwant, min(p, N) - 2))
    if k < 5:
        k = int(min(10, min(p, N) - 1))
    if k <= 0:
        return None, None
    # try svds SM
    try:
        from scipy.sparse.linalg import svds
        Js = csr_matrix(Jmat)
        v0 = None
        if rng is not None:
            v0 = rng.standard_normal(N)
        # svds smallest magnitude
        u, s, vt = svds(Js, k=k, which='SM', tol=1e-6, maxiter=300, v0=v0)
        # svds returns ascending? sort by s ascending (softest first)
        order = np.argsort(s)
        s = s[order]
        vt = vt[order, :]
        # vt shape k x N, want V as N x k (columns = directions)
        V = vt.T.copy()
        return V, s
    except Exception:
        pass
    # fallback: eigh of J^T J
    try:
        JTJ = Jmat.T @ Jmat
        # symmetrize
        JTJ = 0.5 * (JTJ + JTJ.T)
        w, Vfull = np.linalg.eigh(JTJ)
        V = Vfull[:, :k].copy()
        s = np.sqrt(np.maximum(w[:k], 0.0))
        return V, s
    except Exception:
        return None, None

def _geodesic_step(X, B, u_flat, step):
    n, d = X.shape
    k = d - 1
    U = u_flat.reshape(n, k)
    # tangent vectors
    T = np.einsum('nik,nk->ni', B, U)
    rms = float(np.sqrt(np.mean(np.sum(T*T, axis=1))))
    if not np.isfinite(rms) or rms == 0:
        return _norm(X)
    T = T * (step / rms)
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    # avoid divide by zero
    small = norms[:, 0] < 1e-14
    Xn = np.empty_like(X)
    cn = np.cos(norms[:, 0])
    sn = np.sin(norms[:, 0])
    # for small norms use first order
    for i in range(n):
        if small[i]:
            v = X[i] + T[i]
            v /= np.linalg.norm(v)
            Xn[i] = v
        else:
            Xn[i] = cn[i]*X[i] + (sn[i]/norms[i,0])*T[i]
    return _norm(Xn)

def _adam(X, steps, T0, T1, lr0, deadline=1e18):
    X = _norm(X)
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1,1e-12)/max(T0,1e-12))/(steps-1)
    else:
        logr = 0.0
    for t in range(1, steps+1):
        if t % 200 == 0 and time.time() > deadline:
            break
        T = T0*np.exp(logr*(t-1))
        if T < 1e-12:
            T = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(np.max(G))
        E = np.exp((G-mv)/T)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E)*0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X)/S
        dot = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - dot*X
        b1p *= b1; b2p *= b2
        m = b1*m + (1.0-b1)*Gt
        v = b2*v + (1.0-b2)*(Gt*Gt)
        mhat = m/(1.0-b1p)
        vhat = v/(1.0-b2p)
        st = mhat/(np.sqrt(vhat)+eps)
        frac = t/steps
        lrt = lr0*(1.0-0.5*frac)
        X = X - lrt*st
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline=1e18):
    X = _norm(X)
    n, d = X.shape
    best = _mc(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    itn = 0
    while time.time()-start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn >= 40:
            break
        itn += 1
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        thr = max(20.0*radius, 1e-9)
        sel = c > m - thr
        I = iu[0][sel]; J = iu[1][sel]
        if len(I) == 0:
            break
        if len(I) > 2500:
            od = np.argsort(-c[sel])[:2500]
            I, J = I[od], J[od]
            cv = c[sel][od]
        else:
            cv = c[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n,d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        vv = _mc(y)
        if vv < best:
            X, best, radius = y, vv, min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    INTERNAL = t0 + 160.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    vv = _mc(a)
                    if vv < best_val:
                        best_val = vv
                        best = a.copy()
        except Exception:
            continue
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _mc(best)
    else:
        best = _norm(best)
        best_val = _mc(best)
    Xbest = best.copy()

    # build initial soft modes
    try:
        B0 = _tangent_bases(Xbest, rng=np.random.default_rng(1))
    except Exception:
        B0 = None
    Vsoft = None
    ksoft = 24
    if B0 is not None:
        try:
            Jm, _, _, _, _ = _build_J(Xbest, B0, thr=2e-5)
            Vsoft, _ = _soft_modes(Jm, k Hanover if False else ksoft, rng=rng)
        except Exception:
            Vsoft = None
    # need deterministic B0 without consuming main rng too much: rebuild with fixed seed
    # if Vsoft failed, will use random tangent moves
    HOP_END = t0 + 130.0
    if HOP_END > INTERNAL - 28:
        HOP_END = INTERNAL - 28
    steps_cycle = [0.02, 0.06, 0.12, 0.20]
    trial = 0
    # keep best ever
    while time.time() < HOP_END and time.time() < DEAD - 28:
        trial += 1
        step = steps_cycle[(trial-1) % 4]
        # refresh soft modes if Xbest changed recently (every 8 trials or after improvement)
        if trial % 8 == 1 and trial > 1:
            # recompute occasionally to track drift? keep same basin modes actually
            pass
        Xtry = None
        if Vsoft is not None and B0 is not None:
            try:
                kk = Vsoft.shape[1]
                c = rng.standard_normal(kk)
                # weight toward softer: divide by rank factor
                w = 1.0/(1.0+0.15*np.arange(kk))
                c = c*w
                cn = np.linalg.norm(c)
                if cn == 0 or not np.isfinite(cn):
                    c = rng.standard_normal(kk)
                    cn = np.linalg.norm(c)
                c = c/cn
                u = Vsoft @ c
                Xtry = _geodesic_step(Xbest, B0, u, step)
                # tiny extra jitter to break symmetry
                Xtry = Xtry + 2e-4*rng.standard_normal((n,d))
                Xtry = _norm(Xtry)
            except Exception:
                Xtry = None
        if Xtry is None:
            # fallback random tangent-ish global shake
            Xtry = Xbest + step*rng.standard_normal((n,d))
            Xtry = _norm(Xtry)
        # annealed softmax Adam 3e-3 -> 2e-5
        try:
            # adapt steps to remaining time
            trem = HOP_END - time.time()
            if trem < 3:
                break
            steps = 1000
            if step >= 0.12:
                steps = 1200
            elif step <= 0.03:
                steps = 800
            Xref = _adam(Xtry, steps, 3e-3, 2e-5, 0.015, deadline=HOP_END)
        except Exception:
            continue
        try:
            val = _mc(Xref)
        except Exception:
            continue
        # retain best
        if val < best_val:
            Xbest = Xref.copy()
            best_val = val
            # recompute soft modes for new basin
            try:
                B0 = _tangent_bases(Xbest, rng=np.random.default_rng(1))
                Jm, _, _, _, _ = _build_J(Xbest, B0, thr=2e-5)
                Vn, _ = _soft_modes(Jm, ksoft, rng=rng)
                if Vn is not None:
                    Vsoft = Vn
            except Exception:
                pass
            # early exit if big win? continue
        # also test intermediate Xtry true max (spec: test after every stage) - already covered
        # occasional short LP on near-miss to test basin
        if (val < best_val + 0.002) and (trial % 12 == 0) and time.time() < HOP_END - 4:
            try:
                Xp = _lp_polish(Xref, 1.2, deadline=HOP_END)
                vp = _mc(Xp)
                if vp < best_val:
                    Xbest = Xp.copy(); best_val = vp
                    try:
                        B0 = _tangent_bases(Xbest, rng=np.random.default_rng(1))
                        Jm, _, _, _, _ = _build_J(Xbest, B0, thr=2e-5)
                        Vn, _ = _soft_modes(Jm, ksoft, rng=rng)
                        if Vn is not None:
                            Vsoft = Vn
                    except Exception:
                        pass
            except Exception:
                pass
    # final trust-region LP polish, at most 25s, respect internal deadline
    try:
        tnow = time.time()
        budget = min(25.0, INTERNAL - tnow, DEAD - tnow - 2.0)
        if budget > 2.0:
            Xp = _lp_polish(Xbest, budget, deadline=INTERNAL)
            vp = _mc(Xp)
            if vp < best_val and np.all(np.isfinite(Xp)):
                Xbest = Xp; best_val = vp
    except Exception:
        pass
    # final micro Adam squeeze if time left (helps resolve ties)
    try:
        tnow = time.time()
        if DEAD - tnow > 4.0:
            Xt = _adam(Xbest, 400, 5e-4, 2e-5, 0.008, deadline=DEAD-2.0)
            vt = _mc(Xt)
            if vt < best_val:
                Xbest = Xt; best_val = vt
    except Exception:
        pass
    Xbest = _norm(Xbest)
    return Xbest

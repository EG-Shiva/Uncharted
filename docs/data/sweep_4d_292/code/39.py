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

def _quat_left_mats(group):
    Ls = []
    for w, x, y, z in group:
        L = np.array([[w,-x,-y,-z],[x,w,-z,y],[y,z,w,-x],[z,-y,x,w]], dtype=np.float64)
        Ls.append(L)
    return np.array(Ls)

def _cyclic_mats(order, p):
    G = np.zeros((order, 4, 4), dtype=np.float64)
    for t in range(order):
        a1 = 2*np.pi*t/order
        a2 = 2*np.pi*p*t/order
        c1, s1 = np.cos(a1), np.sin(a1)
        c2, s2 = np.cos(a2), np.sin(a2)
        G[t] = np.array([[c1,-s1,0,0],[s1,c1,0,0],[0,0,c2,-s2],[0,0,s2,c2]], dtype=np.float64)
    return G

def _expand_seeds(S, Ls):
    # S (K,4), Ls (M,4,4) -> (K*M,4)
    return np.einsum('mij,kj->kmi', Ls, S).reshape(S.shape[0]*Ls.shape[0], 4)

def _sym_fg(yflat, Ls, K, alpha):
    Y = yflat.reshape(K, 4)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    S = Y / nrm
    X = _expand_seeds(S, Ls)
    Xn = _normr(X)
    G = Xn @ Xn.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    E = np.exp(alpha*(G - m))
    np.fill_diagonal(E, 0.0)
    Ssum = float(E.sum())*0.5
    if Ssum <= 0 or not np.isfinite(Ssum):
        return 1e6, np.zeros_like(yflat)
    f = m + np.log(Ssum)/alpha
    W = E / Ssum
    gradX = W @ Xn
    # back to seeds
    gradS = np.zeros_like(S)
    # gradX reshape (K,M,4)
    gx = gradX.reshape(K, Ls.shape[0], 4)
    for k in range(K):
        acc = np.zeros(4)
        for mm in range(Ls.shape[0]):
            acc += Ls[mm].T @ gx[k, mm]
        gradS[k] = acc
    # through row normalization of X (approx ignore, X already unit if S unit) + S norm
    dot = (gradS*S).sum(axis=1, keepdims=True)
    gradY = (gradS - dot*S)/nrm
    return float(f), gradY.ravel()

def _optim_sym(K, Ls, rng, alphas=(50.0,200.0,800.0), iters=(120,120,140), deadline=None):
    from scipy.optimize import minimize
    Y0 = rng.standard_normal((K, 4))
    y = Y0.ravel()
    for alpha, mi in zip(alphas, iters):
        if deadline is not None and time.time() > deadline-1.5:
            break
        def fg(v):
            return _sym_fg(v, Ls, K, float(alpha))
        try:
            res = minimize(fg, y, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol':1e-12, 'gtol':1e-7, 'maxfun':100000})
            y = res.x
        except Exception:
            break
    Y = y.reshape(K, 4)
    Y = _normr(Y)
    X = _expand_seeds(Y, Ls)
    X = _normr(X)
    return X, Y

def _has_dup(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return bool(float(G.max()) > 1.0-tol)

def _anneal(x, steps, beta0, beta1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1/beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t/max(1, steps-1)
        beta = beta0*np.exp(lb*frac)
        lr = lr0*(1.0-0.5*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)+1e-300
        A = (W@x)/S
        dot = (A*x).sum(axis=1, keepdims=True)
        Gt = A-dot*x
        v = mu*v+Gt
        vd = (v*x).sum(axis=1, keepdims=True)
        v -= vd*x
        x = x-lr*v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _lp_polish_deadline(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _normr(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m-max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        try:
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x+res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _normr(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _full_lse_fg(yflat, N, alpha):
    Y = yflat.reshape(N, 4)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    X = Y/nrm
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    E = np.exp(alpha*(G-m))
    np.fill_diagonal(E, 0.0)
    Ssum = float(E.sum())*0.5
    if Ssum <= 0 or not np.isfinite(Ssum):
        return 1e6, np.zeros_like(yflat)
    f = m + np.log(Ssum)/alpha
    W = E/Ssum
    gradX = W @ X
    dot = (gradX*X).sum(axis=1, keepdims=True)
    gradY = (gradX-dot*X)/nrm
    return float(f), gradY.ravel()

def _lse_polish(x, alpha, maxiter, deadline=None):
    from scipy.optimize import minimize
    N = x.shape[0]
    y = (x/_normr(x)).ravel().copy()
    def fg(v):
        return _full_lse_fg(v, N, float(alpha))
    try:
        res = minimize(fg, y, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol':1e-14, 'gtol':1e-9, 'maxfun':50000})
        Y = res.x.reshape(N, 4)
        return _normr(Y)
    except Exception:
        return x

def solve():
    t0 = time.time()
    TLIM = 170.0
    rng = np.random.default_rng(292292)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.min(nr) > 1e-9:
                    cands.append(a/nr)
        except Exception:
            pass
    if not cands:
        Xb = rng.standard_normal((n, d)); Xb = _normr(Xb)
    else:
        vals = [_max_cos(c) for c in cands]
        Xb = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    # --- binary polyhedral discard check (24,48,120 cannot tile 292 with full free orbits) ---
    # 292 = 4*73; divisors 1,2,4,73,146,292. Orders 24,48,120 do not divide 292,
    # so any union needs small fixed-point orbits; we discard them as non-competitive
    # and focus on groups whose free orbit size divides 292.
    try:
        _ = (292 % 24, 292 % 48, 292 % 120)
    except Exception:
        pass
    # --- symmetry-restricted search from scratch ---
    G4 = np.array([[1,0,0,0],[0,1,0,0],[-1,0,0,0],[0,-1,0,0]], dtype=np.float64)
    L4 = _quat_left_mats(G4)
    sym_cands = []
    sym_deadline = t0+42.0
    # C4 quaternion, 73 seeds -> 292
    for rep in range(2):
        if time.time() > sym_deadline:
            break
        try:
            Xs, _ = _optim_sym(73, L4, rng, alphas=(50.0,200.0,700.0), iters=(90,90,110), deadline=sym_deadline)
            if Xs.shape != (292,4) or _has_dup(Xs, 1e-9):
                continue
            sym_cands.append((_max_cos(Xs), Xs.copy()))
        except Exception:
            continue
    # cyclic torus groups: order 73 (K=4), 146 (K=2), 292 (K=1)
    cyclic_specs = [(73,1,4),(73,2,4),(73,3,4),(73,7,4),(73,11,4),(146,1,2),(146,3,2),(292,1,1),(292,3,1)]
    for order, p, K in cyclic_specs:
        if time.time() > sym_deadline:
            break
        try:
            Ls = _cyclic_mats(order, p)
            Xs, _ = _optim_sym(K, Ls, rng, alphas=(50.0,200.0,800.0), iters=(120,120,140), deadline=sym_deadline)
            if Xs.shape != (292,4):
                continue
            # dedup
            G = Xs @ Xs.T; np.fill_diagonal(G, -2.0)
            if float(G.max()) > 1-1e-8:
                continue
            sym_cands.append((_max_cos(Xs), Xs.copy()))
        except Exception:
            continue
    sym_cands.sort(key=lambda t: t[0])
    sym_cands = sym_cands[:3]
    # expand best symmetric: break symmetry with tiny tangent perturbation + LP polish
    for v, Xs in list(sym_cands):
        if time.time() > t0+TLIM-50.0:
            break
        if v > best_val+0.005:
            continue
        try:
            pert = 1e-4*rng.standard_normal(Xs.shape)
            pert -= (pert*Xs).sum(axis=1, keepdims=True)*Xs
            Y = _normr(Xs+pert)
            Y = _lp_polish_deadline(Y, min(time.time()+5.0, t0+TLIM-38.0))
            vy = _max_cos(Y)
            if vy < best_val:
                X_best, best_val = Y.copy(), vy
            # also polish raw symmetric without break (stays symmetric subspace optimum)
            if v < best_val:
                Z = _lp_polish_deadline(Xs.copy(), min(time.time()+3.0, t0+TLIM-38.0))
                vz = _max_cos(Z)
                if vz < best_val:
                    X_best, best_val = Z.copy(), vz
        except Exception:
            continue
    # --- basin hopping with topology-changing kicks ---
    explore_end = t0+TLIM-22.0
    bases = [X_best.copy()] + [s.copy() for _, s in sym_cands]
    it = 0
    sigmas = [0.003, 0.008, 0.02, 0.05, 0.12]
    while time.time() < explore_end:
        it += 1
        rem = explore_end-time.time()
        if rem < 1.2:
            break
        # pick base: mostly champion, sometimes symmetric basin
        r = rng.random()
        if r < 0.70:
            base = X_best
        else:
            base = bases[int(rng.integers(len(bases)))]
        X = base.copy()
        # topology change: occasionally reinsert worst / rattler-like point
        if it % 7 == 0:
            try:
                G = X @ X.T; np.fill_diagonal(G, -2.0)
                # point with largest row-max (most jammed)
                rowmax = G.max(axis=1)
                idx = int(np.argmax(rowmax))
                # random far reinsertion: try few random dirs, keep farthest
                best_try = None; best_sc = 1e9
                for _ in range(8):
                    cand = rng.standard_normal(4); cand /= np.linalg.norm(cand)
                    sc = float((X @ cand).max())
                    # also avoid chosen idx
                    if sc < best_sc:
                        best_sc = sc; best_try = cand
                X[idx] = best_try
            except Exception:
                pass
        sigma = float(sigmas[int(rng.integers(len(sigmas)))])
        X = X+sigma*rng.standard_normal(X.shape)
        X = _normr(X)
        steps = 380 if rem > 12 else 220
        try:
            X = _anneal(X, steps, 20.0, 380.0, 0.06, 0.85)
        except Exception:
            continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        # polish if promising; threshold tight to save time
        if v < best_val+8e-05:
            dl = min(time.time()+2.2, explore_end)
            if dl-time.time() < 0.6:
                if v < best_val:
                    X_best, best_val = X.copy(), v
                    bases[0] = X_best.copy()
                continue
            try:
                Y = _lp_polish_deadline(X, dl)
                vy = _max_cos(Y)
            except Exception:
                continue
            if vy < best_val:
                X_best, best_val = Y.copy(), vy
                bases[0] = X_best.copy()
        elif v < best_val:
            X_best, best_val = X.copy(), v
            bases[0] = X_best.copy()
    # final reserve polish: LSE smooth + LP trust region
    try:
        if time.time() < t0+TLIM-10.0:
            # quick smooth equalization then LP
            Xs = _lse_polish(X_best, 1500.0, 120)
            if _max_cos(Xs) < _max_cos(X_best):
                X_best = Xs
    except Exception:
        pass
    try:
        if time.time() < t0+TLIM-8.0:
            X_best = _lp_polish_deadline(X_best, t0+TLIM-1.5)
    except Exception:
        pass
    X_best = _normr(X_best)
    return X_best

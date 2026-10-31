import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quat_left_mats(group):
    Ls = []
    for w, x, y, z in group:
        L = np.array([[w, -x, -y, -z],
                      [x, w, -z, y],
                      [y, z, w, -x],
                      [z, -y, x, w]], dtype=np.float64)
        Ls.append(L)
    return np.array(Ls)

def _expand_sym(S, Ls):
    K = S.shape[0]; G = Ls.shape[0]
    X = np.empty((K*G, 4), dtype=np.float64)
    for k in range(K):
        sk = S[k]
        for g in range(G):
            X[k*G+g] = Ls[g] @ sk
    return X

def _sym_fg(yflat, Ls, K, alpha):
    Y = yflat.reshape(K, 4)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    S = Y / nrm
    G = Ls.shape[0]
    N = K*G
    X = _expand_sym(S, Ls)
    Gr = X @ X.T
    np.fill_diagonal(Gr, -2.0)
    m = float(Gr.max())
    E = np.exp(alpha*(Gr - m))
    np.fill_diagonal(E, 0.0)
    Ssum = float(E.sum())*0.5 + 1e-300
    lse = m + np.log(Ssum)/alpha
    W = E / Ssum
    gradX = W @ X
    gradS = np.zeros_like(S)
    for k in range(K):
        base = k*G
        acc = np.zeros(4)
        for g in range(G):
            acc += Ls[g].T @ gradX[base+g]
        gradS[k] = acc
    dot = (gradS*S).sum(axis=1, keepdims=True)
    gradY = (gradS - dot*S)/nrm
    return float(lse), gradY.ravel()

def _optim_sym(K, Ls, rng, alphas, iters, deadline):
    from scipy.optimize import minimize
    Y0 = rng.standard_normal((K, 4))
    y = Y0.ravel()
    for alpha, mi in zip(alphas, iters):
        if time.time() > deadline-1.5:
            break
        def fg(v):
            return _sym_fg(v, Ls, K, alpha)
        try:
            res = minimize(fg, y, jac=True, method='L-BFGS-B',
                           options={'maxiter': mi, 'ftol': 1e-12, 'gtol': 1e-7, 'maxfun': 100000})
            y = res.x
        except Exception:
            break
    Y = y.reshape(K, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    X = _expand_sym(Y, Ls)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _has_dup(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return bool(G.max() > 1.0 - tol)

def _anneal(x, steps, beta0, beta1, lr0, mu):
    n = x.shape[0]
    v = np.zeros_like(x)
    if beta1 > beta0:
        lb = np.log(beta1/beta0)
    else:
        lb = 0.0
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
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
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
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p, 1), n*d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius*1.5, 0.05)
        else:
            radius *= 0.5
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
        Xb = rng.standard_normal((n, d)); Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        Xb = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    # symmetry restricted search from scratch (C4: 73 seeds -> 292, C2: 146 seeds)
    G4 = np.array([[1,0,0,0],[0,1,0,0],[-1,0,0,0],[0,-1,0,0]], dtype=np.float64)
    G2 = np.array([[1,0,0,0],[-1,0,0,0]], dtype=np.float64)
    L4 = _quat_left_mats(G4)
    L2 = _quat_left_mats(G2)
    sym_cands = []
    sym_deadline = t0+28.0
    for rep in range(2):
        if time.time() > sym_deadline:
            break
        try:
            Xs = _optim_sym(73, L4, rng, alphas=(50.0, 200.0, 600.0), iters=(70, 70, 90), deadline=sym_deadline)
            if Xs.shape == (292, 4) and not _has_dup(Xs, 1e-9):
                sym_cands.append((_max_cos(Xs), Xs.copy()))
        except Exception:
            continue
    if time.time() < sym_deadline-6.0:
        try:
            Xs = _optim_sym(146, L2, rng, alphas=(50.0, 200.0), iters=(50, 50), deadline=sym_deadline)
            if Xs.shape == (292, 4) and not _has_dup(Xs, 1e-9):
                sym_cands.append((_max_cos(Xs), Xs.copy()))
        except Exception:
            pass
    sym_cands.sort(key=lambda t: t[0])
    sym_cands = sym_cands[:2]
    # break symmetry + LP polish competitive sym candidates
    for v, Xs in list(sym_cands):
        if time.time() > t0+TLIM-45.0:
            break
        if v > best_val+0.005:
            continue
        try:
            pert = 1e-4*rng.standard_normal(Xs.shape)
            pert -= (pert*Xs).sum(axis=1, keepdims=True)*Xs
            Y = Xs+pert
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _lp_polish_deadline(Y, min(time.time()+5.0, t0+TLIM-35.0))
            vy = _max_cos(Y)
            if vy < best_val:
                X_best, best_val = Y.copy(), vy
            if v < best_val:
                Z = _lp_polish_deadline(Xs.copy(), min(time.time()+3.0, t0+TLIM-35.0))
                vz = _max_cos(Z)
                if vz < best_val:
                    X_best, best_val = Z.copy(), vz
        except Exception:
            continue
    bases = [X_best.copy()]
    for _, Xs in sym_cands:
        bases.append(Xs.copy())
    explore_end = t0+TLIM-26.0
    sigmas = [0.003, 0.007, 0.015, 0.035, 0.08, 0.16]
    it = 0
    while time.time() < explore_end:
        it += 1
        rem = explore_end-time.time()
        if rem < 1.0:
            break
        r = rng.random()
        if r < 0.75:
            base = X_best
        else:
            base = bases[int(rng.integers(len(bases)))]
        X = base.copy()
        # move type
        u = rng.random()
        if u < 0.15:
            # subset reshuffle: replace k worst/hole points
            k = int(rng.choice([4, 8, 12, 20]))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # worst indices by row max
            rm = G.max(axis=1)
            idx = np.argsort(rm)[-k:]
            # largest hole direction: sample random, keep farthest from X
            S = rng.standard_normal((2000, 4)); S /= np.linalg.norm(S, axis=1, keepdims=True)
            sc = S @ X.T
            m2 = sc.max(axis=1)
            order = np.argsort(m2)
            for j, ii in enumerate(idx):
                X[ii] = S[order[j % len(order)]]
            sigma = 0.01
            X += sigma*rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 500 if rem > 12 else 250
            try:
                X = _anneal(X, steps, 25.0, 400.0, 0.06, 0.85)
            except Exception:
                continue
        elif u < 0.30 and it % 3 == 0:
            # targeted: push worst 20 points harder
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            idx = np.argsort(rm)[-20:]
            sigma_big = 0.05+rng.random()*0.08
            sigma_small = 0.005
            N = rng.standard_normal(X.shape)*sigma_small
            N[idx] = rng.standard_normal((20, 4))*sigma_big
            # tangentize
            N -= (N*X).sum(axis=1, keepdims=True)*X
            X = X+N
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 450 if rem > 10 else 220
            try:
                X = _anneal(X, steps, 20.0, 350.0, 0.065, 0.88)
            except Exception:
                continue
        else:
            sigma = float(sigmas[int(rng.integers(len(sigmas)))])
            # occasional scaled tangent noise
            Xn = X+sigma*rng.standard_normal(X.shape)
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            X = Xn
            steps = 450 if rem > 12 else 220
            b0 = float(rng.choice([15.0, 25.0, 40.0]))
            b1 = float(rng.choice([300.0, 450.0, 700.0]))
            lr = float(rng.choice([0.05, 0.06, 0.08]))
            mu = float(rng.choice([0.82, 0.85, 0.9]))
            try:
                X = _anneal(X, steps, b0, b1, lr, mu)
            except Exception:
                continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        if v < best_val:
            # direct win: polish immediately
            dl = min(time.time()+2.5, explore_end)
            if dl-time.time() > 0.6:
                try:
                    Y = _lp_polish_deadline(X, dl)
                    vy = _max_cos(Y)
                    if vy < v:
                        X, v = Y, vy
                except Exception:
                    pass
            if v < best_val:
                X_best, best_val = X.copy(), float(v)
                bases[0] = X_best.copy()
                continue
        if v < best_val+3e-04:
            dl = min(time.time()+1.8, explore_end)
            if dl-time.time() < 0.5:
                continue
            try:
                Y = _lp_polish_deadline(X, dl)
                vy = _max_cos(Y)
            except Exception:
                continue
            if vy < best_val:
                X_best, best_val = Y.copy(), float(vy)
                bases[0] = X_best.copy()
    try:
        if time.time() < t0+TLIM-8.0:
            X_best = _lp_polish_deadline(X_best, t0+TLIM-1.5)
    except Exception:
        pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

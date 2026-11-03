import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

# ---------- symmetric C23 family ----------
def _tables(speeds, m=23):
    j = np.arange(m)[:, None]  # 23x1
    sp = np.array(speeds, dtype=float)[None, :]  # 1x4
    ang = 2.0 * np.pi * j * sp / 23.0
    return np.cos(ang), np.sin(ang)  # 23x4

def _build_X_sym(Y, C, S):
    # Y (k,8) assumed unit rows, C,S (m,4)
    k = Y.shape[0]
    m = C.shape[0]
    Y4 = Y.reshape(k, 4, 2)
    y0 = Y4[:, :, 0][:, None, :]  # k,1,4
    y1 = Y4[:, :, 1][:, None, :]
    Cc = C[None, :, :]
    Ss = S[None, :, :]
    X0 = y0 * Cc - y1 * Ss  # k,m,4
    X1 = y0 * Ss + y1 * Cc
    X4 = np.stack([X0, X1], axis=-1)  # k,m,4,2
    return X4.reshape(k * m, 8)

def _sym_fun(pflat, C, S, beta, k=6):
    p = pflat.reshape(k, 8)
    nr = np.sqrt(np.sum(p * p, axis=1, keepdims=True)) + 1e-30
    Y = p / nr
    X = _build_X_sym(Y, C, S)
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    Gv = np.sum(X[iu[0]] * X[iu[1]], axis=1)
    mv = Gv.max()
    E = np.exp(beta * (Gv - mv))
    Ssum = E.sum() + 1e-300
    f = float(mv + np.log(Ssum) / beta)
    # gradient
    W = np.zeros((n, n))
    W[iu] = E / Ssum
    W = W + W.T
    GX = W @ X  # n,8
    # back to Y
    m = C.shape[0]
    GX4 = GX.reshape(k, m, 4, 2)
    G0 = GX4[:, :, :, 0]  # k,m,4
    G1 = GX4[:, :, :, 1]
    Cc = C[None, :, :]
    Ss = S[None, :, :]
    gy0 = np.sum(G0 * Cc + G1 * Ss, axis=1)  # k,4
    gy1 = np.sum(-G0 * Ss + G1 * Cc, axis=1)
    GY = np.stack([gy0, gy1], axis=-1).reshape(k, 8)
    dot = np.sum(GY * Y, axis=1, keepdims=True)
    GT = GY - dot * Y
    Gp = GT / nr
    return f, Gp.ravel()

def _sym_opt(Y0, C, S, stages, deadline):
    from scipy.optimize import minimize
    k = Y0.shape[0]
    p = Y0.ravel().copy()
    for beta, maxiter in stages:
        if time.time() > deadline:
            break
        def fun(pf):
            return _sym_fun(pf, C, S, float(beta), k=k)
        try:
            res = minimize(fun, p, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 12})
            p = res.x
        except Exception:
            break
    pmat = p.reshape(k, 8)
    pmat = _norm_rows(pmat)
    return pmat

# ---------- full LSE ----------
def _full_fun(pflat, n, d, beta):
    Ym = pflat.reshape(n, d)
    nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
    X = Ym / nr
    iu0 = np.triu_indices(n, 1)
    # compute Gram via chunks to save memory? n=138 fine
    G = X @ X.T
    c = G[iu0]
    mv = c.max()
    E = np.exp(beta * (c - mv))
    Ssum = E.sum() + 1e-300
    f = float(mv + np.log(Ssum) / beta)
    W = np.zeros((n, n))
    W[iu0] = E / Ssum
    W = W + W.T
    GX = W @ X
    dot = np.sum(GX * X, axis=1, keepdims=True)
    Gt = GX - dot * X
    Gp = Gt / nr
    return f, Gp.ravel()

def _full_lbfgs(X0, beta, maxiter, deadline):
    from scipy.optimize import minimize
    n, d = X0.shape
    if time.time() > deadline:
        return _norm_rows(X0)
    Y = _norm_rows(X0.copy())
    p = Y.ravel().copy()
    def fun(pf):
        return _full_fun(pf, n, d, float(beta))
    try:
        res = minimize(fun, p, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 12})
        Y = _norm_rows(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm_rows(Y)

def _lp_polish(X, deadline, radius0=8e-3, cap=2800):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm_rows(X)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p_ = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p_), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p_, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p_, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    import time
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm_rows(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)

    # symmetry search
    speeds_list = [(1, 3, 5, 11), (1, 2, 4, 9), (1, 2, 3, 7), (1, 3, 4, 10), (2, 3, 5, 8)]
    sym_stages = [(50.0, 120), (120.0, 140), (220.0, 160), (350.0, 160), (500.0, 180)]
    sym_cands = []
    # limit sym time to ~55s
    sym_deadline = t0 + 60.0
    trial = 0
    for sp in speeds_list:
        if time.time() > sym_deadline - 2.0 or time.time() > deadline - 40.0:
            break
        C, S = _tables(sp, m=23)
        for rep in range(2):
            if time.time() > sym_deadline - 2.0:
                break
            trial += 1
            Y0 = rng.standard_normal(size=(6, 8))
            Y0 = _norm_rows(Y0)
            try:
                Yopt = _sym_opt(Y0, C, S, sym_stages, min(sym_deadline, deadline - 40.0))
                Xsym = _build_X_sym(Yopt, C, S)
                Xsym = _norm_rows(Xsym)
                v = _max_cos(Xsym)
                sym_cands.append((v, Xsym, sp))
            except Exception:
                continue
    sym_cands.sort(key=lambda t: t[0])
    # release symmetry + refine top candidates
    full_betas = [(80.0, 150), (160.0, 180), (300.0, 220), (500.0, 250)]
    # also include incumbent shake as fallback candidate
    if len(sym_cands) == 0:
        sym_cands = [(best_val, best.copy(), None)]
    nref = min(3, len(sym_cands))
    for ci in range(nref):
        if time.time() > deadline - 18.0:
            break
        _, Xs, _ = sym_cands[ci]
        for amp in [0.004, 0.012]:
            if time.time() > deadline - 16.0:
                break
            try:
                Xc = Xs.copy()
                noise = rng.standard_normal(size=Xc.shape)
                proj = np.sum(noise * Xc, axis=1, keepdims=True)
                noise = noise - proj * Xc
                nn = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-30
                noise = noise / nn
                Xc = Xc + amp * noise
                Xc = _norm_rows(Xc)
                for beta, mit in full_betas:
                    if time.time() > deadline - 12.0:
                        break
                    Xc = _full_lbfgs(Xc, beta, mit, deadline - 10.0)
                # quick LP
                Xc = _lp_polish(Xc, min(deadline - 4.0, time.time() + 4.0))
                v = _max_cos(Xc)
                if v < best_val - 1e-12:
                    best_val = v
                    best = Xc.copy()
            except Exception:
                continue
    # final polish of incumbent
    try:
        if time.time() < deadline - 1.0:
            best = _lp_polish(best, deadline - 0.3)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = _norm_rows(best)
    return best

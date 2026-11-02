import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

_ONES = np.ones(15, dtype=np.float64) / np.sqrt(15.0)

def _expand(Gn):
    # Gn (5,15) assumed normalized rows
    X = np.empty((76, 15), dtype=np.float64)
    for a in range(5):
        g = Gn[a]
        for s in range(15):
            X[a * 15 + s] = np.roll(g, s)
    X[75] = _ONES
    return X

def _lse_val_grad(gflat, beta):
    g = gflat.reshape(5, 15)
    nr = np.linalg.norm(g, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Gn = g / nr
    X = _expand(Gn)
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    m = float(G.max())
    # exp
    E = np.exp(beta * (G - m))
    np.fill_diagonal(E, 0.0)
    sumE = float(E.sum())
    if not np.isfinite(sumE) or sumE <= 0:
        return 1.0, np.zeros_like(gflat)
    Shalf = sumE * 0.5
    f = m + np.log(Shalf) / beta
    # gradient w.r.t X
    P = (2.0 / sumE) * E  # symmetric, diag 0, sum_{i<j} p =1
    GrX = P @ X  # (76,15)
    # pull back to Gn
    GrGn = np.zeros_like(Gn)
    for a in range(5):
        acc = np.zeros(15)
        base = a * 15
        for s in range(15):
            acc += np.roll(GrX[base + s], -s)
        GrGn[a] = acc
    # chain Gn = g/nr : d/dg = (I - Gn Gn^T)/nr * d/dGn
    Grg = np.zeros_like(g)
    for a in range(5):
        v = GrGn[a]
        gn = Gn[a]
        Grg[a] = (v - np.dot(v, gn) * gn) / float(nr[a, 0])
    # penalize near-collision of generator norms? not needed
    return float(f), Grg.ravel()

def _sym_opt(rng, t0, t_lim, n_restart=4):
    from scipy.optimize import minimize
    best_f = 1e9
    best_Gn = None
    best_mc = 1e9
    betas = [50.0, 150.0, 450.0]
    maxiters = [250, 250, 300]
    for r in range(n_restart):
        if time.time() - t0 > t_lim:
            break
        g0 = rng.standard_normal(size=(5, 15))
        g0 = _norm(g0)
        # small spread to avoid exact duplicates
        gflat = g0.ravel().copy()
        for bi, beta in enumerate(betas):
            if time.time() - t0 > t_lim:
                break
            mi = maxiters[bi]
            def fun(gf, b=beta):
                f, gr = _lse_val_grad(gf, b)
                return f, gr
            try:
                res = minimize(fun, gflat, method='L-BFGS-B', jac=True,
                               options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-10})
                gflat = res.x
            except Exception:
                break
        # evaluate expanded maxcos
        try:
            gg = gflat.reshape(5, 15)
            Gn = _norm(gg)
            X = _expand(Gn)
            mc = _maxcos(X)
        except Exception:
            continue
        if mc < best_mc:
            best_mc = mc
            best_Gn = Gn.copy()
            best_f = mc
    if best_Gn is None:
        g0 = rng.standard_normal(size=(5, 15))
        best_Gn = _norm(g0)
        X = _expand(best_Gn)
        best_mc = _maxcos(X)
    else:
        X = _expand(best_Gn)
    return X, float(best_mc)

def _anneal(X, steps, T0, T1, lr0, lr1, t0, t_lim):
    n = X.shape[0]
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for k in range(steps):
        if (k & 63) == 0 and time.time() - t0 > t_lim:
            break
        f = k / max(1, steps - 1)
        T = np.exp(lT0 + (lT1 - lT0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            continue
        Gr = (E @ X) / S
        # tangential
        Gr = Gr - (np.sum(Gr * X, axis=1, keepdims=True)) * X
        rms = float(np.sqrt(np.sum(Gr * Gr) / n)) + 1e-12
        Gr = Gr / rms
        X = X - lr * Gr
        X = _norm(X)
    return X

def _contact_polish(X, steps, tol0, tol1, lr0, lr1, t0, t_lim):
    n = X.shape[0]
    lto = np.log(tol0); lt1 = np.log(tol1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    cur = _maxcos(X)
    for k in range(steps):
        if (k & 63) == 0 and time.time() - t0 > t_lim:
            break
        f = k / max(1, steps - 1)
        tol = np.exp(lto + (lt1 - lto) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= m - tol).astype(np.float64)
        # gradient of max over contacts: sum_j W_kj (x_j - G_kj x_k)
        Gr = W @ X - (np.sum(W * G, axis=1, keepdims=True)) * X
        rms = float(np.sqrt(np.sum(Gr * Gr) / n)) + 1e-12
        if rms < 1e-14:
            continue
        Gr = Gr / rms
        Y = X - lr * Gr
        Y = _norm(Y)
        Gy = Y @ Y.T
        np.fill_diagonal(Gy, -2.0)
        my = float(Gy.max())
        if my <= cur + tol * 0.25:
            X = Y
            cur = my
    return X

def solve():
    t0 = time.time()
    T_LIM = 150.0
    rng = np.random.default_rng(42)
    try:
        Xb = np.load("best.npy").astype(np.float64)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(np.float64)
    except Exception:
        Xr = None
    cands = []
    if Xb is not None and Xb.shape == (76, 15):
        cands.append(_norm(Xb))
    if Xr is not None and Xr.shape == (76, 15):
        cands.append(_norm(Xr))
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal(size=(76, 15))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best = float(vals[bi])

    # 1) symmetry-constrained search
    Xsym, mcsym = _sym_opt(rng, t0, T_LIM - 75.0, n_restart=4)

    # 2) release symmetry: tiny break + short softmax anneal + contact polish
    Xrel = Xsym.copy()
    # break exact C15 symmetry slightly
    Xrel = Xrel + 1e-4 * rng.standard_normal(size=Xrel.shape)
    Xrel = _norm(Xrel)
    # short anneal with two stages if time permits
    if time.time() - t0 < T_LIM - 50.0:
        Xrel = _anneal(Xrel, 1800, 0.016, 0.0020, 2.5, 0.15, t0, T_LIM - 25.0)
    if time.time() - t0 < T_LIM - 25.0:
        Xrel = _anneal(Xrel, 1200, 0.0025, 0.0004, 0.6, 0.06, t0, T_LIM - 12.0)
    if time.time() - t0 < T_LIM - 10.0:
        Xrel = _contact_polish(Xrel, 700, 0.012, 0.0006, 0.004, 0.0004, t0, T_LIM)
    vrel = _maxcos(Xrel)
    if vrel < best:
        best = vrel
        Xbest = Xrel.copy()
    # also try a second release from a perturbed sym with different seed path
    if time.time() - t0 < T_LIM - 15.0:
        try:
            X2 = Xsym + 0.01 * rng.standard_normal(size=Xsym.shape)
            X2 = _norm(X2)
            X2 = _anneal(X2, 900, 0.012, 0.0015, 2.0, 0.12, t0, T_LIM - 5.0)
            X2 = _contact_polish(X2, 400, 0.010, 0.0008, 0.003, 0.0004, t0, T_LIM)
            v2 = _maxcos(X2)
            if v2 < best:
                best = v2
                Xbest = X2.copy()
        except Exception:
            pass

    Xbest = _norm(np.nan_to_num(Xbest))
    return Xbest

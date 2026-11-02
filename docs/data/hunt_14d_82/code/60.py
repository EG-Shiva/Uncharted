import time
import numpy as np
from scipy.optimize import minimize

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _eval(X, iu):
    G = X @ X.T
    c = G[iu]
    m = float(c.max())
    if len(c) >= 100:
        part = np.argpartition(-c, 100 - 1)[:100]
        mean100 = float(c[part].mean())
    else:
        mean100 = float(c.mean())
    if len(c) >= 500:
        part2 = np.argpartition(-c, 500 - 1)[:500]
        mean500 = float(c[part2].mean())
    else:
        mean500 = float(c.mean())
    return m, mean100, mean500, c

def _lex_better(nm, nmean, bm, bmean):
    if nm < bm - 1e-12:
        return True
    if abs(nm - bm) <= 1e-12 and nmean < bmean - 1e-12:
        return True
    return False

def _lse_polish(X0, beta, maxiter):
    n, d = X0.shape
    Y0 = X0.copy().ravel()
    def fun_grad(Yf):
        Y = Yf.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        G = X @ X.T
        # off-diag max
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        S = beta * (G - m)
        np.clip(S, -50.0, 0.0, out=S)
        E = np.exp(S)
        np.fill_diagonal(E, 0.0)
        sumE = float(E.sum() * 0.5)
        if not np.isfinite(sumE) or sumE <= 0:
            sumE = 1.0
        f = m + np.log(sumE + 1e-300) / beta
        W = E / sumE
        Gx = W @ X
        dot = np.sum(Gx * X, axis=1, keepdims=True)
        Gx_t = Gx - dot * X
        Gy = Gx_t / rn
        return float(f), Gy.ravel()
    try:
        res = minimize(fun_grad, Y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-14,
                                'gtol': 1e-7, 'maxcor': 15, 'maxfun': int(maxiter*10)})
        Yf = res.x
        Y = Yf.reshape(n, d)
        X = _norm(Y)
        if not np.all(np.isfinite(X)):
            return _norm(X0)
        return X
    except Exception:
        return _norm(X0)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 160.0
    n, d = 82, 14
    iu = np.triu_indices(n, 1)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        Xbest = _norm(rng0.standard_normal((n, d)))
    else:
        def _mc(X):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            return float(np.max(G))
        cands.sort(key=_mc)
        Xbest = cands[0].copy()
    Xbest = _norm(Xbest)
    bm, bmean, _, _ = _eval(Xbest, iu)
    rng = np.random.default_rng(0)
    betas = [250.0, 700.0, 1800.0, 4500.0, 11000.0]
    iters = [120, 120, 120, 100, 80]
    # first polish incumbent without kick
    # quick initial anneal
    Xcur = Xbest.copy()
    for bi, beta in enumerate(betas):
        if time.time() > DEADLINE - 8.0:
            break
        Xcur = _lse_polish(Xcur, beta, iters[bi])
        m, mean, _, _ = _eval(Xcur, iu)
        if _lex_better(m, mean, bm, bmean):
            bm, bmean = m, mean
            Xbest = Xcur.copy()
    # basin hopping with kicks
    r = 0
    sigmas = [0.006, 0.012, 0.020, 0.008, 0.015, 0.010, 0.025, 0.005]
    while time.time() < DEADLINE - 5.0:
        r += 1
        if r > 60:
            break
        sigma = sigmas[(r - 1) % len(sigmas)]
        # kick from best
        Z = rng.standard_normal((n, d))
        # tangent projection
        dotz = np.sum(Z * Xbest, axis=1, keepdims=True)
        Zt = Z - dotz * Xbest
        # normalize kick scale per point to sigma
        zn = np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-300
        Zt = Zt / zn * sigma * np.sqrt(d - 1)
        # small global rotation component for isotropy
        Xtry = _norm(Xbest + Zt)
        # anneal
        for bi, beta in enumerate(betas):
            if time.time() > DEADLINE - 5.0:
                break
            # fewer iters for kicked runs to save time after first few restarts
            mi = iters[bi]
            if r > 4 and bi < 2:
                mi = 60
            Xtry = _lse_polish(Xtry, beta, mi)
            # early lex check on intermediate beta? only max matters broadly
            # keep going through anneal regardless
        m, mean, _, _ = _eval(Xtry, iu)
        if _lex_better(m, mean, bm, bmean):
            bm, bmean = m, mean
            Xbest = Xtry.copy()
        # extra intensification: if improved, do one more high-beta polish immediately
        if time.time() > DEADLINE - 5.0:
            break
    Xbest = _norm(np.asarray(Xbest, dtype=float))
    return Xbest

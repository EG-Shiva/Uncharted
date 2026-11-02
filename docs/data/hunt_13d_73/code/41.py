import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _expand_C12(T):
    # T (6,13) unit rows -> X (73,13)
    X = np.empty((73, 13), dtype=float)
    for k in range(6):
        a = T[k, :12]
        b = float(T[k, 12])
        for s in range(12):
            X[k*12+s, :12] = np.roll(a, s)
            X[k*12+s, 12] = b
    X[72, :] = 0.0
    X[72, 12] = 1.0
    return X

def _expand_C13(Tt, Tf):
    # Tt (5,13), Tf (8,13) unit rows -> X (73,13): 5*13=65 orbit + 8 free
    X = np.empty((73, 13), dtype=float)
    for k in range(5):
        a = Tt[k, :]
        for s in range(13):
            X[k*13+s, :] = np.roll(a, s)
    X[65:, :] = Tf
    return X

def _lp_polish(X, seconds, hard_end=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if hard_end is not None and time.time() > hard_end:
            break
        if radius < 1e-14:
            break
        c = (X @ X.T)[iu]
        m = float(c.max())
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    SYM_END = t0 + 55.0
    SYM2_END = t0 + 85.0
    GLOB_END = t0 + 158.0
    n, d = 73, 13
    # load incumbent
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = np.random.RandomState(0).randn(n, d)
    if xb.shape != (n, d):
        tmp = np.random.RandomState(0).randn(n, d)
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()

    iu73 = np.triu_indices(73, 1)

    # ---------- C12 symmetric optimization ----------
    cands = []  # list of (val, X)
    def c12_fun(Yflat, alpha):
        Y = Yflat.reshape(6, 13)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        T = Y / rn
        X = _expand_C12(T)
        G = X @ X.T
        dots = G[iu73]
        M = float(dots.max())
        e = np.exp(alpha*(dots - M))
        S = float(e.sum())
        loss = M + np.log(S)/alpha
        p = e / S
        W = np.zeros((73, 73)); W[iu73] = p; W = W + W.T
        gX = W @ X
        gT = np.zeros((6, 13))
        for k in range(6):
            acc = np.zeros(12); acc13 = 0.0
            for s in range(12):
                g = gX[k*12+s]
                acc += np.roll(g[:12], -s)
                acc13 += g[12]
            gT[k, :12] = acc; gT[k, 12] = acc13
        dot = (gT*T).sum(axis=1, keepdims=True)
        gY = (gT - dot*T) / rn
        return float(loss), gY.ravel().astype(float)

    rng_sym = np.random.RandomState(20261102)
    alphas_sym = [60.0, 150.0, 350.0]
    iters_sym = [100, 100, 120]
    nseeds_c12 = 5
    for si in range(nseeds_c12):
        if time.time() > SYM_END or time.time() > HARD_END - 60:
            break
        Y0 = rng_sym.randn(6, 13)
        Y = Y0.ravel().copy()
        for ai, alpha in enumerate(alphas_sym):
            if time.time() > SYM_END or time.time() > HARD_END - 60:
                break
            def fun(Yf, _a=alpha):
                return c12_fun(Yf, _a)
            try:
                res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                               options={"maxiter": iters_sym[ai], "ftol": 1e-12, "gtol": 1e-6, "maxcor": 8})
                Y = res.x
            except Exception:
                break
        try:
            Ym = Y.reshape(6, 13)
            Ym = Ym / np.linalg.norm(Ym, axis=1, keepdims=True)
            Xc = _expand_C12(Ym)
            v = _max_cos(Xc)
            cands.append((v, Xc.copy()))
            if v < best_val:
                best_val = v; Xbest = Xc.copy()
        except Exception:
            pass

    # ---------- C13 five-orbit + 8 free ----------
    iu = iu73
    def c13_fun(Yflat, alpha):
        Yt = Yflat[:65].reshape(5, 13)
        Yf = Yflat[65:].reshape(8, 13)
        rnt = np.maximum(np.linalg.norm(Yt, axis=1, keepdims=True), 1e-12)
        rnf = np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
        Tt = Yt / rnt; Tf = Yf / rnf
        X = _expand_C13(Tt, Tf)
        G = X @ X.T
        dots = G[iu]
        M = float(dots.max())
        e = np.exp(alpha*(dots - M))
        S = float(e.sum())
        loss = M + np.log(S)/alpha
        p = e / S
        W = np.zeros((73, 73)); W[iu] = p; W = W + W.T
        gX = W @ X
        gTt = np.zeros((5, 13))
        for k in range(5):
            acc = np.zeros(13)
            for s in range(13):
                acc += np.roll(gX[k*13+s], -s)
            gTt[k, :] = acc
        gTf = gX[65:, :]
        gYt = (gTt - (gTt*Tt).sum(axis=1, keepdims=True)*Tt) / rnt
        gYf = (gTf - (gTf*Tf).sum(axis=1, keepdims=True)*Tf) / rnf
        gY = np.concatenate([gYt.ravel(), gYf.ravel()])
        return float(loss), gY.astype(float)

    rng2 = np.random.RandomState(777)
    alphas2 = [60.0, 160.0, 350.0]
    iters2 = [80, 80, 100]
    for si in range(3):
        if time.time() > SYM2_END or time.time() > HARD_END - 55:
            break
        Y0 = np.concatenate([rng2.randn(5*13), rng2.randn(8*13)])
        Y = Y0.copy()
        for ai, alpha in enumerate(alphas2):
            if time.time() > SYM2_END or time.time() > HARD_END - 55:
                break
            def fun(Yf, _a=alpha):
                return c13_fun(Yf, _a)
            try:
                res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                               options={"maxiter": iters2[ai], "ftol": 1e-12, "gtol": 1e-6, "maxcor": 8})
                Y = res.x
            except Exception:
                break
        try:
            Yt = Y[:65].reshape(5, 13); Yf = Y[65:].reshape(8, 13)
            Yt /= np.linalg.norm(Yt, axis=1, keepdims=True)
            Yf /= np.linalg.norm(Yf, axis=1, keepdims=True)
            Xc = _expand_C13(Yt, Yf)
            v = _max_cos(Xc)
            cands.append((v, Xc.copy()))
            if v < best_val:
                best_val = v; Xbest = Xc.copy()
        except Exception:
            pass

    # ---------- release symmetry ----------
    # pick up to 3 best candidates + incumbent
    cands_sorted = sorted(cands, key=lambda t: t[0])[:3]
    bases = [X for _, X in cands_sorted]
    bases.append(xb.copy())
    # dedupe: keep at most 3
    if len(bases) > 3:
        # keep best symmetric + incumbent + second best
        bases = bases[:2] + [bases[-1]]
    sigmas = [3e-4, 1e-3, 3e-3, 1e-2]

    def full_fun(Yflat, alpha):
        Y = Yflat.reshape(73, 13)
        rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / rn
        G = X @ X.T
        dots = G[iu73]
        M = float(dots.max())
        e = np.exp(alpha*(dots - M))
        S = float(e.sum())
        loss = M + np.log(S)/alpha
        p = e / S
        W = np.zeros((73, 73)); W[iu73] = p; W = W + W.T
        gX = W @ X
        dg = (gX*X).sum(axis=1, keepdims=True)
        gY = (gX - dg*X) / rn
        return float(loss), gY.ravel().astype(float)

    rng3 = np.random.RandomState(4242)
    # budget full opts
    for bi, base in enumerate(bases):
        for sgi, sig in enumerate(sigmas):
            if time.time() > GLOB_END or time.time() > HARD_END - 30:
                break
            # only try subset to save time: 2 sigmas per base
            if sgi >= 2 and (bi > 0):
                break
            # cycle sigmas
            s = sigmas[(bi + sgi) % len(sigmas)]
            Xn = base + s * rng3.randn(73, 13)
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            Y = Xn.ravel().copy()
            for alpha, mi in [(100.0, 60), (300.0, 60)]:
                if time.time() > GLOB_END or time.time() > HARD_END - 30:
                    break
                def fun(Yf, _a=alpha):
                    return full_fun(Yf, _a)
                try:
                    res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                                   options={"maxiter": mi, "ftol": 1e-12, "gtol": 1e-5, "maxcor": 8})
                    Y = res.x
                except Exception:
                    break
            try:
                Ym = Y.reshape(73, 13)
                Ym /= np.linalg.norm(Ym, axis=1, keepdims=True)
                v = _max_cos(Ym)
                if v < best_val:
                    best_val = v; Xbest = Ym.copy()
            except Exception:
                pass
        if time.time() > GLOB_END or time.time() > HARD_END - 30:
            break

    # ---------- final LP polish on best ----------
    try:
        remain = HARD_END - time.time() - 1.0
        if remain > 2.0:
            secs = min(remain, 14.0)
            Xp = _lp_polish(Xbest, secs, hard_end=HARD_END-0.5)
            vp = _max_cos(Xp)
            if vp < best_val:
                Xbest = Xp; best_val = vp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

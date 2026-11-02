import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _expand_C12(T):
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
    X = np.empty((73, 13), dtype=float)
    for k in range(5):
        a = Tt[k, :]
        for s in range(13):
            X[k*13+s, :] = np.roll(a, s)
    X[65:, :] = Tf
    return X

def _lp_polish(X, seconds, hard_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 8e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < hard_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0*radius, 2e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        csel = c[sel]
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cI = I[:, None]*d + np.arange(d)
        cJ = J[:, None]*d + np.arange(d)
        ct = np.full((p, 1), n*d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            X, best = y, v
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SYM_END = t0 + 45.0
    SYM2_END = t0 + 68.0
    GLOB_END = t0 + 155.0
    n, d = 73, 13
    # load incumbent
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    cc = _max_cos(v)
                    if cc < best_file:
                        best_file = cc
                        Xrec = v.copy()
        except Exception:
            pass
    if Xrec is None:
        rs0 = np.random.RandomState(0)
        Xrec = rs0.randn(n, d)
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)
    iu73 = np.triu_indices(73, 1)

    # ---- C12 symmetric LSE ----
    def c12_fun(Yflat, alpha):
        Y = Yflat.reshape(6, 13)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        T = Y / rn
        X = _expand_C12(T)
        dots = (X @ X.T)[iu73]
        M = float(dots.max())
        e = np.exp(alpha*(dots - M))
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            return M, np.zeros_like(Yflat)
        loss = M + np.log(S)/alpha
        p = e / S
        W = np.zeros((73, 73)); W[iu73] = p; W = W + W.T
        gX = W @ X
        gT = np.zeros((6, 13))
        for k in range(6):
            acc = np.zeros(12); acc13 = 0.0
            base = k*12
            for s in range(12):
                g = gX[base+s]
                acc += np.roll(g[:12], -s)
                acc13 += g[12]
            gT[k, :12] = acc; gT[k, 12] = acc13
        dot = (gT*T).sum(axis=1, keepdims=True)
        gY = (gT - dot*T) / rn
        return float(loss), gY.ravel().astype(float)

    def c13_fun(Yflat, alpha):
        Y = Yflat.reshape(13, 13)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        TT = Y / rn
        Tt = TT[:5, :]; Tf = TT[5:, :]
        X = _expand_C13(Tt, Tf)
        dots = (X @ X.T)[iu73]
        M = float(dots.max())
        e = np.exp(alpha*(dots - M))
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            return M, np.zeros_like(Yflat)
        loss = M + np.log(S)/alpha
        p = e / S
        W = np.zeros((73, 73)); W[iu73] = p; W = W + W.T
        gX = W @ X
        gTT = np.zeros((13, 13))
        for k in range(5):
            acc = np.zeros(13)
            base = k*13
            for s in range(13):
                acc += np.roll(gX[base+s], -s)
            gTT[k, :] = acc
        gTT[5:, :] = gX[65:, :]
        dot = (gTT*TT).sum(axis=1, keepdims=True)
        gY = (gTT - dot*TT) / rn
        return float(loss), gY.ravel().astype(float)

    def full_fun(Yflat, alpha):
        Y = Yflat.reshape(73, 13)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        dots = (X @ X.T)[iu73]
        M = float(dots.max())
        e = np.exp(alpha*(dots - M))
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            return M, np.zeros_like(Yflat)
        loss = M + np.log(S)/alpha
        p = e / S
        W = np.zeros((73, 73)); W[iu73] = p; W = W + W.T
        gX = W @ X
        sprod = (gX*X).sum(axis=1, keepdims=True)
        gY = (gX - sprod*X) / rn
        return float(loss), gY.ravel().astype(float)

    sym_cands = []
    # C12 seeds
    alphas_sym = [50.0, 150.0, 350.0]
    iters_sym = [60, 60, 80]
    try:
        for si in range(6):
            if time.time() > SYM_END or time.time() > HARD_END - 60:
                break
            rs = np.random.RandomState(1000+si)
            Y = rs.randn(6, 13).ravel()
            for ai, al in enumerate(alphas_sym):
                if time.time() > SYM_END or time.time() > HARD_END - 60:
                    break
                def fun(Yf, _a=al):
                    return c12_fun(Yf, _a)
                try:
                    res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                                   options={"maxiter": iters_sym[ai], "ftol": 1e-12, "gtol": 1e-6, "maxcor": 8})
                    Y = res.x
                except Exception:
                    break
            try:
                Ym = Y.reshape(6, 13)
                Tm = Ym / np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
                Xs = _expand_C12(Tm)
                Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
                vs = _max_cos(Xs)
                sym_cands.append((vs, Xs.copy()))
                if vs < best:
                    best = vs; Xbest = Xs.copy()
            except Exception:
                pass
    except Exception:
        pass
    # C13 seeds
    try:
        for si in range(4):
            if time.time() > SYM2_END or time.time() > HARD_END - 55:
                break
            rs = np.random.RandomState(2000+si)
            Y = rs.randn(13, 13).ravel()
            for ai, al in enumerate(alphas_sym):
                if time.time() > SYM2_END or time.time() > HARD_END - 55:
                    break
                def fun2(Yf, _a=al):
                    return c13_fun(Yf, _a)
                try:
                    res = minimize(fun2, Y, jac=True, method="L-BFGS-B",
                                   options={"maxiter": iters_sym[ai], "ftol": 1e-12, "gtol": 1e-6, "maxcor": 8})
                    Y = res.x
                except Exception:
                    break
            try:
                Ym = Y.reshape(13, 13)
                TT = Ym / np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
                Xs = _expand_C13(TT[:5, :], TT[5:, :])
                Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
                vs = _max_cos(Xs)
                sym_cands.append((vs, Xs.copy()))
                if vs < best:
                    best = vs; Xbest = Xs.copy()
            except Exception:
                pass
    except Exception:
        pass

    sym_cands.sort(key=lambda t: t[0])
    # keep top few distinct
    top_sym = [x for _, x in sym_cands[:3]]

    # ---- break symmetry: unconstrained annealed L-BFGS ----
    starts = []
    rsG = np.random.RandomState(7)
    # perturbed symmetric tops
    sigmas = [3e-4, 1e-3, 3e-3, 1e-2]
    for xi, Xs in enumerate(top_sym):
        for sj, sg in enumerate(sigmas[:2] if xi > 0 else sigmas):
            if len(starts) >= 6:
                break
            # deterministic noise
            rs = np.random.RandomState(3000+xi*10+sj)
            X0 = Xs + rs.randn(n, d)*sg
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            starts.append(X0)
        if len(starts) >= 6:
            break
    # one shake of incumbent to allow local rebalance
    if len(starts) < 7 and time.time() < GLOB_END:
        rs = np.random.RandomState(999)
        X0 = Xrec + rs.randn(n, d)*0.002
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        starts.append(X0)

    alphas_full = [80.0, 250.0, 600.0]
    iters_full = [90, 90, 110]
    try:
        for sidx, X0 in enumerate(starts):
            if time.time() > GLOB_END or time.time() > HARD_END - 25:
                break
            Y = (X0*1.0).ravel().copy()
            for ai, al in enumerate(alphas_full):
                if time.time() > GLOB_END or time.time() > HARD_END - 25:
                    break
                def ff(Yf, _a=al):
                    return full_fun(Yf, _a)
                try:
                    res = minimize(ff, Y, jac=True, method="L-BFGS-B",
                                   options={"maxiter": iters_full[ai], "ftol": 1e-12, "gtol": 1e-7, "maxcor": 10})
                    Y = res.x
                except Exception:
                    break
            try:
                Ym = Y.reshape(n, d)
                Xm = Ym / np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
                vm = _max_cos(Xm)
                if vm < best and np.all(np.isfinite(Xm)):
                    best = vm; Xbest = Xm.copy()
            except Exception:
                pass
    except Exception:
        pass

    # final LP polish
    if time.time() < HARD_END - 4.0:
        try:
            remain = HARD_END - time.time() - 1.0
            if remain > 2.0:
                Xp, vp = _lp_polish(Xbest, seconds=min(remain, 14.0), hard_end=HARD_END-0.5)
                if vp < best and np.all(np.isfinite(Xp)):
                    best = vp; Xbest = Xp.copy()
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xbest)):
        Xbest = Xrec / np.linalg.norm(Xrec, axis=1, keepdims=True)
    return Xbest

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

def solve():
    import time
    import numpy as np
    t0 = time.time()
    deadline = t0 + 170.0
    # fallback loads
    def _load(p):
        try:
            a = np.load(p).astype(float)
            if a.shape == (76, 15):
                a = _norm(a)
                return a
        except Exception:
            pass
        return None
    xb = _load("best.npy")
    xr = _load("record.npy")
    fb = None
    fbv = 1e9
    if xb is not None:
        fb = xb.copy(); fbv = _maxcos(fb)
    if xr is not None:
        v = _maxcos(xr)
        if v < fbv:
            fb = xr.copy(); fbv = v
    if fb is None:
        rng0 = np.random.default_rng(0)
        fb = _norm(rng0.normal(size=(76, 15)))
        fbv = _maxcos(fb)

    n, d = 76, 15
    ONES = np.ones(15) / np.sqrt(15.0)
    iu0, iu1 = np.triu_indices(76, 1)
    ar = np.arange(15)
    ROLL = (ar[None, :] - ar[:, None]) % 15
    ADD = (ar[None, :] + ar[:, None]) % 15
    from scipy.optimize import minimize

    def expand(Gn):
        X = np.empty((76, 15))
        for k in range(5):
            X[k*15:(k+1)*15] = Gn[k][ROLL]
        X[75] = ONES
        return X

    # symmetric L-BFGS-B search
    sym_cands = []  # list of (v, X)
    rng = np.random.default_rng(20260601)
    # 6-8 seeds, beta 200-400
    n_restart = 7
    sym_deadline = t0 + 55.0
    for r in range(n_restart):
        if time.time() > sym_deadline:
            break
        P0 = rng.normal(size=(5, 15))
        # normalize rows to avoid scale issues
        P0 = P0 / np.linalg.norm(P0, axis=1, keepdims=True)
        P0 = P0.ravel()
        beta = 200.0 if (r % 2 == 0) else 350.0
        if r == 6:
            beta = 300.0
        def fun(flat, _beta=beta):
            Pm = flat.reshape(5, 15)
            ns = np.linalg.norm(Pm, axis=1)
            ns = np.maximum(ns, 1e-12)
            Gn = Pm / ns[:, None]
            X = np.empty((76, 15))
            for k in range(5):
                X[k*15:(k+1)*15] = Gn[k][ROLL]
            X[75] = ONES
            G = X @ X.T
            s = G[iu0, iu1]
            m = float(s.max())
            E = np.exp(_beta * (s - m))
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                return 1e6, np.zeros_like(flat)
            f = m + np.log(S) / _beta
            W = E / S
            Wm = np.zeros((76, 76))
            Wm[iu0, iu1] = W
            Wm = Wm + Wm.T
            GX = Wm @ X
            GG = np.empty((5, 15))
            for k in range(5):
                Gb = GX[k*15:(k+1)*15]
                # GG[k,q] = sum_t Gb[t, ADD[t,q]]
                GG[k] = Gb[ar[:, None], ADD].sum(axis=0)
            dot = np.sum(GG * Gn, axis=1)
            GP = (GG - dot[:, None] * Gn) / ns[:, None]
            return f, GP.ravel()
        try:
            res = minimize(fun, P0, jac=True, method="L-BFGS-B",
                           options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-7, "maxfun": 3000})
            P = np.asarray(res.x)
        except Exception:
            continue
        try:
            Pm = P.reshape(5, 15)
            ns = np.maximum(np.linalg.norm(Pm, axis=1), 1e-12)
            Gn = Pm / ns[:, None]
            X = expand(Gn)
            v = _maxcos(X)
            if np.isfinite(v):
                # reject fully collapsed
                if v < 0.95:
                    sym_cands.append((v, X.copy()))
                elif not sym_cands:
                    sym_cands.append((v, X.copy()))
        except Exception:
            continue
        if time.time() > deadline - 110.0:
            break

    # tangent Adam release, 1500 steps
    def release(Y, steps=1500, beta0=60.0, beta1=400.0, lr0=0.005, lr1=0.0005):
        Y = Y.copy()
        m1 = np.zeros_like(Y); vv = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        lb = np.log(beta1 / beta0); ll = np.log(lr1 / lr0)
        for s in range(steps):
            if time.time() > deadline - 25.0:
                break
            f = s / max(1, steps - 1)
            beta = beta0 * np.exp(lb * f)
            lr = lr0 * np.exp(ll * f)
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            # upper-only softmax weights via full matrix (double count ok with /2 handling)
            E = np.exp(beta * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S <= 0 or not np.isfinite(S):
                break
            E /= S
            Gr = E @ Y
            Gt = Gr - np.sum(Gr * Y, axis=1, keepdims=True) * Y
            b1t *= b1; b2t *= b2
            m1 *= b1; m1 += (1 - b1) * Gt
            vv *= b2; vv += (1 - b2) * (Gt * Gt)
            mh = m1 / (1 - b1t); vh = vv / (1 - b2t)
            Y -= lr * mh / (np.sqrt(vh) + eps)
            Y = _norm(Y)
        return Y

    bestX = fb.copy(); bestV = float(fbv)
    # sort sym cands
    sym_cands.sort(key=lambda t: t[0])
    # keep top 3 to release
    rel_cands = sym_cands[:3]
    # if no sym cands, create shaken fallback starts (still counts as different basin attempt)
    if not rel_cands:
        rel_cands = [(bestV, fb.copy())]
    for (sv, SX) in rel_cands:
        if time.time() > deadline - 30.0:
            break
        if sv < bestV:
            # symmetric itself could be better (unlikely)
            if sv < bestV:
                bestX = SX.copy(); bestV = sv
        try:
            Y = release(_norm(SX), steps=1500)
            v = _maxcos(Y)
            if v < bestV:
                bestX = Y.copy(); bestV = v
        except Exception:
            continue
        # one shaken variant of the same symmetric point
        if time.time() > deadline - 30.0:
            break
        try:
            rng2 = np.random.default_rng(1000 + int(abs(sv) * 1e6) % 100000)
            Xs = _norm(_norm(SX) + 0.008 * rng2.normal(size=(76, 15)))
            Y2 = release(Xs, steps=1200, beta0=80.0, beta1=450.0, lr0=0.004, lr1=0.0004)
            v2 = _maxcos(Y2)
            if v2 < bestV:
                bestX = Y2.copy(); bestV = v2
        except Exception:
            pass

    # final capped LP polish (~15s)
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        lp_end = time.time() + 15.0
        if lp_end > deadline - 2.0:
            lp_end = deadline - 2.0
        X = _norm(bestX.copy())
        cur = _maxcos(X)
        radius = 8e-4
        iuA, iuB = np.triu_indices(76, 1)
        while time.time() < lp_end - 1.0 and radius > 1e-13:
            if time.time() > deadline - 3.0:
                break
            C = (X @ X.T)[iuA, iuB]
            mm = float(C.max())
            thr = max(20 * radius, 2e-9)
            sel = C > mm - thr
            cnt = int(sel.sum())
            if cnt == 0:
                break
            if cnt > 1200:
                idx = np.argpartition(-C, 1200)[:1200]
                msk = np.zeros_like(sel); msk[idx] = True
                sel = msk
                cnt = 1200
            I = iuA[sel]; J = iuB[sel]
            p = cnt
            nv = 76 * 15 + 1
            rows = np.repeat(np.arange(p), 2 * 15 + 1)
            c1 = X[J].reshape(-1); c2 = X[I].reshape(-1)
            cols1 = (I[:, None] * 15 + ar[None, :]).ravel()
            cols2 = (J[:, None] * 15 + ar[None, :]).ravel()
            cols = np.concatenate([cols1, cols2, np.full(p, 76 * 15)])
            # vals: X[J] for dx_I, X[I] for dx_J, -1 for t ; rhs = -c_ij
            vals = np.concatenate([c1, c2, -np.ones(p)])
            rhs = -C[sel]
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            # equality Xi . dx_i = 0
            er = np.repeat(np.arange(76), 15)
            ec = np.arange(76 * 15)
            ev = X.ravel()
            A_eq = coo_matrix((ev, (er, ec)), shape=(76, nv)).tocsr()
            b_eq = np.zeros(76)
            c_obj = np.zeros(nv); c_obj[-1] = 1.0
            bounds = [(-radius, radius)] * (76 * 15) + [(mm - 0.02, mm)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=rhs, A_eq=A_eq, b_eq=b_eq,
                              bounds=bounds, method="highs",
                              options={"time_limit": max(1.0, lp_end - time.time())})
            except Exception:
                break
            if not getattr(res, "success", False):
                radius *= 0.5
                continue
            dx = np.asarray(res.x[:76*15]).reshape(76, 15)
            nt = float(res.x[-1])
            Xn = _norm(X + dx)
            nv_ = _maxcos(Xn)
            if nv_ < cur - 1e-12:
                X = Xn; cur = nv_
                bestX = X.copy(); bestV = cur
                radius *= 1.3
            else:
                radius *= 0.5
            # early stop if no progress
            if radius < 5e-7 and nv_ >= cur:
                # try once more with tiny radius then exit
                pass
    except Exception:
        pass

    # only return if beats fallback
    try:
        fv = _maxcos(_norm(fb))
    except Exception:
        fv = fbv
    try:
        bv = _maxcos(_norm(bestX))
    except Exception:
        return _norm(fb)
    if bv < fv - 1e-15:
        Xout = _norm(bestX)
    else:
        Xout = _norm(fb)
    # shrink tiny margin guard (renorm exact)
    Xout = _norm(Xout)
    return Xout

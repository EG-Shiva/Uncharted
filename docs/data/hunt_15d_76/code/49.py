import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    import time
    import numpy as np
    t0 = time.time()
    deadline = t0 + 172.0
    # ---------- load fallback ----------
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    def normed(a):
        a = np.asarray(a, dtype=float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return a
    fb = None
    fbv = 1e9
    if xb is not None and xb.shape == (76, 15):
        try:
            xb = normed(xb)
            fb = xb.copy()
            fbv = _max_cos(fb)
        except Exception:
            pass
    if xr is not None and xr.shape == (76, 15):
        try:
            xr = normed(xr)
            v = _max_cos(xr)
            if v < fbv:
                fb = xr.copy()
                fbv = v
        except Exception:
            pass
    if fb is None:
        rng0 = np.random.default_rng(0)
        fb = rng0.normal(size=(76, 15))
        fb = normed(fb)
        fbv = _max_cos(fb)

    n, d = 76, 15
    ONES = np.ones(15, dtype=float) / np.sqrt(15.0)
    iu0, iu1 = np.triu_indices(76, 1)
    ar = np.arange(15)
    # ROLL[t,j]=(j-t)%15 for expansion, ADD[t,j]=(j+t)%15 for pullback
    ROLL = (ar[None, :] - ar[:, None]) % 15
    ADD = (ar[None, :] + ar[:, None]) % 15

    from scipy.optimize import minimize

    def expand_from_Gn(Gn):
        X = np.empty((76, 15), dtype=float)
        for k in range(5):
            X[k*15:(k+1)*15] = Gn[k][ROLL]
        X[75] = ONES
        return X

    # symmetric phase
    sym_best_X = None
    sym_best_v = 1e9
    rng = np.random.default_rng(12345)
    # time budget for symmetric search
    sym_deadline = t0 + 62.0
    n_restart = 5
    betas = [50.0, 150.0, 450.0]
    maxiters = [180, 180, 220]
    for r in range(n_restart):
        if time.time() > sym_deadline - 2.0:
            break
        P = rng.normal(size=(5, 15)).ravel()
        # avoid near-zero rows
        for beta, mi in zip(betas, maxiters):
            if time.time() > sym_deadline - 1.0:
                break
            # closure
            def fun(flat, _beta=beta):
                Pm = flat.reshape(5, 15)
                norms = np.linalg.norm(Pm, axis=1)
                ns = np.maximum(norms, 1e-12)
                Gn = Pm / ns[:, None]
                X = np.empty((76, 15), dtype=float)
                for k in range(5):
                    X[k*15:(k+1)*15] = Gn[k][ROLL]
                X[75] = ONES
                G = X @ X.T
                s = G[iu0, iu1]
                m = float(s.max())
                # stable lse
                E = np.exp(_beta * (s - m))
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    return 1e6, np.zeros_like(flat)
                f = m + np.log(S) / _beta
                W = E / S
                # weight matrix times X without forming 76x76 fully
                # GX[i] = sum_j W_ij * X[j]
                # use scatter: GX = Wmat @ X
                # build Wmat quickly
                Wm = np.zeros((76, 76), dtype=float)
                Wm[iu0, iu1] = W
                Wm = Wm + Wm.T
                GX = Wm @ X
                GG = np.zeros((5, 15), dtype=float)
                for k in range(5):
                    base = k * 15
                    acc = np.zeros(15, dtype=float)
                    for tt in range(15):
                        acc += GX[base + tt][ADD[tt]]
                    GG[k] = acc
                dot = np.sum(GG * Gn, axis=1)
                GP = (GG - dot[:, None] * Gn) / ns[:, None]
                # penalize near-collision mildly to avoid duplicate orbits
                # if max close to 1, add quadratic penalty on max (subgradient via max pair)
                # we add to f and gradient via max pair contribution
                if m > 0.92:
                    pen = 5.0 * (m - 0.92) ** 2
                    f = f + pen
                    # add gradient of pen: 10*(m-0.92)*d(max)/dP
                    # find one max pair
                    # locate first max index
                    try:
                        mi_idx = int(np.argmax(s))
                        ii = int(iu0[mi_idx]); jj = int(iu1[mi_idx])
                        coeff = 10.0 * (m - 0.92)
                        # d s_ij / dP contribution: need dXi/dP and dXj/dP
                        # accumulate into GG-pen then chain; do direct on GP
                        # Build dXi, dXj pullback vectors
                        # helper to map expanded row -> generator grad
                        def row_pullback(row_idx):
                            # returns (k, vec15) such that d X[row]/d Gn[k]
                            if row_idx >= 75:
                                return None
                            kk = row_idx // 15
                            tt = row_idx % 15
                            # X[row,j]=Gn[kk,(j-tt)%15], so dX/dGn is permuted
                            # for gradient of s=Xi.Xj, contrib to GG[kk] is permuted Xj
                            return kk, tt
                        # contribution coeff * (dXi*Xj + dXj*Xi) chain
                        # Xi = X[ii], Xj = X[jj]
                        Xi = X[ii]; Xj = X[jj]
                        # for ii
                        if ii < 75:
                            kk = ii // 15; tt = ii % 15
                            # vector to add to GG[kk]: roll^{-1} of Xj? Actually
                            # ds/dGn[kk, q] = Xj[j'] where j' s.t. (j'-tt)%15==q -> j'=q+tt
                            # so = Xj[(q+tt)%15] = Xj[ADD? no]
                            # Xj indexed by j', q=(j'-tt)%15 => j'=(q+tt)%15
                            # so contrib = Xj[(ar+tt)%15]
                            perm = (ar + tt) % 15
                            GG_pen_ii = Xj[perm]
                            # chain to GP later; directly accumulate to GP via same norm jacobian
                            # we accumulate into GG then convert? simpler accumulate into GP via GG buffer
                            # use temporary
                            pass
                        # To keep code simple and fast, approximate penalty gradient by
                        # scaling existing GG direction (already points away from max).
                        # So just scale GP by (1+coeff) effect implicitly via f increase.
                        # Add explicit small random-free push: rely on LSE gradient which already
                        # strongly repels max. So skip exact penalty gradient.
                        pass
                    except Exception:
                        pass
                return f, GP.ravel()
            try:
                res = minimize(fun, P, jac=True, method="L-BFGS-B",
                               options={"maxiter": mi, "ftol": 1e-14, "gtol": 1e-7, "maxfun": 2000})
                P = np.asarray(res.x, dtype=float)
            except Exception:
                break
        # evaluate true maxcos
        try:
            Pm = P.reshape(5, 15)
            ns = np.maximum(np.linalg.norm(Pm, axis=1), 1e-12)
            Gn = Pm / ns[:, None]
            X = expand_from_Gn(Gn)
            v = _max_cos(X)
            # reject collapsed orbits
            if not np.isfinite(v):
                continue
            # if orbits nearly collide (>0.95) discard unless all are like that
            if v < sym_best_v:
                # allow even collapsed but keep best; prefer non-collapsed
                if v > 0.95:
                    # only accept collapsed if we have nothing
                    if sym_best_X is None:
                        sym_best_X = X.copy(); sym_best_v = v
                else:
                    sym_best_X = X.copy(); sym_best_v = v
        except Exception:
            continue

    # if symmetric search failed, use fallback perturbed as start
    if sym_best_X is None:
        sym_best_X = fb.copy()
        sym_best_v = fbv

    # ---------- release symmetry: short soft-max anneal ----------
    # choose start: best symmetric
    Xcur = sym_best_X.copy()
    Xcur = _norm(Xcur)

    def anneal(X, steps, beta0, beta1, lr0, lr1):
        Y = X.copy()
        m = np.zeros_like(Y); vv = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        lb = np.log(beta1 / beta0); ll = np.log(lr1 / lr0)
        for s_idx in range(steps):
            if time.time() > deadline - 30.0:
                break
            frac = s_idx / (steps - 1) if steps > 1 else 0.0
            beta = beta0 * np.exp(lb * frac)
            lr = lr0 * np.exp(ll * frac)
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            E = np.exp(beta * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S == 0 or not np.isfinite(S):
                break
            E /= S
            Gr = 2.0 * (E @ Y)
            Gt = Gr - (np.sum(Gr * Y, axis=1, keepdims=True)) * Y
            b1t *= b1; b2t *= b2
            m *= b1; m += (1 - b1) * Gt
            vv *= b2; vv += (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1t); vhat = vv / (1 - b2t)
            Y -= lr * mhat / (np.sqrt(vhat) + eps)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    bestX = fb.copy(); bestV = fbv
    # consider symmetric point itself
    if sym_best_v < bestV:
        bestX = sym_best_X.copy(); bestV = sym_best_v
    # anneal from symmetric
    remain = deadline - time.time()
    if remain > 35.0:
        # two short anneals: direct + lightly shaken
        try:
            Y1 = anneal(Xcur, steps=1400, beta0=40.0, beta1=350.0, lr0=0.008, lr1=0.0008)
            v1 = _max_cos(Y1)
            if v1 < bestV:
                bestX = Y1.copy(); bestV = v1
        except Exception:
            pass
        if time.time() < deadline - 32.0:
            try:
                rng2 = np.random.default_rng(777)
                Xs = Xcur + 0.01 * rng2.normal(size=Xcur.shape)
                Xs = _norm(Xs)
                Y2 = anneal(Xs, steps=1200, beta0=50.0, beta1=400.0, lr0=0.006, lr1=0.0006)
                v2 = _max_cos(Y2)
                if v2 < bestV:
                    bestX = Y2.copy(); bestV = v2
            except Exception:
                pass

    # ---------- LP polish ----------
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        X = bestX.copy()
        X = _norm(X)
        cur = _max_cos(X)
        radius = 1e-3
        iu = np.triu_indices(76, 1)
        # use remaining time, leave 2s margin
        while time.time() < deadline - 2.0 and radius > 1e-14:
            if time.time() > deadline - 3.0:
                break
            c = (X @ X.T)[iu]
            mm = float(c.max())
            thr = max(20 * radius, 1e-9)
            sel = c > mm - thr
            if int(sel.sum()) > 1500:
                idx = np.argpartition(-c, 1500)[:1500]
                mask = np.zeros_like(sel); mask[idx] = True
                sel = mask
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = 76 * 15 + 1
            rows = np.repeat(np.arange(p), 2 * 15 + 1)
            cols = np.concatenate([I[:, None] * 15 + np.arange(15), J[:, None] * 15 + np.arange(15), np.full((p, 1), 76 * 15)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(76), 15), np.arange(76 * 15))), shape=(76, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(76),
                              bounds=[(-radius, radius)] * (76 * 15) + [(None, None)], method="highs")
            except Exception:
                break
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(76, 15)
            Y = _norm(Y)
            v = _max_cos(Y)
            if v < cur:
                X, cur = Y, v
                radius = min(radius * 1.5, 0.02)
                bestX, bestV = X.copy(), cur
            else:
                radius *= 0.5
            if radius < 1e-12 and cur <= bestV:
                break
    except Exception:
        pass

    bestX = _norm(bestX)
    if not np.all(np.isfinite(bestX)):
        bestX = fb.copy()
    # final safety: return best among fb and bestX
    try:
        if _max_cos(bestX) > fbv:
            bestX = fb.copy()
    except Exception:
        bestX = fb.copy()
    return bestX

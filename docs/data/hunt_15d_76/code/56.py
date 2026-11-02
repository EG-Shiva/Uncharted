import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_and_pairs(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0, iu1 = np.triu_indices(X.shape[0], 1)
    s = G[iu0, iu1]
    cnt = int(np.sum(s >= m - thr))
    return m, cnt, G, (iu0, iu1, s)

def _tangent_bases(X):
    n, d = X.shape
    Ub = []
    for i in range(n):
        x = X[i]
        # null space of 1x15 row: Vh[1:] is tangent basis
        try:
            _, _, Vh = np.linalg.svd(x.reshape(1, -1), full_matrices=True)
            Ui = Vh[1:].T.copy()  # 15x14
        except Exception:
            # fallback: householder-ish
            q, _ = np.linalg.qr(np.random.randn(d, d))
            # project out x
            q = q - np.outer(x, (q.T @ x))
            q, _ = np.linalg.qr(q)
            Ui = q[:, :d-1]
        # ensure orthogonality to x
        Ui = Ui - np.outer(x, (x @ Ui))
        # re-orthonormalize (QR 15x14)
        try:
            Ui, _ = np.linalg.qr(Ui)
        except Exception:
            pass
        Ub.append(Ui)
    return Ub

def _build_A(X, Ub, pi, pj):
    n, d = X.shape
    D = n * (d - 1)
    m = len(pi)
    A = np.zeros((m, D), dtype=np.float64)
    for r in range(m):
        i = int(pi[r]); j = int(pj[r])
        xi = X[j]  # gradient wrt i uses x_j
        xj = X[i]
        Ui = Ub[i]; Uj = Ub[j]
        A[r, i*(d-1):(i+1)*(d-1)] = Ui.T @ xi
        A[r, j*(d-1):(j+1)*(d-1)] = Uj.T @ xj
    return A

def _anneal(X, steps, T0, T1, lr0, lr1, t0, tlim):
    n, d = X.shape
    for k in range(steps):
        if (k & 255) == 0 and (time.time() - t0) > tlim:
            break
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f if T1 > 0 else T0
        lr = lr0 * (lr1 / lr0) ** f if lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        X = X - lr * Gt
        # project
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X = X / nn
    return X

def _lp_polish(X, t0, tlim, iters=2, eps0=0.02, slack_cap=0.004, max_pairs=650):
    try:
        from scipy.optimize import linprog
    except Exception:
        return X
    n, d = X.shape
    D = n * (d - 1)
    eps = eps0
    for it in range(iters):
        if (time.time() - t0) > tlim:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu0, iu1 = np.triu_indices(n, 1)
        s = G[iu0, iu1]
        order = np.argsort(-s)  # largest first
        # select pairs within slack_cap
        sel = order[s[order] >= m - slack_cap]
        if len(sel) > max_pairs:
            sel = sel[:max_pairs]
        if len(sel) < 10:
            break
        pi = iu0[sel]; pj = iu1[sel]
        g = s[sel]
        Ub = _tangent_bases(X)
        Amat = _build_A(X, Ub, pi, pj)
        mm = len(sel)
        Aub = np.empty((mm, D + 1), dtype=np.float64)
        Aub[:, :D] = Amat
        Aub[:, D] = -1.0
        bub = -g
        c = np.zeros(D + 1, dtype=np.float64)
        c[D] = 1.0
        bounds = [(-eps, eps)] * D + [(m - 0.08, m + 0.08)]
        try:
            res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method='highs',
                          options={'time_limit': 4.0})
        except Exception:
            try:
                res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method='highs')
            except Exception:
                break
        if not hasattr(res, 'success') or not res.success:
            eps *= 0.5
            continue
        y = np.asarray(res.x[:D], dtype=np.float64)
        # form DX
        DX = np.zeros_like(X)
        for i in range(n):
            yb = y[i*(d-1):(i+1)*(d-1)]
            DX[i] = Ub[i] @ yb
        Xn = _norm(X + DX)
        # exact check
        Gn = Xn @ Xn.T
        np.fill_diagonal(Gn, -2.0)
        mn = float(Gn.max())
        if mn < m + 1e-12:
            X = Xn
            eps *= 0.6
            if mn < m - 1e-12:
                pass
        else:
            eps *= 0.5
            # do not accept worsening
        if eps < 1e-4:
            break
        if (time.time() - t0) > tlim:
            break
    return X

def solve():
    t0 = time.time()
    TDEAD = 150.0  # internal deadline, leave margin to 180
    rng = np.random.default_rng(7)
    n, d = 76, 15
    # load incumbent
    def _load(p):
        try:
            a = np.asarray(np.load(p), dtype=np.float64)
            if a.shape == (n, d):
                return _norm(a)
        except Exception:
            return None
        return None
    Xb = _load("best.npy")
    Xr = _load("record.npy")
    cands = []
    if Xb is not None:
        cands.append(Xb)
    if Xr is not None:
        cands.append(Xr)
    if not cands:
        cands.append(_norm(rng.standard_normal(size=(n, d))))
    def _mc(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    vals = [_mc(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best = float(vals[bi])

    # ---- build rigidity Jacobian on incumbent (one SVD only) ----
    try:
        m0, cnt0, G0, (iu0, iu1, s0) = _max_and_pairs(Xbest, thr=1e-7)
        # select active pairs slack<1e-7
        act = np.where(s0 >= m0 - 1e-7)[0]
        if len(act) > 900:
            # keep closest 900
            o = np.argsort(-s0[act])[:900]
            act = act[o]
        if len(act) < 20:
            # widen slightly to have meaningful matrix
            act = np.where(s0 >= m0 - 1e-6)[0]
            if len(act) > 900:
                o = np.argsort(-s0[act])[:900]
                act = act[o]
        pi0 = iu0[act]; pj0 = iu1[act]
        Ub0 = _tangent_bases(Xbest)
        A0 = _build_A(Xbest, Ub0, pi0, pj0)  # ~m x 1064
        D = n * (d - 1)
        # one SVD, full to capture nullspace
        # cap: use full_matrices=True once
        Uu, Ss, Vtf = np.linalg.svd(A0, full_matrices=True)
        smax = float(Ss[0]) if len(Ss) > 0 else 1.0
        if not np.isfinite(smax) or smax <= 0:
            smax = 1.0
        thr_sv = 1e-3 * smax
        msv = len(Ss)
        flop_idx = []
        for k in range(msv):
            if Ss[k] < thr_sv:
                flop_idx.append(k)
        # nullspace rows beyond msv
        for k in range(msv, D):
            flop_idx.append(k)
        if len(flop_idx) < 5:
            # take smallest 50 singular vectors anyway
            flop_idx = list(range(max(0, msv - 50), D))
        Qf = Vtf[flop_idx, :]  # F x D
        F = Qf.shape[0]
    except Exception:
        # cannot do floppy escape, return polished incumbent
        if time.time() - t0 < TDEAD - 8:
            try:
                Xbest = _anneal(Xbest, 1200, 0.002, 0.0003, 0.6, 0.05, t0, TDEAD - 2)
                v = _mc(Xbest)
                best = v
            except Exception:
                pass
        return _norm(np.nan_to_num(Xbest))

    # ---- generate trials: random combos of floppy modes ----
    amps = [0.02, 0.05, 0.10]
    reps_per_amp = 5
    trials = []  # (max, cnt, X)
    try:
        for amp in amps:
            if (time.time() - t0) > TDEAD - 60:
                break
            for rep in range(reps_per_amp):
                if (time.time() - t0) > TDEAD - 60:
                    break
                coeff = rng.standard_normal(size=F)
                y = coeff @ Qf  # D,
                ny = float(np.linalg.norm(y))
                if not np.isfinite(ny) or ny == 0:
                    continue
                y = y / ny
                for sgn in (+1.0, -1.0):
                    yy = sgn * amp * y
                    # map to DX
                    DX = np.zeros((n, d), dtype=np.float64)
                    for i in range(n):
                        yb = yy[i*(d-1):(i+1)*(d-1)]
                        DX[i] = Ub0[i] @ yb
                    Xt = _norm(Xbest + DX)
                    mt, cntt, _, _ = _max_and_pairs(Xt, thr=1e-7)
                    trials.append((float(mt), int(cntt), Xt))
                    if (time.time() - t0) > TDEAD - 60:
                        break
    except Exception:
        pass

    # ---- filter: rise<=0.003 and count<300 ----
    escaped = []
    for (mt, cntt, Xt) in trials:
        if mt <= best + 0.003 and cntt < 300:
            escaped.append((mt, cntt, Xt))
    # sort by max then count
    escaped.sort(key=lambda z: (z[0], z[1]))
    # keep at most 4
    K = min(4, len(escaped))
    escaped = escaped[:K]

    # ---- re-descend filtered trials ----
    for k in range(K):
        if (time.time() - t0) > TDEAD - 25:
            break
        mt, cntt, Xt = escaped[k]
        Xc = Xt.copy()
        try:
            # annealed softmax descent
            # budget steps depending on remaining time
            remain = TDEAD - (time.time() - t0)
            if remain > 60:
                steps = 2200
            elif remain > 35:
                steps = 1500
            else:
                steps = 900
            Xc = _anneal(Xc, steps, 0.020, 0.0008, 5.0, 0.15, t0, TDEAD - 12)
            # short capped LP polish
            if (time.time() - t0) < TDEAD - 14:
                Xc = _lp_polish(Xc, t0, TDEAD - 8, iters=2, eps0=0.015, slack_cap=0.004, max_pairs=650)
            # final tiny low-T squeeze
            if (time.time() - t0) < TDEAD - 8:
                Xc = _anneal(Xc, 600, 0.002, 0.0003, 0.6, 0.05, t0, TDEAD - 2)
        except Exception:
            pass
        try:
            v = _mc(Xc)
        except Exception:
            continue
        if v < best - 1e-15 and np.isfinite(v):
            best = v
            Xbest = Xc.copy()

    # if nothing improved, light polish incumbent if time allows (keep only if improves)
    if time.time() - t0 < TDEAD - 6:
        try:
            Xp = _anneal(Xbest.copy(), 800, 0.0015, 0.00025, 0.5, 0.04, t0, TDEAD - 1)
            vp = _mc(Xp)
            if vp < best and np.isfinite(vp):
                best = vp
                Xbest = Xp
        except Exception:
            pass

    Xbest = _norm(np.nan_to_num(Xbest))
    return Xbest

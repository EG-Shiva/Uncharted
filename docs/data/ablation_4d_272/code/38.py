import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _qconj(q):
    q = np.asarray(q, dtype=np.float64)
    c = q.copy()
    c[..., 1:] *= -1.0
    return c

def _qmul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)

def _hurwitz():
    H = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            H.append(v)
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    H.append([s0, s1, s2, s3])
    return np.array(H, dtype=np.float64)

_H = _hurwitz()
_Hconj = _qconj(_H)

def _build_left(S, U, F):
    K = S.shape[0]
    n = K * 24 + F.shape[0]
    X = np.empty((n, 4), dtype=np.float64)
    Yall = []
    for k in range(K):
        Y = _qmul(_H, S[k])  # (24,4)
        P = _qmul(U[k], Y)
        X[k*24:(k+1)*24] = P
        Yall.append(Y)
    X[K*24:] = F
    return X, Yall

def _build_lr(S, U, V, F):
    K = S.shape[0]
    n = K * 24 + F.shape[0]
    X = np.empty((n, 4), dtype=np.float64)
    Yall = []
    Zall = []
    for k in range(K):
        Y = _qmul(_H, S[k])
        Vc = _qconj(V[k])
        Z = _qmul(Y, Vc)
        P = _qmul(U[k], Z)
        X[k*24:(k+1)*24] = P
        Yall.append(Y); Zall.append(Z)
    X[K*24:] = F
    return X, Yall, Zall

def _sym_anneal(S0, U0, V0, F0, use_lr, steps, a0, a1, s0, s1, mu=0.85, rng=None):
    S = _norm_rows(S0.copy()); U = _norm_rows(U0.copy()); F = _norm_rows(F0.copy())
    if use_lr:
        V = _norm_rows(V0.copy())
    else:
        V = None
    vS = np.zeros_like(S); vU = np.zeros_like(U); vF = np.zeros_like(F)
    if use_lr:
        vV = np.zeros_like(V)
    K = S.shape[0]
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        if use_lr:
            X, Yall, Zall = _build_lr(S, U, V, F)
        else:
            X, Yall = _build_left(S, U, F)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        gradX = W @ X
        # split
        gS = np.zeros_like(S); gU = np.zeros_like(U); gF = gradX[K*24:].copy()
        if use_lr:
            gV = np.zeros_like(V)
        else:
            gV = None
        Uc = _qconj(U)
        for k in range(K):
            gP = gradX[k*24:(k+1)*24]  # (24,4)
            Y = Yall[k]
            if use_lr:
                Z = Zall[k]
                Zc = _qconj(Z)
                Yc = _qconj(Y)
                # grad U
                gU[k] = np.sum(_qmul(gP, Zc), axis=0)
                # grad Z
                gZ = _qmul(Uc[k], gP)
                # grad Vc
                gVc = np.sum(_qmul(Yc, gZ), axis=0)
                # Vc -> V (conj flips)
                gV[k, 0] = gVc[0]; gV[k, 1:] = -gVc[1:]
                # grad Y
                gY = _qmul(gZ, V[k])
                # grad S
                gS[k] = np.sum(_qmul(_Hconj, gY), axis=0)
            else:
                Yc = _qconj(Y)
                gU[k] = np.sum(_qmul(gP, Yc), axis=0)
                gY = _qmul(Uc[k], gP)
                gS[k] = np.sum(_qmul(_Hconj, gY), axis=0)
        # tangent project
        def proj(g, p):
            d = np.sum(g * p, axis=1, keepdims=True)
            return g - d * p
        gS = proj(gS, S); gU = proj(gU, U); gF = proj(gF, F)
        if use_lr:
            gV = proj(gV, V)
        # momentum
        vS = mu * vS + gS; vU = mu * vU + gU; vF = mu * vF + gF
        if use_lr:
            vV = mu * vV + gV
            nrm = max(float(np.linalg.norm(vS, axis=1).max()),
                      float(np.linalg.norm(vU, axis=1).max()),
                      float(np.linalg.norm(vV, axis=1).max()),
                      float(np.linalg.norm(vF, axis=1).max()))
        else:
            nrm = max(float(np.linalg.norm(vS, axis=1).max()),
                      float(np.linalg.norm(vU, axis=1).max()),
                      float(np.linalg.norm(vF, axis=1).max()))
        if not np.isfinite(nrm) or nrm < 1e-14:
            continue
        st = smax / nrm
        S = _norm_rows(S - st * vS)
        U = _norm_rows(U - st * vU)
        F = _norm_rows(F - st * vF)
        if use_lr:
            V = _norm_rows(V - st * vV)
    if use_lr:
        return S, U, V, F
    else:
        return S, U, None, F

def _full_anneal(X, steps, a0, a1, lr0, lr1, beta=0.85):
    X = _norm_rows(X.copy())
    vel = np.zeros_like(X)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        import math
        k = float(math.exp(math.log(a0) + (math.log(a1) - math.log(a0)) * frac))
        lr = float(lr0 + (lr1 - lr0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum())
        if not np.isfinite(Ssum) or Ssum <= 0:
            break
        Grad = (W @ X) * (2.0 / Ssum)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        mx = float(nr.max())
        if mx > 0.5:
            Gt = Gt * (0.5 / mx)
        vel = beta * vel - lr * Gt
        X = X + vel
        X = _norm_rows(X)
    return X

def _lp_polish(x, deadline):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        if time.time() > deadline - 0.05:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    rng = np.random.default_rng(20261031)
    K = 11
    Nfree = 272 - K * 24
    # load incumbents
    X_best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (272, 4):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; X_best = a.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _norm_rows(rng.standard_normal((272, 4)))
        best_val = _max_cos(X_best)
    X0_best = X_best.copy(); v0_best = best_val

    # Phase A: symmetric seed search
    cands = []  # (val, S,U,V,F,use_lr)
    n_restart = 0
    # time limit for phase A
    T_A = t0 + 55.0
    while time.time() < T_A and n_restart < 28:
        use_lr = (n_restart % 2 == 1)
        # init seeds
        S0 = _norm_rows(rng.standard_normal((K, 4)))
        U0 = _norm_rows(rng.standard_normal((K, 4)))
        V0 = _norm_rows(rng.standard_normal((K, 4)))
        F0 = _norm_rows(rng.standard_normal((Nfree, 4)))
        # occasionally seed from record to diversify: pick random record points as seeds
        if n_restart % 7 == 6:
            idx = rng.choice(272, size=K, replace=False)
            S0 = X0_best[idx].copy() + 0.05 * rng.standard_normal((K, 4))
            S0 = _norm_rows(S0)
        # random hyperparams
        a0 = float(rng.choice([8.0, 12.0, 15.0, 20.0]))
        a1 = float(rng.choice([250.0, 400.0, 600.0]))
        s0 = float(rng.choice([0.015, 0.025, 0.04]))
        s1 = float(rng.choice([0.002, 0.004]))
        steps = 450 if n_restart < 20 else 300
        # one orbit rotated by general pair: randomize one U,V strongly (already random)
        try:
            S, U, V, F = _sym_anneal(S0, U0, V0, F0, use_lr, steps, a0, a1, s0, s1, mu=0.85)
            if use_lr:
                X, _, _ = _build_lr(S, U, V, F)
            else:
                X, _ = _build_left(S, U, F)
            v = _max_cos(X)
        except Exception:
            n_restart += 1
            continue
        cands.append((v, S.copy(), U.copy(), (V.copy() if V is not None else None), F.copy(), use_lr))
        n_restart += 1
        # keep sorted, prune to top 8
        cands.sort(key=lambda t: t[0])
        cands = cands[:8]
    # Phase B: symmetry break + full anneal
    # order candidates best first
    cands.sort(key=lambda t: t[0])
    # also consider at most top 5 symmetric
    trial_cands = cands[:5]
    sigmas = [0.004, 0.010, 0.022]
    for (sv, S, U, V, F, use_lr) in trial_cands:
        if time.time() > T_END - 45.0:
            break
        if use_lr:
            Xsym, _, _ = _build_lr(S, U, V, F)
        else:
            Xsym, _ = _build_left(S, U, F)
        for sig in sigmas:
            if time.time() > T_END - 45.0:
                break
            try:
                Y = Xsym + sig * rng.standard_normal(Xsym.shape)
                Y = _norm_rows(Y)
                # full anneal: two stages balanced then sharp
                Y = _full_anneal(Y, steps=700, a0=10.0, a1=220.0, lr0=0.020, lr1=0.002, beta=0.85)
                if time.time() > T_END - 40.0:
                    v = _max_cos(Y)
                    if v < best_val:
                        best_val = v; X_best = Y.copy()
                    continue
                Y = _full_anneal(Y, steps=400, a0=60.0, a1=900.0, lr0=0.010, lr1=0.001, beta=0.88)
                v = _max_cos(Y)
                # short LP
                if v < best_val + 0.004:
                    dl = min(time.time() + 2.5, T_END - 30.0)
                    if dl > time.time() + 0.5:
                        Y2 = _lp_polish(Y, dl)
                        v2 = _max_cos(Y2)
                        if v2 < v:
                            Y, v = Y2, v2
                if v < best_val - 1e-12:
                    best_val = v; X_best = Y.copy()
            except Exception:
                continue
    # Phase C: perturb incumbent record with hole surgery + anneal to ensure not worse
    # quick diversification around X0_best / X_best
    while time.time() < T_END - 28.0:
        try:
            base = X_best.copy()
            # random kick strength
            sig = float(rng.choice([0.06, 0.12, 0.20, 0.30]))
            Y = base + sig * rng.standard_normal(base.shape)
            Y = _norm_rows(Y)
            Y = _full_anneal(Y, steps=600, a0=8.0, a1=300.0, lr0=0.020, lr1=0.0015, beta=0.84)
            v = _max_cos(Y)
            if v < best_val + 0.003:
                dl = min(time.time() + 2.0, T_END - 25.0)
                if dl > time.time() + 0.5:
                    Y = _lp_polish(Y, dl)
                    v = _max_cos(Y)
            if v < best_val - 1e-12:
                best_val = v; X_best = Y.copy()
            # if stuck and time remains, continue
            if time.time() > T_END - 28.0:
                break
        except Exception:
            continue
        # only a few iterations
        # check elapsed: allow ~3-4 kicks
        pass
    # Final thorough polish of best
    try:
        remain = T_END - time.time()
        if remain > 3.0:
            X_best = _lp_polish(X_best, time.time() + remain - 0.5)
            best_val = _max_cos(X_best)
    except Exception:
        pass
    X_best = _norm_rows(X_best)
    # tiny shrink guard (no-op, keep strictly feasible)
    return X_best

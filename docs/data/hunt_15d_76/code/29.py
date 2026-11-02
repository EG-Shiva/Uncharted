import time
import numpy as np

def _norm1(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_single(x, budget, t_dead):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm1(x.astype(float).copy())
    n, d = x.shape
    try:
        best = _maxcos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    t0 = time.time()
    while time.time() < t_dead and (time.time() - t0) < budget and radius > 1e-14:
        if t_dead - time.time() < 0.8:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 1000:
            idx = np.argpartition(-c, 1000)[:1000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu0[0][sel], iu0[1][sel]
        csel = c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(1.0, min(4.0, t_dead - time.time() - 0.2))})
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm1(y)
        try:
            v = _maxcos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_HARD = t0 + 165.0
    rng = np.random.default_rng(12345)
    n, d = 76, 15

    # fallback
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = None
    if Xb is not None and Xb.shape == (76, 15):
        Xb = _norm1(Xb)
    else:
        Xb = None
    if Xr is not None and Xr.shape == (76, 15):
        Xr = _norm1(Xr)
    else:
        Xr = None
    if Xb is None and Xr is None:
        X = _norm1(rng.normal(size=(76, 15)))
        return X
    if Xb is None:
        Xb = Xr.copy()
    if Xr is None:
        Xr = Xb.copy()
    fb = Xb.copy(); fbv = _maxcos(fb)
    v_r = _maxcos(Xr)
    if v_r < fbv:
        fb = Xr.copy(); fbv = v_r
    best_ever = fb.copy(); best_val = fbv

    # precompute orbit matrices
    Mc = np.zeros((d, d))
    Mc[0, d - 1] = 1.0
    for i in range(1, d):
        Mc[i, i - 1] = 1.0
    Ms = Mc.copy(); Ms[0, d - 1] = -1.0
    Pc = np.zeros((15, d, d)); Ps = np.zeros((30, d, d))
    Pc[0] = np.eye(d); Ps[0] = np.eye(d)
    for t in range(1, 15):
        Pc[t] = Mc @ Pc[t - 1]
    for t in range(1, 30):
        Ps[t] = Ms @ Ps[t - 1]
    fixed = np.ones(d) / np.sqrt(d)

    def build_c15(S):
        # S (5,15)
        O = np.einsum('tjp,kp->ktj', Pc, S)  # (5,15,15)
        X = O.reshape(75, d)
        return np.vstack([X, fixed[None, :]])

    def grad_c15(Gx):
        # Gx (76,15) gradient w.r.t X
        Ga = Gx[:75].reshape(5, 15, 15)  # (k,t,j)
        return np.einsum('tjp,ktj->kp', Pc, Ga)

    def build_mix(AB):
        # AB: (3,15): first 2 signed, last 1 cyclic
        A = AB[:2]; B = AB[2:3]
        Os = np.einsum('tjp,ap->atj', Ps, A)  # (2,30,15)
        Oc = np.einsum('tjp,kp->ktj', Pc, B)  # (1,15,15)
        X = np.vstack([Os.reshape(60, d), Oc.reshape(15, d), fixed[None, :]])
        return X

    def grad_mix(Gx):
        Gs = Gx[:60].reshape(2, 30, 15)
        Gc = Gx[60:75].reshape(1, 15, 15)
        gA = np.einsum('tjp,atj->ap', Ps, Gs)
        gB = np.einsum('tjp,ktj->kp', Pc, Gc)
        return np.vstack([gA, gB])

    def build_anti(Y):
        return np.vstack([Y, -Y])

    def grad_anti(Gx):
        return Gx[:38] - Gx[38:]

    def anneal_sym(Sinit, build_fn, grad_fn, n_iter, beta0, beta1, lr0, lr1, noise0, mu):
        S = _norm1(Sinit.astype(float).copy())
        V = np.zeros_like(S)
        for k in range(n_iter):
            if time.time() > T_HARD - 55:
                break
            f = k / max(1, n_iter - 1)
            beta = beta0 * (beta1 / beta0) ** f
            lr = lr0 * (lr1 / lr0) ** f
            ns = noise0 * (1.0 - f)
            X = build_fn(S)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            E = np.exp(beta * (G - m))
            np.fill_diagonal(E, 0.0)
            Ssum = E.sum()
            if not np.isfinite(Ssum) or Ssum <= 0:
                continue
            W = E / Ssum
            Gr = 2.0 * (W @ X)
            Gs = grad_fn(Gr)
            dot = np.sum(Gs * S, axis=1, keepdims=True)
            Gt = Gs - dot * S
            # clip
            gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
            sc = np.minimum(1.0, 2.0 / gn)
            Gt = Gt * sc
            V = mu * V + Gt
            S = S - lr * V
            if ns > 0:
                Nz = rng.normal(size=S.shape) * ns
                Nz = Nz - np.sum(Nz * S, axis=1, keepdims=True) * S
                S = S + Nz
            S = _norm1(S)
        X = build_fn(S)
        return S, X

    def anneal_free(Xinit, n_iter, beta0, beta1, lr0, lr1, noise0, mu):
        X = _norm1(Xinit.astype(float).copy())
        V = np.zeros_like(X)
        for k in range(n_iter):
            if time.time() > T_HARD - 40:
                break
            f = k / max(1, n_iter - 1)
            beta = beta0 * (beta1 / beta0) ** f
            lr = lr0 * (lr1 / lr0) ** f
            ns = noise0 * (1.0 - f)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            E = np.exp(beta * (G - m))
            np.fill_diagonal(E, 0.0)
            Ssum = E.sum()
            if not np.isfinite(Ssum) or Ssum <= 0:
                continue
            W = E / Ssum
            Gr = 2.0 * (W @ X)
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
            sc = np.minimum(1.0, 2.0 / gn)
            Gt = Gt * sc
            V = mu * V + Gt
            X = X - lr * V
            if ns > 0:
                Nz = rng.normal(size=X.shape) * ns
                Nz = Nz - np.sum(Nz * X, axis=1, keepdims=True) * X
                X = X + Nz
            X = _norm1(X)
        return X

    cands = []

    # --- symmetry search (time-boxed) ---
    # C15: 5 seeds
    for rep in range(6):
        if time.time() > T_HARD - 90:
            break
        Sinit = rng.normal(size=(5, d))
        Sinit = _norm1(Sinit)
        # vary schedules slightly per rep
        b0 = 25.0; b1 = 450.0 + 50 * (rep % 3)
        _, X = anneal_sym(Sinit, build_c15, grad_c15, n_iter=1800,
                          beta0=b0, beta1=b1, lr0=2.0, lr1=0.10, noise0=0.003, mu=0.85)
        v = _maxcos(X)
        cands.append((v, X.copy()))
        if v < best_val:
            best_val = v; best_ever = X.copy()

    # mixed signed: 3 seeds (2 signed +1 cyclic)
    for rep in range(6):
        if time.time() > T_HARD - 80:
            break
        ABinit = _norm1(rng.normal(size=(3, d)))
        _, X = anneal_sym(ABinit, build_mix, grad_mix, n_iter=1800,
                          beta0=25.0, beta1=500.0, lr0=2.0, lr1=0.10, noise0=0.003, mu=0.85)
        v = _maxcos(X)
        cands.append((v, X.copy()))
        if v < best_val:
            best_val = v; best_ever = X.copy()

    # antipodal: 38 seeds
    for rep in range(5):
        if time.time() > T_HARD - 70:
            break
        Yinit = _norm1(rng.normal(size=(38, d)))
        _, X = anneal_sym(Yinit, build_anti, grad_anti, n_iter=1500,
                          beta0=20.0, beta1=400.0, lr0=1.5, lr1=0.08, noise0=0.002, mu=0.85)
        v = _maxcos(X)
        cands.append((v, X.copy()))
        if v < best_val:
            best_val = v; best_ever = X.copy()

    # unconstrained release of best symmetric candidates
    cands.sort(key=lambda t: t[0])
    nrel = min(3, len(cands))
    released = []
    for i in range(nrel):
        if time.time() > T_HARD - 45:
            break
        _, X0 = cands[i]
        # two-stage free anneal: coarse then fine
        X1 = anneal_free(X0, n_iter=2200, beta0=30.0, beta1=600.0, lr0=2.0, lr1=0.08, noise0=0.0015, mu=0.85)
        v1 = _maxcos(X1)
        released.append((v1, X1))
        if v1 < best_val:
            best_val = v1; best_ever = X1.copy()
    # also release fallback record with short anneal to try different basin via perturbation
    if time.time() < T_HARD - 50:
        Xp = _norm1(fb + 0.05 * rng.normal(size=(76, 15)))
        Xp = anneal_free(Xp, n_iter=2000, beta0=30.0, beta1=600.0, lr0=2.0, lr1=0.08, noise0=0.001, mu=0.85)
        vp = _maxcos(Xp)
        released.append((vp, Xp))
        if vp < best_val:
            best_val = vp; best_ever = Xp.copy()

    # final LP polish on top 2
    allc = cands + released
    allc.sort(key=lambda t: t[0])
    # always include fallback
    allc.append((fbv, fb))
    allc.sort(key=lambda t: t[0])
    seen = []
    for v, X in allc[:3]:
        if time.time() > T_HARD - 5:
            break
        if v > best_val + 0.015 and len(seen) > 0:
            continue
        Xp = _lp_polish_single(X, budget=9.0, t_dead=T_HARD)
        vp = _maxcos(Xp)
        seen.append((vp, Xp))
        if vp < best_val:
            best_val = vp; best_ever = Xp.copy()

    if fbv < best_val:
        best_ever = fb.copy(); best_val = fbv
    best_ever = _norm1(best_ever)
    return best_ever

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

def _lp_polish(x, deadline):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
            cnt = 6000
        if cnt == 0:
            radius *= 0.5
            continue
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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

def _repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    x = _norm_rows(x.copy())
    vel = np.zeros_like(x)
    best_X = x.copy()
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            best_X = x.copy()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / (S * 0.5 + 1e-300) * 0.5
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm_rows(x)
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    fv = float(Gf.max())
    if fv < bv:
        return x, fv
    return best_X, bv

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SYM_END = t0 + 42.0
    ANNEAL_END = t0 + 140.0
    rng = np.random.default_rng(20261031)
    n, d = 272, 4
    K = 11
    Nsym = K * 24
    Nfree = n - Nsym

    # Hurwitz units
    H_arr = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            H_arr.append(v)
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    H_arr.append([s0, s1, s2, s3])
    H_arr = np.array(H_arr, dtype=np.float64)  # 24,4

    def left_mat(q):
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([[w, -x, -y, -z],
                         [x, w, -z, y],
                         [y, z, w, -x],
                         [z, -y, x, w]], dtype=np.float64)
    def right_mat(q):
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([[w, -x, -y, -z],
                         [x, w, z, -y],
                         [y, -z, w, x],
                         [z, y, -x, w]], dtype=np.float64)
    Hmats = np.stack([left_mat(q) for q in H_arr], axis=0)  # 24,4,4
    # derivative bases
    DL = np.zeros((4, 4, 4))
    DR = np.zeros((4, 4, 4))
    DL[0] = np.eye(4)
    DL[1] = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=float)
    DL[2] = np.array([[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]], dtype=float)
    DL[3] = np.array([[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
    DR[0] = np.eye(4)
    DR[1] = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)
    DR[2] = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    DR[3] = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=float)

    def build_X(S, U, V, F):
        # S,U,V: (K,4) unit, F: (8,4) unit
        Lk = np.einsum('kj,jab->kab', U, DL)
        Rk = np.einsum('kj,jab->kab', V, DR)
        Mk = np.matmul(Rk, Lk)  # K,4,4
        Yk = np.einsum('iab,kb->kia', Hmats, S)  # K,24,4
        Pk = np.einsum('kab,kib->kia', Mk, Yk)
        X = np.empty((n, d), dtype=np.float64)
        for k in range(K):
            X[k * 24:(k + 1) * 24] = Pk[k]
        X[Nsym:] = F
        X = _norm_rows(X)
        return X

    def sym_opt(S0, U0, V0, F0, steps, a0, a1, s0, s1, mu=0.85):
        S = S0.copy(); U = U0.copy(); V = V0.copy(); F = F0.copy()
        vS = np.zeros_like(S); vU = np.zeros_like(U); vV = np.zeros_like(V); vF = np.zeros_like(F)
        best_val = 2.0
        best_X = None
        for it in range(steps):
            if time.time() > SYM_END:
                break
            frac = it / max(steps - 1, 1)
            alpha = a0 * (a1 / a0) ** frac
            smax = s0 * (s1 / s0) ** frac
            Lk = np.einsum('kj,jab->kab', U, DL)
            Rk = np.einsum('kj,jab->kab', V, DR)
            Mk = np.matmul(Rk, Lk)
            Yk = np.einsum('iab,kb->kia', Hmats, S)
            Pk = np.einsum('kab,kib->kia', Mk, Yk)
            X = np.empty((n, d), dtype=np.float64)
            for k in range(K):
                X[k * 24:(k + 1) * 24] = Pk[k]
            X[Nsym:] = F
            # normalize rows (should already be unit, but enforce)
            X = _norm_rows(X)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            if m < best_val:
                best_val = m
                best_X = X.copy()
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            Ssum = float(W.sum())
            if not np.isfinite(Ssum) or Ssum <= 0:
                continue
            gradX = (W @ X) / Ssum
            # split
            gS = np.zeros_like(S); gU = np.zeros_like(U); gV = np.zeros_like(V)
            gF = gradX[Nsym:].copy()
            for k in range(K):
                Gp = gradX[k * 24:(k + 1) * 24]  # 24,4
                Y = Yk[k]; M = Mk[k]; A = Rk[k]; B = Lk[k]
                gradM = Gp.T @ Y  # 4x4
                gradA = gradM @ B.T
                gradB = A.T @ gradM
                gu = np.array([(gradB * DL[j]).sum() for j in range(4)])
                gv = np.array([(gradA * DR[j]).sum() for j in range(4)])
                Gy = Gp @ M  # 24,4
                gs = np.einsum('iab,ia->b', Hmats, Gy)
                gS[k] = gs; gU[k] = gu; gV[k] = gv
            # project to tangent
            def proj(g, q):
                return g - np.sum(g * q, axis=1, keepdims=True) * q
            tS = proj(gS, S); tU = proj(gU, U); tV = proj(gV, V); tF = proj(gF, F)
            vS = mu * vS + tS; vU = mu * vU + tU; vV = mu * vV + tV; vF = mu * vF + tF
            nrm = 0.0
            for vv in (vS, vU, vV, vF):
                nn = float(np.linalg.norm(vv, axis=1).max()) if vv.size else 0.0
                if nn > nrm:
                    nrm = nn
            if nrm < 1e-14:
                continue
            S = S - (smax / nrm) * vS
            U = U - (smax / nrm) * vU
            V = V - (smax / nrm) * vV
            F = F - (smax / nrm) * vF
            S = _norm_rows(S); U = _norm_rows(U); V = _norm_rows(V); F = _norm_rows(F)
        return best_X, best_val, (S, U, V, F)

    # load fallback
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        a = rng.standard_normal((n, d))
        cand.append(_norm_rows(a))
    vals = [_max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best = cand[bi].copy()
    best_val = float(vals[bi])

    # symmetric restarts
    sym_best_X = None
    sym_best_val = 2.0
    # orthoplex for free init reference
    ortho = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0],
                      [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]], dtype=float)
    n_restart = 6
    for r in range(n_restart):
        if time.time() > SYM_END - 2:
            break
        S0 = _norm_rows(rng.standard_normal((K, 4)))
        U0 = _norm_rows(rng.standard_normal((K, 4)))
        V0 = _norm_rows(rng.standard_normal((K, 4)))
        if r == 0:
            F0 = ortho.copy()
            # random rotation of orthoplex
            Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            F0 = F0 @ Q.T
            F0 = _norm_rows(F0 + 0.05 * rng.standard_normal((Nfree, 4)))
        else:
            F0 = _norm_rows(rng.standard_normal((Nfree, 4)))
        steps = 450 if r < 4 else 300
        bx, bv, _ = sym_opt(S0, U0, V0, F0, steps, 35.0, 350.0, 0.05, 0.004, mu=0.85)
        if bx is not None and bv < sym_best_val:
            sym_best_val = bv
            sym_best_X = bx.copy()
    # if symmetric failed, use best
    if sym_best_X is None:
        sym_best_X = best.copy()
        sym_best_val = best_val

    if sym_best_val < best_val:
        # keep global best as is; symmetric will be broken below
        pass

    # build list of broken-symmetry starts
    starts = []
    # symmetric broken with several noise levels
    for sig in (0.004, 0.012, 0.03, 0.06):
        if sym_best_X is not None:
            Xt = sym_best_X + rng.standard_normal((n, d)) * sig
            starts.append(_norm_rows(Xt))
    # one-orbit rotated mixtures
    for _ in range(2):
        if sym_best_X is not None and time.time() < ANNEAL_END:
            Xt = sym_best_X.copy()
            k0 = int(rng.integers(0, K))
            u = _norm_rows(rng.standard_normal((1, 4)))[0]
            v = _norm_rows(rng.standard_normal((1, 4)))[0]
            Mr = right_mat(v) @ left_mat(u)
            sl = slice(k0 * 24, (k0 + 1) * 24)
            Xt[sl] = Xt[sl] @ Mr.T
            Xt = _norm_rows(Xt + rng.standard_normal((n, d)) * 0.008)
            starts.append(Xt)
    # also include current best with tiny kick to not lose record
    starts.append(best.copy())
    Xt = best + rng.standard_normal((n, d)) * 0.008
    starts.append(_norm_rows(Xt))

    cur_best = best.copy()
    cur_val = best_val
    # consider symmetric directly annealed too
    for idx, st in enumerate(starts):
        if time.time() > ANNEAL_END:
            break
        # vary schedule by index
        if idx % 3 == 0:
            Xo, vo = _repulse_gd(st, 550, 30.0, 420.0, 0.06, 0.004, mu=0.85)
        elif idx % 3 == 1:
            mid, _ = _repulse_gd(st, 250, 18.0, 60.0, 0.07, 0.01, mu=0.8)
            Xo, vo = _repulse_gd(mid, 350, 70.0, 500.0, 0.03, 0.003, mu=0.85)
        else:
            Xo, vo = _repulse_gd(st, 500, 50.0, 380.0, 0.05, 0.005, mu=0.9)
        if vo < cur_val:
            cur_val = vo
            cur_best = Xo.copy()
        # also try short second quench from Xo with small noise
        if time.time() < ANNEAL_END - 10 and vo < best_val + 0.002:
            Xt2 = Xo + rng.standard_normal((n, d)) * 0.006
            Xt2 = _norm_rows(Xt2)
            Xo2, vo2 = _repulse_gd(Xt2, 250, 80.0, 500.0, 0.02, 0.003, mu=0.85)
            if vo2 < cur_val:
                cur_val = vo2
                cur_best = Xo2.copy()

    best = cur_best
    best_val = cur_val
    # final LP polish with remaining time
    polish_deadline = HARD_END - 1.0
    # give at least 20s to polish if possible
    if polish_deadline - time.time() > 3:
        best = _lp_polish(best, polish_deadline)
        best_val = _max_cos(best)
    best = _norm_rows(best)
    return best

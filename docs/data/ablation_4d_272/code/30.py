import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, float(rm.max())

def left_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([[w, -x, -y, -z],
                     [x,  w, -z,  y],
                     [y,  z,  w, -x],
                     [z, -y,  x,  w]], dtype=np.float64)

def right_mat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([[w, -x, -y, -z],
                     [x,  w,  z, -y],
                     [y, -z,  w,  x],
                     [z,  y, -x,  w]], dtype=np.float64)

def qmul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)

def hurwitz_units():
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

def repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm(x)
    return x

def lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = max_cos(x)
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
        y = _norm(y)
        v = max_cos(y)
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
    rng = np.random.default_rng(987654321)
    n, d = 272, 4
    K = 11
    Nsym = K * 24
    Nfree = n - Nsym  # 8
    H_arr = hurwitz_units()  # (24,4)
    Hmats = np.stack([left_mat(q) for q in H_arr], axis=0)  # (24,4,4)

    def build_X(S, U, F):
        X = np.empty((n, d), dtype=np.float64)
        for k in range(K):
            Y = qmul(H_arr, S[k])  # (24,4) = h*s
            P = qmul(U[k], Y)      # U*(h*s)
            X[k * 24:(k + 1) * 24] = P
        X[Nsym:] = F
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        return X / nn

    def sym_search(S0, U0, F0, steps, a0, a1, st0, st1, mu=0.85):
        S = S0.copy(); U = U0.copy(); F = F0.copy()
        vS = np.zeros_like(S); vU = np.zeros_like(U); vF = np.zeros_like(F)
        for it in range(steps):
            if time.time() > T_END - 125.0 and steps > 200:
                # leave time for full phase
                pass
            frac = it / max(steps - 1, 1)
            alpha = a0 * (a1 / a0) ** frac
            st = st0 * (st1 / st0) ** frac
            X = build_X(S, U, F)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            Ssum = float(W.sum() * 0.5)
            if not np.isfinite(Ssum) or Ssum <= 0:
                Ssum = 1.0
            grad = (W @ X) / Ssum  # (272,4)
            gS = np.zeros_like(S); gU = np.zeros_like(U)
            for k in range(K):
                gb = grad[k * 24:(k + 1) * 24]  # (24,4)
                MU = left_mat(U[k])  # (4,4)
                # M(Uh) = MU @ Mh
                MUH = np.einsum('ab,hbc->hac', MU, Hmats)  # (24,4,4)
                ds = np.einsum('hai,ha->i', MUH, gb)
                # Y for right mats
                Y = qmul(H_arr, S[k])  # (24,4)
                Rstk = np.stack([right_mat(q) for q in Y], axis=0)
                du = np.einsum('hai,ha->i', Rstk, gb)
                gS[k] = ds
                gU[k] = du
            gF = grad[Nsym:]
            # tangent projection
            gS = gS - np.sum(gS * S, axis=1, keepdims=True) * S
            gU = gU - np.sum(gU * U, axis=1, keepdims=True) * U
            gF = gF - np.sum(gF * F, axis=1, keepdims=True) * F
            vS = mu * vS + gS
            vU = mu * vU + gU
            vF = mu * vF + gF
            nrm = max(float(np.linalg.norm(vS, axis=1).max()),
                      float(np.linalg.norm(vU, axis=1).max()),
                      float(np.linalg.norm(vF, axis=1).max()))
            if nrm > 1e-14:
                S = S - (st / nrm) * vS
                U = U - (st / nrm) * vU
                F = F - (st / nrm) * vF
                S = _norm(S); U = _norm(U); F = _norm(F)
            if time.time() > T_END - 20.0:
                break
        Xf = build_X(S, U, F)
        return Xf, S, U, F

    # load incumbents
    cands = []
    try:
        bx = np.load("best.npy")
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            cands.append(_norm(bx))
    except Exception:
        pass
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            cands.append(_norm(rx))
    except Exception:
        pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [(max_cos(c), c) for c in cands]
    vals.sort(key=lambda t: t[0])
    best_v, best_x = vals[0][0], vals[0][1].copy()

    # ---- symmetric phase ----
    sym_best_v = 2.0
    sym_best_X = None
    n_rest = 5
    for r in range(n_rest):
        if time.time() > T_END - 130.0:
            break
        S0 = _norm(rng.standard_normal((K, d)))
        U0 = _norm(rng.standard_normal((K, d)))
        F0 = _norm(rng.standard_normal((Nfree, d)))
        # two-stage: coarse then fine
        Xc, S1, U1, F1 = sym_search(S0, U0, F0, steps=500, a0=12.0, a1=90.0, st0=0.03, st1=0.006, mu=0.82)
        Xf, S2, U2, F2 = sym_search(S1, U1, F1, steps=500, a0=90.0, a1=260.0, st0=0.010, st1=0.002, mu=0.88)
        v = max_cos(Xf)
        if v < sym_best_v:
            sym_best_v = v
            sym_best_X = Xf.copy()
        # also try single rotated-orbit mixture: rotate orbit 0 by extra random pair
        if r == 0:
            try:
                # left-right rotation of first orbit: apply extra random rotation
                Q = _norm(rng.standard_normal((1, d)))[0]
                QM = left_mat(Q)
                Xt = Xf.copy()
                blk = Xt[0:24] @ QM.T  # rigid rotation of block (orthogonal)
                # renormalize (orthogonal preserves norm)
                Xt[0:24] = _norm(blk)
                vt = max_cos(Xt)
                if vt < sym_best_v:
                    sym_best_v = vt
                    sym_best_X = Xt.copy()
            except Exception:
                pass

    pool = [(best_v, best_x.copy())]
    if sym_best_X is not None:
        pool.append((sym_best_v, sym_best_X.copy()))
        # symmetry-broken descendants: small noise + short anneal
        for sig in (0.004, 0.012, 0.03):
            if time.time() > T_END - 115.0:
                break
            Xp = _norm(sym_best_X + rng.standard_normal(sym_best_X.shape) * sig)
            Xp = repulse_gd(Xp, 500, 25.0, 180.0, 0.008, 0.0009, mu=0.85)
            v = max_cos(Xp)
            pool.append((v, Xp.copy()))
    pool.sort(key=lambda t: t[0])
    pool = pool[:6]
    if pool[0][0] < best_v:
        best_v, best_x = pool[0][0], pool[0][1].copy()

    # quick lock of current best if time allows
    if time.time() < T_END - 110.0:
        bx2 = lp_polish(best_x, min(T_END - 105.0, time.time() + 4.0))
        bv2 = max_cos(bx2)
        pool.append((bv2, bx2.copy()))
        pool.sort(key=lambda t: t[0])
        pool = pool[:6]
        if bv2 < best_v:
            best_v, best_x = bv2, bx2.copy()

    iu_all = np.triu_indices(n, 1)
    LOOP_END = T_END - 18.0
    it = 0
    while time.time() < LOOP_END:
        it += 1
        remain = LOOP_END - time.time()
        if remain < 2.0:
            break
        # pick base: 60% best, else random pool
        if len(pool) == 1 or rng.random() < 0.6:
            base = pool[0][1]
        else:
            base = pool[rng.integers(len(pool))][1]
        u = rng.random()
        if u < 0.60:
            sigma = float(10.0 ** rng.uniform(-2.7, -1.0))
            xp = base + rng.standard_normal(base.shape) * sigma
            if rng.random() < 0.5:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                cg = G[iu_all]
                k = 20
                top = np.argpartition(-cg, k - 1)[:k]
                ch = top[rng.integers(k)]
                i = int(iu_all[0][ch]); j = int(iu_all[1][ch])
                push = 0.5 * sigma + 0.004
                xp[i] += push * (base[i] - base[j])
                xp[j] += push * (base[j] - base[i])
            xp = _norm(xp)
            a0 = 18.0 + 14.0 * rng.random()
            a1 = 120.0 + 120.0 * rng.random()
            s0 = 0.006 + 0.008 * rng.random()
            s1 = 0.0005 + 0.0012 * rng.random()
            steps = 700 + int(rng.integers(0, 500))
            if remain < 25:
                steps = min(steps, 450)
            mu = 0.80 + 0.12 * rng.random()
            xp = repulse_gd(xp, steps, a0, a1, s0, s1, mu=mu)
        elif u < 0.85:
            # break symmetry further: rotate a random 24-block rigidly
            xp = base.copy()
            b = int(rng.integers(K))
            Q = _norm(rng.standard_normal((1, d)))[0]
            # random small rotation: slerp between identity and Q
            t = float(rng.uniform(0.05, 0.5))
            e0 = np.array([1.0, 0.0, 0.0, 0.0])
            Qm = (1 - t) * e0 + t * Q
            Qm /= np.linalg.norm(Qm)
            QM = left_mat(Qm)
            blk = xp[b * 24:(b + 1) * 24] @ QM.T
            xp[b * 24:(b + 1) * 24] = blk
            xp = _norm(xp + rng.standard_normal(xp.shape) * 0.004)
            xp = repulse_gd(xp, 550, 25.0, 200.0, 0.007, 0.0008, mu=0.85)
        else:
            xp = _norm(rng.standard_normal(base.shape))
            xp = repulse_gd(xp, 400, 7.0, 28.0, 0.02, 0.005, mu=0.8)
            xp = repulse_gd(xp, 500, 28.0, 200.0, 0.008, 0.0008, mu=0.85)
        v = max_cos(xp)
        if v < best_v - 1e-12:
            best_v, best_x = v, xp.copy()
            pool.append((v, xp.copy()))
            pool.sort(key=lambda t: t[0])
            del pool[6:]
            if time.time() < LOOP_END - 3.0:
                yp = lp_polish(xp, min(LOOP_END, time.time() + 2.0))
                vy = max_cos(yp)
                if vy < best_v:
                    best_v, best_x = vy, yp.copy()
                pool.append((vy, yp.copy()))
                pool.sort(key=lambda t: t[0])
                del pool[6:]
        elif v < best_v + 0.0035:
            if len(pool) < 6 or v < pool[-1][0]:
                pool.append((v, xp.copy()))
                pool.sort(key=lambda t: t[0])
                del pool[6:]
            if v < best_v + 0.0006 and rng.random() < 0.25 and time.time() < LOOP_END - 3.0:
                yp = lp_polish(xp, min(LOOP_END, time.time() + 1.0))
                vy = max_cos(yp)
                if vy < best_v:
                    best_v, best_x = vy, yp.copy()
                    pool.append((vy, yp.copy()))
                    pool.sort(key=lambda t: t[0])
                    del pool[6:]

    # final polish
    try:
        best_x = lp_polish(best_x, T_END - 1.0)
        best_v = max_cos(best_x)
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x

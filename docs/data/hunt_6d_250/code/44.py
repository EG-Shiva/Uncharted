import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _gd_opt(X0, beta, steps, lr, mu=0.86):
    X = X0.copy()
    V = np.zeros_like(X)
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = beta * (G - m)
        np.clip(D, -30.0, 0.0, out=D)
        W = np.exp(D)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ X) / rs
        dot = np.sum(mean * X, axis=1, keepdims=True)
        tang = mean - dot * X
        V = mu * V + tang
        cur_lr = lr * (1.0 - 0.4 * it / max(1, steps))
        X = X - cur_lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = X / nr
    return X

def _lp_polish(x, deadline, k_top=4500, radius_init=2.0e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0, iu1 = np.triu_indices(n, 1)
    ntri = iu0.shape[0]
    K = min(k_top, ntri)
    while time.time() < deadline - 0.8 and radius > 1e-13:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        if K < ntri:
            part = np.argpartition(-c_all, K)[:K]
            I = iu0[part]; J = iu1[part]; c_sel = c_all[part]
        else:
            I = iu0; J = iu1; c_sel = c_all
        p = I.shape[0]
        nv = n * d + 1
        cols_blocks = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols_blocks)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        val = _max_cos(y)
        if val < best - 1e-12:
            x, best = y, val
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(0)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                cands.append(z)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        cands.append(z)
    elite = min(cands, key=_max_cos).copy()
    elite_val = _max_cos(elite)
    pool = [elite.copy()]

    # ---- symmetric orbit search: R6 = C^3, cyclic rotations ----
    sym_configs = [
        (25, 10, (1, 2, 7)),
        (50, 5, (1, 3, 7)),
        (10, 25, (1, 3, 7)),
        (5, 50, (1, 2, 3)),
    ]
    for (m, t, avec) in sym_configs:
        if remaining() < 110:
            break
        avec = np.asarray(avec, dtype=float)
        ks = np.arange(m)[:, None]  # (m,1)
        th = 2.0 * np.pi * ks * avec[None, :] / float(m)  # (m,3)
        ctab = np.cos(th); stab = np.sin(th)
        def build_X(Zb):
            X = np.empty((m * t, d), dtype=np.float64)
            for k in range(m):
                c0, c1, c2 = ctab[k, 0], ctab[k, 1], ctab[k, 2]
                s0, s1, s2 = stab[k, 0], stab[k, 1], stab[k, 2]
                z0, z1, z2, z3, z4, z5 = Zb[:, 0], Zb[:, 1], Zb[:, 2], Zb[:, 3], Zb[:, 4], Zb[:, 5]
                blk = np.empty_like(Zb)
                blk[:, 0] = c0 * z0 - s0 * z1
                blk[:, 1] = s0 * z0 + c0 * z1
                blk[:, 2] = c1 * z2 - s1 * z3
                blk[:, 3] = s1 * z2 + c1 * z3
                blk[:, 4] = c2 * z4 - s2 * z5
                blk[:, 5] = s2 * z4 + c2 * z5
                X[k * t:(k + 1) * t, :] = blk
            return X
        # closure for given beta
        Ycur = rng.standard_normal((t, d))
        Ycur /= np.maximum(np.linalg.norm(Ycur, axis=1, keepdims=True), 1e-12)
        for beta, mit in [(2000.0, 120), (8000.0, 120)]:
            if remaining() < 90:
                break
            def fg(Yflat, _beta=beta):
                Y = Yflat.reshape(t, d)
                nrm = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Zb = Y / nrm
                X = build_X(Zb)
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                M = float(np.max(G))
                D = _beta * (G - M)
                np.clip(D, -30.0, 0.0, out=D)
                E = np.exp(D)
                np.fill_diagonal(E, 0.0)
                S = float(np.sum(E))
                if S <= 0 or not np.isfinite(S):
                    S = 1e-300
                loss = M + np.log(S) / _beta
                W = E / S
                GX = 2.0 * (W @ X)
                GZ = np.zeros_like(Zb)
                for k in range(m):
                    Gk = GX[k * t:(k + 1) * t, :]
                    c0, c1, c2 = ctab[k, 0], ctab[k, 1], ctab[k, 2]
                    s0, s1, s2 = stab[k, 0], stab[k, 1], stab[k, 2]
                    # R^T * Gk
                    GZ[:, 0] += c0 * Gk[:, 0] + s0 * Gk[:, 1]
                    GZ[:, 1] += -s0 * Gk[:, 0] + c0 * Gk[:, 1]
                    GZ[:, 2] += c1 * Gk[:, 2] + s1 * Gk[:, 3]
                    GZ[:, 3] += -s1 * Gk[:, 2] + c1 * Gk[:, 3]
                    GZ[:, 4] += c2 * Gk[:, 4] + s2 * Gk[:, 5]
                    GZ[:, 5] += -s2 * Gk[:, 4] + c2 * Gk[:, 5]
                dot = np.sum(GZ * Zb, axis=1, keepdims=True)
                GY = (GZ - dot * Zb) / nrm
                return loss, GY.ravel()
            try:
                res = minimize(fg, Ycur.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": mit, "ftol": 1e-20, "gtol": 1e-12,
                                        "maxfun": mit * 5, "maxcor": 8})
                Ycur = res.x.reshape(t, d)
            except Exception:
                break
        try:
            Yn = Ycur / np.maximum(np.linalg.norm(Ycur, axis=1, keepdims=True), 1e-12)
            Xsym = build_X(Yn)
            vsym = _max_cos(Xsym)
            # break symmetry and relax
            Xb = Xsym + rng.standard_normal(Xsym.shape) * 0.003
            Xb /= np.maximum(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-12)
            Xb = _gd_opt(Xb, 2500.0, 300, 0.006)
            Xb = _gd_opt(Xb, 8000.0, 250, 0.003)
            vb = _max_cos(Xb)
            pool.append(Xb)
            if vb < elite_val:
                elite = Xb.copy(); elite_val = vb
            elif vsym < elite_val:
                # keep symmetric itself if better (rare)
                if vsym < elite_val:
                    elite = Xsym.copy(); elite_val = vsym
        except Exception:
            continue

    # ---- random multi-start diverse basins ----
    nstart = 10
    for si in range(nstart):
        if remaining() < 62:
            break
        try:
            X0 = rng.standard_normal((n, d))
            X0 /= np.maximum(np.linalg.norm(X0, axis=1, keepdims=True), 1e-12)
            X1 = _gd_opt(X0, 900.0, 350, 0.012)
            X2 = _gd_opt(X1, 3200.0, 350, 0.005)
            v2 = _max_cos(X2)
            # escalate promising ones
            if v2 < elite_val + 0.004 and remaining() > 65:
                X3 = _gd_opt(X2, 9000.0, 250, 0.0025)
                v3 = _max_cos(X3)
                if v3 < v2:
                    X2, v2 = X3, v3
            pool.append(X2.copy())
            if v2 < elite_val:
                elite = X2.copy(); elite_val = v2
        except Exception:
            continue

    # ---- basin hopping mixing elite and diverse pool ----
    cur = elite.copy(); cur_val = elite_val
    # keep small diverse set (best 4 distinct)
    try:
        pool_sorted = sorted(pool, key=_max_cos)[:4]
    except Exception:
        pool_sorted = [elite.copy()]
    trial = 0
    while remaining() > 26 and trial < 34:
        trial += 1
        try:
            r = rng.random()
            if r < 0.55:
                base = cur
            elif r < 0.8:
                base = elite
            else:
                base = pool_sorted[int(rng.integers(0, len(pool_sorted)))]
            Y = base.copy()
            mode = trial % 4
            if mode == 0:
                Y += rng.standard_normal(Y.shape) * 0.012
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_opt(Y, 3200.0, 300, 0.005)
            elif mode == 1:
                krep = 8 + (trial % 5)
                rmG = Y @ Y.T
                np.fill_diagonal(rmG, -2.0)
                rm = rmG.max(axis=1)
                order = np.argsort(-rm)
                victims = [int(v) for v in order[:krep]]
                for w in victims:
                    Nc = 6000
                    cand = rng.standard_normal((Nc, d))
                    cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                    sc = cand @ Y.T
                    sc[:, w] = -2.0
                    mx = sc.max(axis=1)
                    bi = int(np.argmin(mx))
                    Y[w] = cand[bi]
                Y += rng.standard_normal(Y.shape) * 0.0008
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_opt(Y, 2500.0, 300, 0.006)
                Y = _gd_opt(Y, 8000.0, 200, 0.0025)
            elif mode == 2:
                idx = rng.choice(n, 12, replace=False)
                R = rng.standard_normal((12, d))
                R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
                Y[idx] = R
                Y += rng.standard_normal(Y.shape) * 0.0006
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_opt(Y, 2000.0, 320, 0.007)
            else:
                Y += rng.standard_normal(Y.shape) * 0.03
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Y = _gd_opt(Y, 1500.0, 300, 0.008)
            v = _max_cos(Y)
            if v < elite_val - 1e-12:
                elite = Y.copy(); elite_val = v
                cur = Y.copy(); cur_val = v
            elif v < cur_val:
                cur = Y.copy(); cur_val = v
            elif v < cur_val + 2e-4 and rng.random() < 0.25:
                cur = Y.copy(); cur_val = v
            if cur_val > elite_val + 5e-4 and rng.random() < 0.35:
                cur = elite.copy(); cur_val = elite_val
        except Exception:
            continue

    # ---- final squeeze ----
    if remaining() > 8:
        try:
            nxt = _gd_opt(elite, 14000.0, 350, 0.002)
            v = _max_cos(nxt)
            if v < elite_val:
                elite = nxt; elite_val = v
        except Exception:
            pass
    if remaining() > 5:
        try:
            polished = _lp_polish(elite, time.time() + remaining() - 1.0)
            vp = _max_cos(polished)
            if vp < elite_val:
                elite = polished; elite_val = vp
        except Exception:
            pass
    elite = elite / np.maximum(np.linalg.norm(elite, axis=1, keepdims=True), 1e-12)
    elite = np.where(np.isfinite(elite), elite, 0.0)
    elite = elite / np.maximum(np.linalg.norm(elite, axis=1, keepdims=True), 1e-12)
    return elite

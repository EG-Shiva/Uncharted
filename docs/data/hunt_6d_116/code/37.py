import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    try:
        best = _maxcos(X)
    except Exception:
        return X
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx

def _build_G():
    G = np.zeros((6, 6))
    G[0, 5] = -1.0
    G[1, 0] = 1.0
    G[2, 1] = 1.0
    G[3, 2] = 1.0
    G[4, 3] = 1.0
    G[5, 4] = 1.0
    return G

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    rng = np.random.default_rng(20261031)
    # incumbent for final comparison only
    inc_best = None
    inc_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(f), dtype=float)
            if a.shape == (116, 6) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _maxcos(a)
                if v < inc_val:
                    inc_val = v
                    inc_best = a.copy()
        except Exception:
            pass

    Gmat = _build_G()
    Gks = [np.eye(6)]
    for k in range(1, 12):
        Gks.append(Gks[-1] @ Gmat)  # careful: Gks[k]=Gmat^k? need consistent direction
    # Actually Gmat acts as x->Gmat @ x (col). So powers:
    Gks = [np.linalg.matrix_power(Gmat, k) for k in range(12)]
    Gks_arr = np.stack(Gks, axis=0)  # 12,6,6
    # fixed plane of G^4
    G4 = Gks[4]
    U, S, Vt = np.linalg.svd(G4 - np.eye(6))
    e1 = Vt[-1, :].copy()
    e1 /= np.linalg.norm(e1)
    e2 = Vt[-2, :].copy()
    # orthogonalize
    e2 = e2 - float(e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)

    def build_antipodal(Y):
        return np.concatenate([Y, -Y], axis=0)

    def build_orbit(Ygen, thetas):
        # Ygen 9x6, thetas 2
        parts = []
        for k in range(12):
            Gk = Gks[k]
            parts.append(Ygen @ Gk.T)  # 9x6
        Xg = np.concatenate(parts, axis=0)  # 108x6 (order k-major? grouping fine)
        # special
        specs = []
        for th in thetas:
            s = float(np.cos(th)) * e1 + float(np.sin(th)) * e2
            for k in range(4):
                Gk = Gks[k]
                specs.append(Gk @ s)
        Xs = np.stack(specs, axis=0)  # 8x6
        return np.concatenate([Xg, Xs], axis=0)

    def optim_antipodal(Y0, steps, beta0, beta1, lr0, lr1, deadline):
        Y = _norm(Y0.copy())
        m1 = np.zeros_like(Y); m2 = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for s in range(1, steps + 1):
            if (s & 63) == 0 and time.time() > deadline:
                break
            frac = (s - 1) / max(steps - 1, 1)
            beta = beta0 * (beta1 / beta0) ** frac if beta0 > 0 and beta1 > 0 else beta0
            lr = lr0 + (lr1 - lr0) * frac
            X = build_antipodal(Y)
            Gm = X @ X.T
            np.fill_diagonal(Gm, -2.0)
            mx = float(Gm.max())
            W = np.exp(beta * (Gm - mx))
            np.fill_diagonal(W, 0.0)
            Sx = float(W.sum())
            if not np.isfinite(Sx) or Sx <= 0:
                break
            Gr = (W @ X) / Sx  # 116x6
            GY = Gr[:58] - Gr[58:]
            dot = np.sum(GY * Y, axis=1, keepdims=True)
            Gt = GY - dot * Y
            m1 = b1 * m1 + (1 - b1) * Gt
            m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
            c1 = 1 - b1 ** s; c2 = 1 - b2 ** s
            mh = m1 / c1; vh = m2 / c2
            step = mh / (np.sqrt(vh) + eps)
            d2 = np.sum(step * Y, axis=1, keepdims=True)
            stept = step - d2 * Y
            Y = Y - lr * stept
            Y = _norm(Y)
        return Y

    def optim_orbit(Yg0, th0, steps, beta0, beta1, lr0, lr1, deadline):
        Yg = _norm(Yg0.copy())
        th = np.array(th0, dtype=float).copy()
        m1 = np.zeros_like(Yg); m2 = np.zeros_like(Yg)
        mt1 = np.zeros_like(th); mt2 = np.zeros_like(th)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for s in range(1, steps + 1):
            if (s & 63) == 0 and time.time() > deadline:
                break
            frac = (s - 1) / max(steps - 1, 1)
            beta = beta0 * (beta1 / beta0) ** frac if beta0 > 0 and beta1 > 0 else beta0
            lr = lr0 + (lr1 - lr0) * frac
            X = build_orbit(Yg, th)
            Gm = X @ X.T
            np.fill_diagonal(Gm, -2.0)
            mx = float(Gm.max())
            W = np.exp(beta * (Gm - mx))
            np.fill_diagonal(W, 0.0)
            Sx = float(W.sum())
            if not np.isfinite(Sx) or Sx <= 0:
                break
            Gr = (W @ X) / Sx  # 116x6
            # layout of build_orbit: first 108 are k-major? we concatenated per-k blocks of 9
            # blocks k*9:(k+1)*9 correspond to Gk
            GYg = np.zeros_like(Yg)
            for k in range(12):
                blk = Gr[k * 9:(k + 1) * 9, :]  # 9x6
                GYg += blk @ Gks[k]
            dot = np.sum(GYg * Yg, axis=1, keepdims=True)
            Gt = GYg - dot * Yg
            m1 = b1 * m1 + (1 - b1) * Gt
            m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
            c1 = 1 - b1 ** s; c2 = 1 - b2 ** s
            mh = m1 / c1; vh = m2 / c2
            step = mh / (np.sqrt(vh) + eps)
            d2 = np.sum(step * Yg, axis=1, keepdims=True)
            stept = step - d2 * Yg
            Yg = Yg - lr * stept
            Yg = _norm(Yg)
            # theta grad
            # special rows 108:116, order: th0 k0..k3, th1 k0..k3
            gth = np.zeros_like(th)
            for ti in range(2):
                dLds = np.zeros(6)
                for k in range(4):
                    idx = 108 + ti * 4 + k
                    dLds += Gr[idx, :] @ Gks[k]
                tang = -float(np.sin(th[ti])) * e1 + float(np.cos(th[ti])) * e2
                gth[ti] = float(dLds @ tang)
            mt1 = b1 * mt1 + (1 - b1) * gth
            mt2 = b2 * mt2 + (1 - b2) * (gth * gth)
            mth = mt1 / c1; vth = mt2 / c2
            sth = mth / (np.sqrt(vth) + eps)
            th = th - lr * 0.5 * sth
        return Yg, th

    def optim_free(X0, steps, beta0, beta1, lr0, lr1, deadline):
        X = _norm(X0.copy())
        m1 = np.zeros_like(X); m2 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for s in range(1, steps + 1):
            if (s & 31) == 0 and time.time() > deadline:
                break
            frac = (s - 1) / max(steps - 1, 1)
            beta = beta0 * (beta1 / beta0) ** frac if beta0 > 0 and beta1 > 0 else beta0
            lr = lr0 + (lr1 - lr0) * frac
            Gm = X @ X.T
            np.fill_diagonal(Gm, -2.0)
            mx = float(Gm.max())
            W = np.exp(beta * (Gm - mx))
            np.fill_diagonal(W, 0.0)
            Sx = float(W.sum())
            if not np.isfinite(Sx) or Sx <= 0:
                break
            Gr = (W @ X) / Sx
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            m1 = b1 * m1 + (1 - b1) * Gt
            m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
            c1 = 1 - b1 ** s; c2 = 1 - b2 ** s
            mh = m1 / c1; vh = m2 / c2
            step = mh / (np.sqrt(vh) + eps)
            d2 = np.sum(step * X, axis=1, keepdims=True)
            stept = step - d2 * X
            X = X - lr * stept
            X = _norm(X)
        return X

    cand_list = []  # (val, X)

    # Phase A: antipodal restarts
    T_A_END = t0 + 52.0
    nA = 0
    while time.time() < T_A_END - 0.5:
        if time.time() > HARD - 60:
            break
        Y0 = _norm(rng.standard_normal((58, 6)))
        steps = 1100
        # vary beta range slightly per restart
        b0 = 50.0
        b1 = 1200.0 if (nA % 2 == 0) else 1500.0
        Yopt = optim_antipodal(Y0, steps, b0, b1, 0.03, 0.002, T_A_END)
        Xf = build_antipodal(_norm(Yopt))
        v = _maxcos(Xf)
        cand_list.append((v, Xf.copy()))
        nA += 1
        if len(cand_list) > 40:
            cand_list = sorted(cand_list, key=lambda z: z[0])[:12]
    cand_list = sorted(cand_list, key=lambda z: z[0])
    antip_best = cand_list[:3] if len(cand_list) >= 3 else cand_list[:]

    # Phase B: orbit restarts
    T_B_END = t0 + 92.0
    orb_cands = []
    nB = 0
    while time.time() < T_B_END - 0.5:
        if time.time() > HARD - 50:
            break
        Yg0 = _norm(rng.standard_normal((9, 6)))
        th0 = rng.uniform(0, 2 * np.pi, size=2)
        steps = 1100
        b0 = 50.0
        b1 = 1200.0 if (nB % 2 == 0) else 1500.0
        Yg, th = optim_orbit(Yg0, th0, steps, b0, b1, 0.03, 0.002, T_B_END)
        Xf = build_orbit(_norm(Yg), th)
        Xf = _norm(Xf)
        v = _maxcos(Xf)
        orb_cands.append((v, Xf.copy()))
        nB += 1
        if len(orb_cands) > 40:
            orb_cands = sorted(orb_cands, key=lambda z: z[0])[:12]
    orb_cands = sorted(orb_cands, key=lambda z: z[0])
    orbit_best = orb_cands[:3] if len(orb_cands) >= 3 else orb_cands[:]

    # Phase C: release + requench
    pool = []
    for v, X in (antip_best + orbit_best):
        pool.append((v, X))
    # keep only best ~5 unique
    pool = sorted(pool, key=lambda z: z[0])[:5]
    released = []
    T_C_END = HARD - 12.0
    for idx, (v0, X0) in enumerate(pool):
        if time.time() > T_C_END:
            break
        for rep in range(2):
            if time.time() > T_C_END:
                break
            scale = 0.01 if rep == 0 else 0.003
            # deterministic perturbation per candidate
            pert = rng.standard_normal(X0.shape)
            pert = pert - np.sum(pert * X0, axis=1, keepdims=True) * X0
            prms = float(np.sqrt(np.mean(np.sum(pert * pert, axis=1)))) + 1e-15
            pert *= (scale / prms)
            Xp = _norm(X0 + pert)
            # annealed free softmax
            steps = 700
            Xq = optim_free(Xp, steps, 30.0, 400.0, 0.03, 0.002, T_C_END)
            vq = _maxcos(Xq)
            released.append((vq, Xq.copy()))
            # extra finer quench from same point with higher beta
            if time.time() < T_C_END - 1:
                Xq2 = optim_free(Xq, 400, 200.0, 1200.0, 0.008, 0.001, T_C_END)
                vq2 = _maxcos(Xq2)
                released.append((vq2, Xq2.copy()))
    # also try pure random free restarts briefly to diversify (2 tries)
    while time.time() < T_C_END - 4 and len(released) < 14:
        Xr = _norm(rng.standard_normal((116, 6)))
        Xq = optim_free(Xr, 600, 30.0, 400.0, 0.03, 0.002, T_C_END)
        released.append((_maxcos(Xq), Xq.copy()))

    all_c = sorted(released, key=lambda z: z[0])
    # Phase D: LP polish best 2-3 released
    final_pool = all_c[:3] if len(all_c) >= 3 else all_c[:]
    polished = []
    for v, X in final_pool:
        if time.time() > HARD - 3:
            polished.append((v, X))
            continue
        dl = min(time.time() + 4.0, HARD - 2.0)
        try:
            Xp = _lp_polish(X, dl)
            vp = _maxcos(Xp)
        except Exception:
            Xp, vp = X, v
        polished.append((vp, Xp))
    # also include unpolished released best and constrained best
    for v, X in all_c[:3]:
        polished.append((v, X))
    for v, X in pool[:2]:
        polished.append((v, X))
    if inc_best is not None:
        polished.append((inc_val, inc_best))
    # pick best
    polished = sorted(polished, key=lambda z: z[0])
    bestX = polished[0][1].copy()
    bestX = _norm(bestX)
    # tiny shrink guard (no-op, ensure finite)
    if not np.all(np.isfinite(bestX)):
        bestX = inc_best.copy() if inc_best is not None else _norm(rng.standard_normal((116, 6)))
    return bestX

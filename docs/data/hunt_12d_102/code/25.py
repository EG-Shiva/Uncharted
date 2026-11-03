import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def adam_quench_free(x_init, steps, T0, T1, lr, t_end=None):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0); l1 = np.log(T1)
    for s in range(1, steps + 1):
        if t_end is not None and s % 200 == 0 and time.time() > t_end:
            break
        T = float(np.exp(l0 + (l1 - l0) * (s / steps)))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1 ** s)
        vh = v / (1.0 - b2 ** s)
        upd = mh / (np.sqrt(vh) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def sym_optim(S_init, Ccos, Csin, order, steps, T0, T1, lr):
    S = S_init.copy()
    ns = S.shape[0]
    mA = np.zeros_like(S); vA = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0); l1 = np.log(T1)
    n = ns * order
    for s in range(1, steps + 1):
        T = float(np.exp(l0 + (l1 - l0) * (s / steps)))
        # expand S -> X
        X = np.empty((n, 12))
        Xr = X.reshape(ns, order, 12)
        for b in range(6):
            c = Ccos[b]; sn = Csin[b]
            x0 = S[:, 2 * b]; x1 = S[:, 2 * b + 1]
            Xr[:, :, 2 * b] = x0[:, None] * c[None, :] - x1[:, None] * sn[None, :]
            Xr[:, :, 2 * b + 1] = x0[:, None] * sn[None, :] + x1[:, None] * c[None, :]
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) / T
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        Gr = Gt.reshape(ns, order, 12)
        GS = np.empty_like(S)
        for b in range(6):
            c = Ccos[b]; sn = Csin[b]
            G0 = Gr[:, :, 2 * b]; G1 = Gr[:, :, 2 * b + 1]
            GS[:, 2 * b] = np.sum(G0 * c[None, :] + G1 * sn[None, :], axis=1)
            GS[:, 2 * b + 1] = np.sum(-G0 * sn[None, :] + G1 * c[None, :], axis=1)
        # tangent to seeds
        d = np.sum(GS * S, axis=1, keepdims=True)
        GSt = GS - d * S
        mA = b1 * mA + (1.0 - b1) * GSt
        vA = b2 * vA + (1.0 - b2) * (GSt * GSt)
        mh = mA / (1.0 - b1 ** s)
        vh = vA / (1.0 - b2 ** s)
        upd = mh / (np.sqrt(vh) + eps)
        S = S - lr * upd
        nr = np.linalg.norm(S, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        S /= nr
    # final expansion
    X = np.empty((n, 12))
    Xr = X.reshape(ns, order, 12)
    for b in range(6):
        c = Ccos[b]; sn = Csin[b]
        x0 = S[:, 2 * b]; x1 = S[:, 2 * b + 1]
        Xr[:, :, 2 * b] = x0[:, None] * c[None, :] - x1[:, None] * sn[None, :]
        Xr[:, :, 2 * b + 1] = x0[:, None] * sn[None, :] + x1[:, None] * c[None, :]
    return S, X

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
    except Exception:
        xr = None
    if xb is not None and xb.shape == (102, 12):
        x_best = xb.copy(); best = max_cos(xb)
    elif xr is not None and xr.shape == (102, 12):
        x_best = xr.copy(); best = max_cos(xr)
    else:
        a = rng.standard_normal((102, 12)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a; best = max_cos(a)
    if xr is not None and xr.shape == (102, 12):
        v = max_cos(xr)
        if v < best:
            best, x_best = v, xr.copy()

    configs = [
        (17, 6, [1, 2, 4, 6, 8, 9], 12, 650),
        (17, 6, [1, 2, 3, 5, 7, 11], 6, 650),
        (34, 3, [1, 3, 5, 7, 9, 11], 6, 650),
    ]
    sym_best_val = 1e9
    sym_best_X = None
    for (order, ns, mult, nrest, steps) in configs:
        # precompute rotations
        Ccos = []; Csin = []
        for b in range(6):
            k = np.arange(order)
            th = 2.0 * np.pi * float(mult[b]) * k / float(order)
            Ccos.append(np.cos(th)); Csin.append(np.sin(th))
        for r in range(nrest):
            if time.time() > T_END - 95.0:
                break
            S0 = rng.standard_normal((ns, 12))
            S0 /= np.linalg.norm(S0, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.02, 0.05))
            T1 = float(rng.uniform(0.002, 0.005))
            lr = float(rng.uniform(0.015, 0.04))
            try:
                _, X = sym_optim(S0, Ccos, Csin, order, steps, T0, T1, lr)
            except Exception:
                continue
            # collision reject
            try:
                v = max_cos(X)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            if v < sym_best_val:
                sym_best_val = v
                sym_best_X = X.copy()
    # staged symmetry breaking from best symmetric code
    cand_X = None
    cand_v = best
    if sym_best_X is not None:
        # quick free quench of symmetric code at low temp (orbit-common then independent)
        X0 = sym_best_X / np.linalg.norm(sym_best_X, axis=1, keepdims=True)
        v0 = max_cos(X0)
        # keep it as candidate even if worse, for diversity
        trials = []
        trials.append((X0, 900, 0.015, 0.0025, 0.003))
        for ti in range(4):
            if time.time() > T_END - 60.0:
                break
            amp = [0.003, 0.008, 0.015, 0.03][ti % 4]
            Xn = X0 + amp * rng.standard_normal((102, 12))
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            T0t = float(rng.uniform(0.012, 0.028))
            T1t = float(rng.uniform(0.002, 0.0045))
            lrt = float(rng.uniform(0.002, 0.004))
            trials.append((Xn, 800, T0t, T1t, lrt))
        qbest = None; qbestv = 1e9
        for (Xi, st, a0, a1, lr_) in trials:
            if time.time() > T_END - 45.0:
                break
            try:
                Xq = adam_quench_free(Xi, st, a0, a1, lr_, t_end=T_END - 35.0)
                vq = max_cos(Xq)
            except Exception:
                continue
            if vq < qbestv:
                qbestv = vq; qbest = Xq.copy()
        if qbest is not None:
            # longer refine of the best released
            if time.time() < T_END - 45.0:
                try:
                    Xq2 = adam_quench_free(qbest, 900, 0.008, 0.0015, 0.002, t_end=T_END - 30.0)
                    vq2 = max_cos(Xq2)
                    if vq2 < qbestv:
                        qbestv = vq2; qbest = Xq2
                except Exception:
                    pass
            if qbestv < cand_v:
                cand_v = qbestv; cand_X = qbest
            # even if not better than incumbent, keep best released for final polish race
            if cand_X is None:
                cand_X = qbest; cand_v = qbestv
    # final race: polish incumbent and symmetric-released candidate
    # short polish of incumbent to have fresh best
    remaining = T_END - time.time()
    # allocate polish time: prefer candidate
    if cand_X is not None:
        # polish candidate
        t_pol = min(22.0, max(6.0, remaining * 0.55))
        try:
            Xp = lp_polish(cand_X, t_pol, T_END - 2.0)
            vp = max_cos(Xp)
            if vp < best:
                best, x_best = vp, Xp
        except Exception:
            pass
        # also polish incumbent briefly if time left and still not beaten
        if time.time() < T_END - 8.0:
            t2 = min(10.0, T_END - 2.0 - time.time())
            if t2 > 2.0:
                try:
                    Xp2 = lp_polish(x_best, t2, T_END - 1.0)
                    vp2 = max_cos(Xp2)
                    if vp2 < best:
                        best, x_best = vp2, Xp2
                except Exception:
                    pass
    else:
        t_pol = min(25.0, max(5.0, remaining - 2.0))
        try:
            Xp = lp_polish(x_best, t_pol, T_END - 1.0)
            vp = max_cos(Xp)
            if vp < best:
                best, x_best = vp, Xp
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

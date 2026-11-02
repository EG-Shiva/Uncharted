import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_quench(X0, steps, beta0, beta1, lam0, lr=0.02, seed=0):
    n, d = X0.shape
    X = X0.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Iod = np.eye(d) / d
    for t in range(1, steps + 1):
        frac = t / steps
        beta = beta0 * (beta1 / beta0) ** frac
        lam = lam0 * (1.0 - frac)
        G = X @ X.T
        gv = G[iu0, iu1]
        m = float(gv.max())
        e = np.exp(beta * (gv - m))
        s = float(e.sum())
        w = e / s
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        if lam != 0.0:
            C = (X.T @ X) / n
            D = C - Iod
            Gi = (4.0 / n) * (X @ D)
            Gr = Gr + lam * Gi
        # tangent projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # adam
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** t
        c2 = 1 - b2 ** t
        step = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        # clip step
        X = X - lr * step
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = X / rn
    return X

def _lp_polish(X, deadline, r_init=0.008, r_max=0.04, cap=2000):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bx = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() + 0.5 < deadline and radius > 1e-13:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; bx = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    base = None
    bv = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < bv:
                    bv = v; base = a.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        bv = _max_cos(base)
    x_best = base.copy(); v_best = bv

    # --- Stage 1: isotropic annealed Adam from random restarts (fast, vectorized)
    cands = []
    # number of restarts adapted to time; each ~800 steps is <0.3s
    N = 14
    for s in range(N):
        if time.time() > t0 + 45:
            break
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # vary schedules slightly for diversity
        if s % 3 == 0:
            b0, b1, l0, steps, lr = 50.0, 500.0, 40.0, 900, 0.02
        elif s % 3 == 1:
            b0, b1, l0, steps, lr = 80.0, 700.0, 15.0, 900, 0.015
        else:
            b0, b1, l0, steps, lr = 40.0, 400.0, 80.0, 700, 0.025
        Xq = _adam_quench(X0, steps, b0, b1, l0, lr=lr)
        v = _max_cos(Xq)
        cands.append((v, Xq))
        # also whitened variant of best-so-far quenched briefly (topology morph)
        if s % 4 == 3 and time.time() < t0 + 40:
            # whiten x_best then perturb -> isotropic version of record basin
            try:
                C = (x_best.T @ x_best) / n
                w, V = np.linalg.eigh(C)
                w = np.maximum(w, 1e-6)
                S = V @ np.diag(1.0 / np.sqrt(w * d)) @ V.T
                Xw = x_best @ S
                Xw /= np.linalg.norm(Xw, axis=1, keepdims=True)
                Xw = Xw + 0.02 * rng.standard_normal((n, d))
                Xw /= np.linalg.norm(Xw, axis=1, keepdims=True)
                Xq2 = _adam_quench(Xw, 500, 60.0, 500.0, 5.0, lr=0.015)
                v2 = _max_cos(Xq2)
                cands.append((v2, Xq2))
            except Exception:
                pass
    # also quench strongly perturbed record (large kick, new basin)
    for k in range(3):
        if time.time() > t0 + 50:
            break
        Xp = x_best + (0.15 + 0.1 * k) * rng.standard_normal((n, d))
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        Xq = _adam_quench(Xp, 700, 60.0, 600.0, 10.0, lr=0.02)
        cands.append((_max_cos(Xq), Xq))

    cands.sort(key=lambda t: t[0])
    # keep top few unique
    top = cands[:4]

    # --- Stage 2: LP polish top new topologies
    # budget: leave ~25s for final polish
    for i, (v, Xq) in enumerate(top):
        if time.time() > HARD_END - 28:
            break
        # skip if far worse than best (no hope after polish ~0.005)
        if v > v_best + 0.02 and i >= 2:
            continue
        budget = min(12.0, (HARD_END - 28 - time.time()) / (len(top) - i + 1e-9))
        if budget < 3:
            break
        Xp, vp = _lp_polish(Xq, time.time() + budget, r_init=0.008, r_max=0.04)
        if vp < v_best - 1e-13:
            x_best = Xp; v_best = vp

    # --- Stage 3: final intensive polish of best
    if time.time() < HARD_END - 1.0:
        x_best, v_best = _lp_polish(x_best, HARD_END - 0.6, r_init=0.003, r_max=0.02)
    if time.time() < HARD_END - 0.4:
        x_best, v_best = _lp_polish(x_best, HARD_END - 0.2, r_init=0.0006, r_max=0.005)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = base / np.linalg.norm(base, axis=1, keepdims=True)
    return x_best

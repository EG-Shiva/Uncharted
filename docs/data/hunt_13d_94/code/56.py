import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_eq(x0, steps, k0, k1, lr0, lr1, beta, Q):
    x = x0.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu0 = np.triu_indices(n, 1)
    cf = None
    Iact = Jact = None
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        # update active set every 20 steps
        if s % 20 == 0:
            cfull = G[iu0, iu1] if False else G[iu0[0], iu0[1]]
            # top Q
            Qc = int(min(Q, cfull.size))
            part = np.argpartition(cfull, cfull.size - Qc)[-Qc:]
            # order not needed
            Iact = iu0[0][part]
            Jact = iu0[1][part]
        mval = float(np.max(G - np.eye(n) * 2.0))
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ x) / S
        if beta != 0.0:
            # mean of active set
            cact = np.sum(x[Iact] * x[Jact], axis=1)
            # grad of mean
            Gm = np.zeros_like(x)
            np.add.at(Gm, Iact, x[Jact])
            np.add.at(Gm, Jact, x[Iact])
            Gm /= float(len(Iact))
            dotm = (Gm * x).sum(axis=1, keepdims=True)
            Gm_t = Gm - dotm * x
            dotg = (Grad * x).sum(axis=1, keepdims=True)
            Gt_lse = Grad - dotg * x
            Gt = Gt_lse + beta * Gm_t
        else:
            dotg = (Grad * x).sum(axis=1, keepdims=True)
            Gt = Grad - dotg * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 2500:
            # keep tightest 2500
            idx = np.argpartition(cmat[sel], -2500)[-2500:]
            I, J = I[idx], J[idx]
            p = 2500
            sels = cmat[sel][idx]
        else:
            sels = cmat[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-sels, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xn):
            xb = xr.copy()
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    best_X = xb.copy()
    best_v = _max_cos(best_X)
    init_v = best_v

    def try_adam_from(X0, steps, k0, k1, lr0, lr1, beta, Q):
        nonlocal best_X, best_v
        try:
            Y = _adam_eq(X0, steps, k0, k1, lr0, lr1, beta, Q)
        except Exception:
            return False
        if not np.all(np.isfinite(Y)):
            return False
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best_v - 1e-15:
            best_X = Y.copy()
            best_v = v
            return True
        return False

    # Stage 1: direct equalizing polishes of incumbent (no perturbation)
    configs = [
        (1200, 50.0, 3000.0, 0.015, 0.002, 0.4, 800),
        (1200, 80.0, 5000.0, 0.010, 0.001, 0.2, 1000),
        (1000, 30.0, 2000.0, 0.020, 0.003, 0.6, 600),
        (1000, 100.0, 4000.0, 0.008, 0.001, 0.0, 800),
    ]
    for cf in configs:
        if time.time() > END - 100.0:
            break
        try_adam_from(best_X.copy(), *cf)

    # Stage 2: basin hopping on worst points
    trial = 0
    sigmas = [0.004, 0.008, 0.015, 0.030, 0.060]
    npts_list = [6, 10, 16, 4, 20]
    betas = [0.3, 0.5, 0.2, 0.0, 0.7]
    while time.time() < END - 14.0:
        trial += 1
        # per-point worst max
        G = best_X @ best_X.T
        np.fill_diagonal(G, -2.0)
        prow = G.max(axis=1)
        order = np.argsort(-prow)
        si = sigmas[trial % len(sigmas)]
        # occasionally global shake
        if trial % 7 == 0:
            si_g = 0.003 + 0.002 * ((trial // 7) % 3)
            X0 = best_X + rng.standard_normal(size=(n, d)) * si_g
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            npts = n
        else:
            npts = npts_list[(trial // 1) % len(npts_list)]
            # sometimes random subset instead of worst
            if trial % 5 == 0:
                victims = rng.choice(n, size=npts, replace=False)
            else:
                # top npts worst + 2 random for diversity
                victims = order[:npts].copy()
                if rng.random() < 0.5:
                    victims[rng.integers(0, npts)] = rng.integers(0, n)
            X0 = best_X.copy()
            X0[victims] += rng.standard_normal(size=(len(victims), d)) * si
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        beta = betas[rng.integers(0, len(betas))]
        Q = int(rng.choice([600, 800, 1000, 1200]))
        k1 = float(rng.uniform(2000.0, 6000.0))
        k0 = float(rng.uniform(30.0, 100.0))
        lr0 = float(rng.uniform(0.008, 0.022))
        lr1 = lr0 * float(rng.uniform(0.08, 0.25))
        steps = 700 if (END - time.time()) < 60 else 1000
        # shrink steps if low time
        if END - time.time() < 30:
            steps = 500
        try_adam_from(X0, steps, k0, k1, lr0, lr1, beta, Q)

    # Stage 3: final LP polish
    if END - time.time() > 5.0:
        try:
            px, pv = _lp_polish_deadline(best_X.copy(), END - 1.0)
            if pv < best_v and np.all(np.isfinite(px)):
                best_X, best_v = px, pv
        except Exception:
            pass
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    # safety: never return worse than input
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(best_X) > _max_cos(x0) + 1e-12 and best_v > init_v + 1e-12:
            best_X = x0
    except Exception:
        pass
    return best_X

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _smooth_refine(X, iters, a0, a1, s0, s1):
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = a0 * (a1 / a0) ** frac if iters > 1 else a0
        step = s0 * (s1 / s0) ** frac if iters > 1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        # exp(alpha*(G-gmax)) in [-inf,0] -> safe
        W = np.exp(alpha * (G - gmax))
        np.fill_diagonal(W, 0.0)
        # sparsify tiny weights for stability
        # (threshold very small to keep focus)
        W[W < 1e-12] = 0.0
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        m = float(np.linalg.norm(Ft, axis=1).mean()) + 1e-12
        X = X - (step / m) * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X


def _hole_vector(X_rem, rng, n_sample=2500, hill_iters=160):
    n, d = X_rem.shape
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    dots = C @ X_rem.T
    m = dots.max(axis=1)
    idx = np.argpartition(m, 4)[:4]
    best_v = None
    best_s = 1e9
    for k in idx:
        v = C[k].copy()
        for it in range(hill_iters):
            dts = X_rem @ v
            cur = float(dts.max())
            mask = dts > cur - 0.04
            g = X_rem[mask].sum(axis=0)
            gt = g - float(g @ v) * v
            nn = float(np.linalg.norm(gt)) + 1e-12
            gt /= nn
            step = 0.08 * (0.15 ** (it / max(1, hill_iters)))
            v = v - step * gt
            v /= np.linalg.norm(v) + 1e-30
        s = float((X_rem @ v).max())
        if s < best_s:
            best_s = s
            best_v = v
    return best_v


def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.5:
            break
        Gf = X @ X.T
        c = Gf[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 3000:
            # keep most-violated 3000
            thr = np.sort(c[sel])[-3000]
            sel = c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    n, d = 162, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.array(a, dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-30
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestV = float(vals[bi])

    # short initial squeeze to get true local value (cheap)
    # keep time for exploration: only 4s here
    bestX = _lp_polish(bestX, 4.0)
    bestV = _max_cos(bestX)

    N_TRIAL = 130
    for trial in range(N_TRIAL):
        elapsed = time.time() - t0
        # reserve 38s for final polish
        if elapsed > 135.0:
            break
        # choose kill size: mostly 1-6, sometimes pure jitter
        if trial % 7 == 6:
            q = 0
        else:
            q = int(rng.integers(1, 7))  # 1..6
            if trial % 5 == 0:
                q = int(rng.integers(6, 10))  # bigger shake
        sigma = float(rng.uniform(0.004, 0.028))
        if q == 0:
            X = bestX.copy()
            X = X + sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
        else:
            X = bestX.copy()
            G = X @ X.T
            np.fill_diagonal(G, -np.inf)
            per_max = G.max(axis=1)
            order = np.argsort(-per_max)
            qq = min(q, n)
            kill_idx = order[:qq].copy()
            # diversify: randomly replace one victim with random point
            if rng.random() < 0.6:
                kill_idx[rng.integers(0, qq)] = int(rng.integers(0, n))
                kill_idx = np.unique(kill_idx)
            keep = np.ones(n, dtype=bool)
            keep[kill_idx] = False
            X_rem = X[keep]
            new_pts = []
            for _ in range(len(kill_idx)):
                v = _hole_vector(X_rem, rng, n_sample=2200, hill_iters=140)
                new_pts.append(v)
                X_rem = np.vstack([X_rem, v.reshape(1, -1)])
            X[kill_idx] = np.array(new_pts)
            # jitter remaining points a bit less
            X = X + (sigma * 0.7) * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30

        a0 = 60.0 + float(rng.uniform(0, 50.0))
        a1 = 300.0 + float(rng.uniform(0, 150.0))
        s0 = 0.030 + float(rng.uniform(0, 0.020))
        s1 = 0.002 + float(rng.uniform(0, 0.003))
        iters = 1100 + int(rng.integers(0, 500))
        X = _smooth_refine(X, iters, a0, a1, s0, s1)
        v = _max_cos(X)
        improved = v < bestV
        if improved:
            bestV = v
            bestX = X.copy()
            # lock in with short LP if time allows
            if time.time() - t0 < 125.0:
                bestX = _lp_polish(bestX, 3.0)
                bestV = _max_cos(bestX)
        else:
            # occasional polish of near-miss to find different basin optimum
            if (v < bestV + 0.003) and (trial % 12 == 0) and (time.time() - t0 < 120.0):
                Y = _lp_polish(X, 2.0)
                vy = _max_cos(Y)
                if vy < bestV:
                    bestV = vy
                    bestX = Y.copy()

    # final thorough polish
    remain = 175.0 - (time.time() - t0)
    final_sec = min(38.0, max(8.0, remain - 2.0))
    if final_sec > 1.0:
        bestX = _lp_polish(bestX, final_sec)
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30
    return bestX

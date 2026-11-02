import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_deadline(X, deadline, r_init=0.004):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() + 0.6 < deadline and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:3000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
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
            X = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x, best

def _whiten_center(X):
    n = X.shape[0]
    d = X.shape[1]
    # center in ambient then renormalize
    mu = X.mean(axis=0, keepdims=True)
    X = X - 0.5 * mu
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = (X.T @ X) / n
    wv, U = np.linalg.eigh(C)
    wv = np.maximum(wv, 1e-8)
    invsqrt = (U * (1.0 / np.sqrt(wv))) @ U.T
    M = invsqrt / np.sqrt(d)
    X = X @ M
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    mu2 = X.mean(axis=0, keepdims=True)
    X = X - 0.5 * mu2
    rn = np.linalg.norm(X, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    return X / rn

def _quench(X0, steps, beta0, beta1, s0, s1, noise0, rng, whiten_every, deadline):
    X = X0.copy()
    n, d = X.shape
    for t in range(steps):
        if (t & 127) == 0 and time.time() > deadline:
            break
        frac = t / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        step = s0 * (s1 / s0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # exp weights
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Dir = Gt / nr
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 1e-6:
                N = rng.standard_normal((n, d))
                N -= (N * X).sum(axis=1, keepdims=True) * X
                N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
                Dir = Dir + amp * 2.0 * N
                # renormalize direction per row to keep step scale stable
                Dir /= (np.linalg.norm(Dir, axis=1, keepdims=True) + 1e-12)
                # effective step slightly larger with noise; keep as is
        X = X - step * Dir
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        if whiten_every > 0 and (t + 1) % whiten_every == 0 and frac < 0.6:
            X = _whiten_center(X)
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    # load baseline
    baseX = None; baseV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < baseV:
                baseV = v; baseX = a.copy()
        except Exception:
            continue
    if baseX is None:
        baseX = rng.standard_normal((n, d))
        baseX /= np.linalg.norm(baseX, axis=1, keepdims=True)
        baseV = _max_cos(baseX)
    # reserve time for final LP
    RESERVE = 28.0
    screen_deadline = HARD - RESERVE
    cands = [(baseV, baseX.copy())]
    # param diversity
    k = 0
    while time.time() < screen_deadline and k < 30:
        # time per quench guard
        if HARD - RESERVE - time.time() < 1.0:
            break
        if k == 0:
            X0 = baseX + 0.55 * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            X0 = _whiten_center(X0)
            steps, beta0, beta1, s0, s1, nz, we = 1100, 25.0, 600.0, 0.06, 0.002, 0.6, 180
        elif k == 1:
            X0 = _whiten_center(baseX.copy())
            X0 = X0 + 0.25 * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps, beta0, beta1, s0, s1, nz, we = 1100, 30.0, 700.0, 0.05, 0.0015, 0.4, 220
        elif k == 2:
            X0 = rng.standard_normal((n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            X0 = _whiten_center(X0)
            steps, beta0, beta1, s0, s1, nz, we = 1300, 18.0, 500.0, 0.07, 0.002, 0.8, 200
        else:
            X0 = rng.standard_normal((n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            if (k % 2) == 0:
                X0 = _whiten_center(X0)
            # vary schedule deterministically
            beta1 = float(400 + 100 * ((k * 37) % 5))
            beta0 = float(15 + (k % 3) * 8)
            s0 = float(0.05 + 0.01 * (k % 4))
            s1 = 0.0015 if (k % 2 == 0) else 0.0025
            steps = 1000 + (k % 3) * 200
            nz = 0.5 + 0.2 * ((k * 13) % 3)
            we = 160 + (k % 4) * 40
        qdead = min(HARD - RESERVE, time.time() + 8.0)
        Xq = _quench(X0, steps, beta0, beta1, s0, s1, nz, rng, we, qdead)
        vq = _max_cos(Xq)
        cands.append((vq, Xq))
        # keep only top few to save memory
        cands.sort(key=lambda z: z[0])
        cands = cands[:5]
        k += 1
    cands.sort(key=lambda z: z[0])
    # final LP polish top 2 distinct candidates within remaining time
    bestV, bestX = cands[0][0], cands[0][1].copy()
    # ensure baseline included
    # polish best candidate
    time_left = HARD - time.time()
    if time_left > 4.0:
        # split budget: 70% to best, 30% to second if exists and different
        b1_dead = time.time() + time_left * (0.7 if len(cands) > 1 else 0.95)
        Xp, vp = _lp_polish_deadline(cands[0][1], b1_dead)
        if vp < bestV:
            bestV, bestX = vp, Xp
        if len(cands) > 1 and time.time() + 2.0 < HARD:
            # only try second if promising (within 0.02 of best)
            if cands[1][0] < bestV + 0.02:
                Xp2, vp2 = _lp_polish_deadline(cands[1][1], HARD - 0.5)
                if vp2 < bestV:
                    bestV, bestX = vp2, Xp2
    # safety: if somehow worse than baseline and baseline not polished, return baseline
    if bestV > baseV and baseV < 0.2:
        # best found is worse; return baseline (already best known)
        bestX = baseX
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # tiny jitter guard against NaN
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    rn = np.linalg.norm(bestX, axis=1, keepdims=True)
    rn = np.where(rn < 1e-12, 1.0, rn)
    bestX = bestX / rn
    return bestX

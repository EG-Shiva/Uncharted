import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _smooth_opt(X, steps, tau0, tau1, lr0, lr1, mu=0.9, rng=None):
    V = np.zeros_like(X)
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True) + 1e-30
        Grad = (W @ X) / rs
        d = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - d * X
        V = mu * V + (1.0 - mu) * Gt
        bc = 1.0 - mu ** (s + 1)
        Vn = V / bc
        X = X - lr * Vn
        X = _norm(X)
    return X


def _hard_refine(X, iters, step0, step1, tol=0.004):
    step = step0
    best = _max_cos(X)
    Xb = X.copy()
    for it in range(iters):
        f = it / max(1, iters - 1)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        window = tol * (1.0 - f * 0.7)
        Mask = G > (m - window)
        cnt = np.sum(Mask, axis=1, keepdims=True) + 1e-12
        Grad = (Mask.astype(float) @ X) / cnt
        d = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - d * X
        lr = step0 * (step1 / step0) ** f if step0 != step1 else step0
        Y = _norm(X - lr * Gt)
        v = _max_cos(Y)
        if v < m:
            X = Y
            if v < best:
                best = v
                Xb = Y.copy()
        else:
            # shrinking step on failure, small tangent kick
            X = _norm(X - (lr * 0.3) * Gt)
    return Xb


def _lp_polish(x, seconds, t_global_end):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iu0_0 = iu0[0]
    iu0_1 = iu0[1]
    while time.time() - start < seconds and time.time() < t_global_end:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        iu_vals = Gfull[iu0_0, iu0_1]
        m = float(np.max(iu_vals))
        sel = iu_vals > m - max(20 * radius, 1e-9)
        if np.sum(sel) < 1:
            break
        I = iu0_0[sel]
        J = iu0_1[sel]
        c = iu_vals[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # keep x at best if drifted
        if _max_cos(x) > best + 1e-12:
            x = xb.copy()
            radius *= 0.5
    return xb


def solve():
    t0 = time.time()
    BUDGET = 175.0
    t_end_global = t0 + BUDGET - 1.0
    rng = np.random.RandomState(12345)
    n = 168

    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = rng.randn(n, 4)
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = Xb.copy()
    Xb = _norm(Xb)
    Xr = _norm(Xr)
    vb = _max_cos(Xb)
    vr = _max_cos(Xr)
    if vr < vb:
        Xb = Xr.copy()
        vb = vr
    best_val = vb
    best_X = Xb.copy()

    cand_pool = [(best_val, best_X.copy())]

    def farthest_reinsert(Y, idx_list, n_cand=3000):
        for idx in idx_list:
            cand = _norm(rng.randn(n_cand, 4))
            Gc = cand @ Y.T
            Gc[:, idx] = -2.0
            worst = np.max(Gc, axis=1)
            pick = int(np.argmin(worst))
            Y[idx] = cand[pick]
        return Y

    trial = 0
    # exploration phase: reserve ~45s for LP finish
    while time.time() - t0 < BUDGET - 48.0:
        trial += 1
        Y = best_X.copy()
        mode = trial % 7
        if mode == 0:
            Y = _norm(Y + 0.012 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 600, 0.012, 0.002, 0.06, 0.008, 0.9
        elif mode == 1:
            Y = _norm(Y + 0.06 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 900, 0.018, 0.0015, 0.10, 0.008, 0.9
        elif mode == 2:
            Y = _norm(Y + 0.20 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1200, 0.025, 0.001, 0.12, 0.006, 0.88
        elif mode == 3:
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > (m - 0.004), axis=1)
            k = 4 + (trial % 5)
            idx = np.argsort(-deg)[:k]
            Y = farthest_reinsert(Y, idx, n_cand=3000)
            Y = _norm(Y + 0.02 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1100, 0.02, 0.001, 0.10, 0.005, 0.9
        elif mode == 4:
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > (m - 0.004), axis=1)
            idx_lo = np.argsort(deg)[:3]
            idx_hi = np.argsort(-deg)[:3]
            idx = np.unique(np.concatenate([idx_lo, idx_hi]))
            Y = farthest_reinsert(Y, idx, n_cand=3000)
            Y = _norm(Y + 0.012 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1100, 0.016, 0.0008, 0.09, 0.005, 0.9
        elif mode == 5:
            k = 10 + (trial % 8)
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = _norm(rng.randn(k, 4))
            Y = _norm(Y + 0.03 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1300, 0.025, 0.001, 0.12, 0.006, 0.88
        else:
            # full random restart to find new basin
            Y = _norm(rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1500, 0.03, 0.0012, 0.13, 0.006, 0.88

        remain = (BUDGET - 48.0) - (time.time() - t0)
        if remain < 6:
            steps = min(steps, 350)
        Y = _smooth_opt(Y, steps, tau0, tau1, lr0, lr1, mu=mu)
        # short hard refine to get true max estimate
        Y = _hard_refine(Y, 120, 0.006, 0.0008, tol=0.004)
        v = _max_cos(Y)
        if v < best_val - 1e-12:
            best_val = v
            best_X = Y.copy()
            cand_pool.append((v, Y.copy()))
            cand_pool = sorted(cand_pool, key=lambda z: z[0])[:4]
        elif v < best_val + 0.003 and len(cand_pool) < 6:
            # keep near-best diverse basins for LP
            cand_pool.append((v, Y.copy()))
            cand_pool = sorted(cand_pool, key=lambda z: z[0])[:6]
        if time.time() > t0 + BUDGET - 49.0:
            break

    # LP squeeze on top candidates
    cand_pool = sorted(cand_pool, key=lambda z: z[0])[:3]
    # allocate time: best gets most
    total_lp = max(5.0, t_end_global - time.time() - 1.0)
    if total_lp < 5:
        return _norm(best_X)
    shares = [0.6, 0.25, 0.15][:len(cand_pool)]
    shares = np.array(shares)
    shares = shares / shares.sum() * total_lp
    for (v0, X0), sec in zip(cand_pool, shares):
        if time.time() >= t_end_global - 0.5:
            break
        sec = min(sec, t_end_global - time.time() - 0.5)
        if sec < 1.0:
            continue
        Xp = _lp_polish(X0, sec, t_end_global - 0.3)
        vp = _max_cos(Xp)
        if vp < best_val:
            best_val = vp
            best_X = Xp.copy()
    # final tiny smooth touch if time remains (helps rounding)
    if time.time() < t_end_global - 2.0:
        Y = _smooth_opt(best_X.copy(), 400, 0.004, 0.0006, 0.02, 0.003, mu=0.9)
        Y = _hard_refine(Y, 80, 0.003, 0.0005, tol=0.003)
        v = _max_cos(Y)
        if v < best_val:
            best_val = v
            best_X = Y.copy()
    best_X = _norm(best_X)
    return best_X

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        if cnt > 2200:
            thr = np.sort(c)[-2200]
            sel = c >= thr
        I = iu[0][sel]; J = iu[1][sel]; cc = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
    return x

def _adam_smooth(X, steps, k0, k1, lr0, deadline, clip=0.05):
    n, d = X.shape
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lk0, lk1 = np.log(max(k0, 1e-9)), np.log(max(k1, 1e-9))
    b1t, b2t = 1.0, 1.0
    for t in range(1, steps + 1):
        if t % 100 == 0 and time.time() > deadline:
            break
        frac = t / steps
        k = float(np.exp(lk0 + (lk1 - lk0) * frac))
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s == 0.0:
            continue
        Grad = (W @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1t *= b1; b2t *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1t); vhat = v1 / (1.0 - b2t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -clip, clip)
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _greedy_reinsert(X, rng, n_try, n_points, deadline):
    n, d = X.shape
    X = X.copy()
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    cur = float(rowmax.max())
    for _ in range(2):
        if time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        improved = False
        for idx in order[:n_points]:
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool); mask[idx] = False
            Gr = float(G[np.ix_(mask, mask)].max())
            C = rng.standard_normal((n_try, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = X[mask] @ C.T
            bv = S.max(axis=0)
            j = int(np.argmin(bv))
            new_global = bv[j] if bv[j] > Gr else Gr
            if new_global < cur - 1e-12:
                X[idx] = C[j]
                cur = float(new_global)
                g = X @ X[idx]
                G[idx, :] = g; G[:, idx] = g; G[idx, idx] = -2.0
                improved = True
        if not improved:
            break
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        xr0 = rng.standard_normal((121, 8))
        xr0 /= np.linalg.norm(xr0, axis=1, keepdims=True)
        cands.append(xr0)
    base = None; base_val = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < base_val:
            base_val = v; base = c.copy()
    best = base.copy(); best_val = base_val
    # micro squeeze of incumbent with large-k smoothing
    for k, lr, steps, clip in [(500.0, 8e-5, 1500, 0.01), (2000.0, 3e-5, 1500, 0.005), (8000.0, 1e-5, 1200, 0.002)]:
        if time.time() > DEAD - 100:
            break
        X = _adam_smooth(best.copy(), steps, k, k, lr, DEAD - 90, clip=clip)
        v = _max_cos(X)
        if v < best_val:
            best = X; best_val = v
    if time.time() < DEAD - 95:
        best = _lp_polish(best, min(8.0, max(1.0, DEAD - 95 - time.time())), DEAD - 85)
        best_val = _max_cos(best)
    sigmas = [0.03, 0.015, 0.007, 0.003, 0.02, 0.01, 0.005, 0.0015]
    trial = 0
    while time.time() < DEAD - 18:
        sig = sigmas[trial % len(sigmas)]
        trial += 1
        Xb = best.copy()
        n, d = Xb.shape
        G = Xb @ Xb.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        nw = 10 if sig > 0.01 else 6
        worst = order[:nw]
        noise = rng.standard_normal((n, d)) * (sig * 0.3)
        noise[worst] += rng.standard_normal((len(worst), d)) * sig
        # occasional single-point teleport to change topology
        if trial % 7 == 0:
            idx = int(order[0])
            v0 = rng.standard_normal(d); v0 /= np.linalg.norm(v0)
            Xb[idx] = v0
            noise[idx] = 0.0
        X = Xb + noise
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if sig > 0.012:
            X = _adam_smooth(X, 900, 25.0, 350.0, 0.012, DEAD - 15, clip=0.05)
        elif sig > 0.004:
            X = _adam_smooth(X, 1000, 40.0, 800.0, 0.006, DEAD - 15, clip=0.03)
        else:
            X = _adam_smooth(X, 1100, 100.0, 2000.0, 0.002, DEAD - 15, clip=0.01)
        if time.time() > DEAD - 16:
            v = _max_cos(X)
            if v < best_val:
                best = X; best_val = v
            break
        X = _greedy_reinsert(X, rng, 2000, 4, DEAD - 14)
        X = _lp_polish(X, 2.5, DEAD - 12)
        v = _max_cos(X)
        if v < best_val - 1e-12:
            best = X; best_val = v
        # also try micro squeeze of this basin without accepting worse
        if trial % 3 == 0 and time.time() < DEAD - 25:
            Xs = _adam_smooth(X, 600, 3000.0, 3000.0, 2e-5, DEAD - 14, clip=0.003)
            vs = _max_cos(Xs)
            if vs < best_val - 1e-12:
                best = Xs; best_val = vs
    if time.time() < DEAD - 4:
        best = _lp_polish(best, max(0.5, DEAD - 4 - time.time()), DEAD - 2)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # strict finite check
    best = np.where(np.isfinite(best), best, 1e-3)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

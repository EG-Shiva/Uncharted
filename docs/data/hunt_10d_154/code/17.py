import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_refine(X, steps, a0, a1, lr0, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    for t in range(1, steps + 1):
        if t % 300 == 0 and time.time() > deadline:
            break
        frac = t / steps
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            break
        Grad = (2.0 / S) * (E @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mhat = m / c1
        vhat = v / c2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X

def quick_push(X, iters=150, eta0=0.02):
    X = X.copy()
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        thr = M - 0.008
        W = (G > thr).astype(float)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        eta = eta0 * (1.0 - 0.8 * it / max(iters, 1))
        X = X - eta * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    np.random.seed(0)
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
    n, d = 154, 10
    if len(cands) == 0:
        Xbest = rng.standard_normal((n, d))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    else:
        vals = []
        normed = []
        for c in cands:
            c = np.asarray(c, dtype=float)
            if c.shape != (n, d):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            normed.append(c)
            vals.append(max_cos(c))
        if len(normed) == 0:
            Xbest = rng.standard_normal((n, d))
            Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
        else:
            Xbest = normed[int(np.argmin(vals))].copy()
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best_val = max_cos(Xbest)

    trial = 0
    HOP_END = t0 + 132.0
    while time.time() < HOP_END and time.time() < DEAD - 25:
        trial += 1
        mode = trial % 5
        Gc = Xbest @ Xbest.T
        np.fill_diagonal(Gc, -2.0)
        thr_c = best_val - 0.01
        deg = np.sum(Gc > thr_c, axis=1).astype(float)
        rowmax = np.max(Gc, axis=1)
        order = np.argsort(rowmax)

        if mode == 0:
            sigma = 0.04 + 0.02 * ((trial // 5) % 3)
            Xtry = Xbest + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 800, 30.0, 450.0, 0.015
        elif mode == 1:
            prob = deg + 1.0
            prob = prob / prob.sum()
            k = int(rng.integers(6, 13))
            idx = rng.choice(n, size=k, replace=False, p=prob)
            Xtry = Xbest.copy()
            Xtry[idx] = rng.standard_normal((k, d))
            Xtry += 0.012 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1100, 25.0, 550.0, 0.015
        elif mode == 2:
            # spatial cluster: pick center, take K nearest, resample them
            K = int(rng.integers(8, 16))
            c = int(rng.integers(0, n))
            sims = Gc[c]
            nn = np.argsort(-sims)[:K]
            Xtry = Xbest.copy()
            Xtry[nn] = rng.standard_normal((K, d))
            Xtry += 0.01 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1200, 22.0, 550.0, 0.015
        elif mode == 3:
            # overload -> hole: move tight points near loose/rattler holes
            k_tight = 5
            k_loose = 5
            idx_tight = order[-8:]
            idx_loose = order[:8]
            pick_t = rng.choice(idx_tight, size=k_tight, replace=False)
            pick_l = rng.choice(idx_loose, size=k_loose, replace=False)
            Xtry = Xbest.copy()
            for ii, jj in zip(pick_t, pick_l):
                Xtry[ii] = Xbest[jj] + 0.25 * rng.standard_normal(d)
            # also randomize a couple rattlers fully
            if len(pick_l) > 0:
                Xtry[pick_l[:2]] = rng.standard_normal((min(2, len(pick_l)), d))
            Xtry += 0.008 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1200, 25.0, 600.0, 0.012
        else:
            # rare full random for diversity
            if trial % 15 != 0:
                # medium shake instead to save time
                Xtry = Xbest + 0.10 * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                steps, a0, a1, lr0 = 900, 28.0, 500.0, 0.015
            else:
                Xtry = rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                steps, a0, a1, lr0 = 1800, 20.0, 500.0, 0.02

        Xref = smooth_refine(Xtry, steps, a0, a1, lr0, deadline=HOP_END)
        val = max_cos(Xref)
        # cheap hard-descent estimate to filter
        if val < best_val + 0.008:
            Xq = quick_push(Xref, iters=120, eta0=0.015)
            vq = max_cos(Xq)
        else:
            vq = val

        improved = False
        if vq < best_val - 1e-9 or val < best_val - 5e-5:
            tl = min(DEAD - 22 - (time.time() - t0), 165 - (time.time() - t0))
            if tl < 1.0:
                if val < best_val:
                    Xbest, best_val = Xref, val
                    improved = True
            else:
                sec = min(3.5, max(1.2, tl / 10.0))
                # polish the better of Xref/Xq
                base = Xq if vq < val else Xref
                Xpol = lp_polish(base, sec)
                vpol = max_cos(Xpol)
                if vpol < best_val:
                    Xbest, best_val = Xpol, vpol
                    improved = True
                elif val < best_val:
                    Xbest, best_val = Xref, val
                    improved = True
        elif trial % 20 == 0 and time.time() < HOP_END - 5:
            # occasional polish of near-miss to escape
            if vq < best_val + 0.002:
                Xpol = lp_polish(Xq, 1.5)
                vpol = max_cos(Xpol)
                if vpol < best_val:
                    Xbest, best_val = Xpol, vpol
        if time.time() > HOP_END:
            break

    remaining = DEAD - (time.time() - t0) - 2.0
    # remaining is actually mis-scaled; recompute correctly
    remaining = (t0 + 172.0) - time.time() - 1.0
    if remaining > 4:
        try:
            Xbest = lp_polish(Xbest, remaining)
        except Exception:
            pass
    # final micro shake + repolish to squeeze
    t_left = (t0 + 174.0) - time.time()
    if t_left > 6:
        for _ in range(2):
            if time.time() > (t0 + 173.0):
                break
            Xt = Xbest + 2e-4 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = lp_polish(Xt, 1.0)
            vt = max_cos(Xt)
            if vt < best_val:
                Xbest, best_val = Xt, vt
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

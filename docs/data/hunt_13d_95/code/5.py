import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_refine(Y0, steps, beta0, beta1, lr0, lr1, mu=0.85):
    Y = Y0.copy()
    V = np.zeros_like(Y)
    for it in range(steps):
        frac = it / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True)
        EY = E @ Y
        GradAvg = EY / np.maximum(row, 1e-30)
        dot = np.sum(GradAvg * Y, axis=1, keepdims=True)
        Gt = GradAvg - dot * Y
        V = mu * V + (1.0 - mu) * Gt
        bc = 1.0 - mu ** (it + 1)
        Vhat = V / max(bc, 1e-12)
        Y = Y - lr * Vhat
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        Y = Y / np.maximum(nrm, 1e-12)
    return Y


def best_hole_sample(Xk, rng, n_samples=15000):
    n, d = Xk.shape[0], Xk.shape[1]
    C = rng.randn(n_samples, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xk.T
    scores = S.max(axis=1)
    idx = int(np.argmin(scores))
    return C[idx]


def refine_hole_vec(cc, Xk, steps=150, beta=200.0, lr=0.02):
    c = cc.copy()
    for _ in range(steps):
        g = Xk @ c
        mm = g.max()
        w = np.exp(beta * (g - mm))
        s = w.sum()
        if s == 0:
            break
        w /= s
        grad = w @ Xk
        gt = grad - float(grad @ c) * c
        c = c - lr * gt
        n = np.linalg.norm(c)
        if n < 1e-12:
            break
        c /= n
    return c


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
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


def solve():
    T0 = time.time()
    rng = np.random.RandomState(0)
    np.random.seed(0)
    n, d = 95, 13
    X_best = None
    best_val = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            X = np.load(fname).astype(float)
            if X.shape != (n, d):
                continue
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            v = max_cos(X)
            if v < best_val:
                best_val = v
                X_best = X.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = rng.randn(n, d)
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = max_cos(X_best)
    X_cur = X_best.copy()
    cur_val = best_val

    SEARCH_BUDGET = 115.0
    it = 0
    while time.time() - T0 < SEARCH_BUDGET and it < 5000:
        it += 1
        r = rng.rand()
        if r < 0.35:
            sigma = 0.004 + 0.012 * rng.rand()
            Xt = X_best + sigma * rng.randn(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, steps=700, beta0=60.0, beta1=450.0, lr0=0.02, lr1=0.002)
        elif r < 0.65:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.004).sum(axis=1)
            k = int(rng.randint(3, 8))
            order = np.argsort(-cnt + 0.15 * rng.randn(n))
            sel = order[:k]
            sigma = 0.08 + 0.12 * rng.rand()
            Xt = X_best.copy()
            Xt[sel] += sigma * rng.randn(k, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, steps=900, beta0=50.0, beta1=500.0, lr0=0.025, lr1=0.002)
        elif r < 0.85:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.003).sum(axis=1)
            order = np.argsort(-cnt + 0.05 * rng.randn(n))
            cand = order[:5]
            idx = int(cand[rng.randint(len(cand))])
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xk = X_best[mask]
            c = best_hole_sample(Xk, rng, n_samples=15000)
            c = refine_hole_vec(c, Xk, steps=150, beta=200.0, lr=0.02)
            Xt = X_best.copy()
            Xt[idx] = c
            Xt = smooth_refine(Xt, steps=900, beta0=50.0, beta1=500.0, lr0=0.02, lr1=0.002)
        else:
            sigma = 0.03 + 0.04 * rng.rand()
            Xt = X_cur + sigma * rng.randn(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, steps=600, beta0=40.0, beta1=300.0, lr0=0.02, lr1=0.003)
        G = Xt @ Xt.T
        np.fill_diagonal(G, -2.0)
        v = float(G.max())
        if not np.isfinite(v):
            continue
        if v < best_val - 1e-12:
            X_best = Xt
            best_val = v
            X_cur = Xt
            cur_val = v
        elif v < cur_val + 0.0015:
            elapsed = time.time() - T0
            Tt = 0.001 * max(0.0, 1.0 - elapsed / 130.0)
            Tt = max(Tt, 2e-04)
            if v <= cur_val or rng.rand() < np.exp((cur_val - v) / Tt):
                X_cur = Xt
                cur_val = v

    # final smooth tighten
    try:
        X_best = smooth_refine(X_best, steps=1200, beta0=100.0, beta1=800.0, lr0=0.012, lr1=0.001)
        bv = max_cos(X_best)
        if np.isfinite(bv):
            best_val = bv
    except Exception:
        pass

    remaining = 172.0 - (time.time() - T0)
    if remaining > 8:
        try:
            X_best = lp_polish(X_best, seconds=min(30.0, remaining - 2.0))
        except Exception:
            pass

    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    # safety: replace any bad row
    for i in range(n):
        if not np.all(np.isfinite(X_best[i])) or np.linalg.norm(X_best[i]) < 1e-9:
            v = rng.randn(d)
            v /= np.linalg.norm(v)
            X_best[i] = v
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

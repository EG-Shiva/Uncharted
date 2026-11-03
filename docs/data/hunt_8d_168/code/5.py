import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_opt(X, steps, a0, a1, lr0, lr1):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best = X.copy()
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S == 0 or not np.isfinite(S):
            continue
        Grad = (E @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        X -= lr * Gt
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        X /= n
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_m:
        best = X.copy()
    return best


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def build_E8():
    vecs = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    for mask in range(256):
        cnt = 0
        s = np.ones(8)
        for k in range(8):
            if (mask >> k) & 1:
                s[k] = -1.0
                cnt += 1
        if cnt % 2 == 0:
            vecs.append(0.5 * s)
    E = np.array(vecs, dtype=float)
    E /= np.sqrt(2.0)
    return E


def solve():
    np.random.seed(0)
    t0 = time.time()
    TOTAL = 172.0
    try:
        X_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            X_best = np.load("record.npy").astype(float)
        except Exception:
            X_best = np.random.randn(168, 8)
    X_best = X_best.reshape(168, 8)
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    best_val = max_cos(X_best)
    X_cur = X_best.copy()
    cur_val = best_val
    E8 = build_E8()
    T = 0.003
    it = 0
    # reserve ~32s for final polish
    while time.time() - t0 < TOTAL - 32:
        it += 1
        r = np.random.rand()
        if r < 0.12 and it > 4:
            X_init = np.random.randn(168, 8)
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            steps, a0, a1, lr0, lr1 = 3000, 12.0, 400.0, 0.05, 0.003
        elif r < 0.28:
            idx = np.random.choice(240, 168, replace=False)
            X_init = E8[idx] + 0.02 * np.random.randn(168, 8)
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            steps, a0, a1, lr0, lr1 = 2200, 15.0, 450.0, 0.04, 0.0025
        else:
            base = X_cur if np.random.rand() < 0.7 else X_best
            sigma = float(np.random.choice([0.005, 0.01, 0.02, 0.04, 0.08, 0.15]))
            X_init = base + sigma * np.random.randn(168, 8)
            if np.random.rand() < 0.55:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                worst = np.argsort(mx)[-5:]
                X_init[worst] += np.random.randn(5, 8) * sigma * 2.0
            X_init /= np.linalg.norm(X_init, axis=1, keepdims=True)
            steps, a0, a1, lr0, lr1 = 1800, 20.0, 500.0, 0.04, 0.002
            if sigma >= 0.08:
                steps = 2500
                a0 = 12.0
        X_opt = smooth_opt(X_init, steps, a0, a1, lr0, lr1)
        v = max_cos(X_opt)
        if v < best_val:
            best_val = v
            X_best = X_opt.copy()
            X_cur = X_opt.copy()
            cur_val = v
        else:
            if v < cur_val or np.random.rand() < np.exp(-(v - cur_val) / max(T, 1e-6)):
                X_cur = X_opt.copy()
                cur_val = v
        T = max(T * 0.997, 0.00025)
        if time.time() - t0 > TOTAL - 34:
            break
    # high-alpha squeeze
    try:
        if time.time() - t0 < TOTAL - 28:
            X_best = smooth_opt(X_best, 1200, 100.0, 900.0, 0.01, 0.001)
    except Exception:
        pass
    remain = TOTAL - (time.time() - t0) - 1.0
    if remain > 4:
        try:
            X_best = lp_polish(X_best, seconds=min(remain, 28.0))
        except Exception:
            pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

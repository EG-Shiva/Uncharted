import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(X, steps, lr, tau0, tau1, mu):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * (1 - f) + tau1 * f
        if tau < 1e-4:
            tau = 1e-4
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # exponential weights, sparsify far pairs
        # shift for stability: (G-m)/tau <= 0
        W = np.exp((G - m) / tau)
        # cut tiny weights to sparsify
        W[W < 1e-3] = 0.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        V = mu * V + F
        lrs = lr * (1.0 - 0.5 * f)
        X = X - lrs * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    def probs_from(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # hubs: count near-max contacts
        C = G >= (mv - 5e-3)
        deg = C.sum(axis=1).astype(float)
        # also worst row max
        rowmax = G.max(axis=1)
        w = (deg + 1.0) * (1.0 + 4.0 * np.clip((rowmax - (mv - 5e-3)) / 5e-3, 0, 1))
        w = w ** 1.4 + 0.2
        return w / w.sum(), deg, rowmax

    probs, _, _ = probs_from(cur, cur_val)

    it = 0
    # phase 1: annealed hopping until ~125s
    T_END1 = t0 + 125.0
    while time.time() - t0 < T_END1:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(1e-9, (T_END1 - t0))
        temp = 0.003 * (1.0 - frac) + 2e-6
        if it % 8 == 1:
            probs, _, _ = probs_from(cur, cur_val)
        mode = it % 8
        Y = cur.copy()
        if mode == 0:
            Y += 0.008 * rng.standard_normal(Y.shape)
        elif mode == 1:
            Y += 0.020 * rng.standard_normal(Y.shape)
        elif mode == 2:
            k, sig = 1, 0.18
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif mode == 3:
            k, sig = 2, 0.14
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif mode == 4:
            k, sig = 3, 0.20
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
        elif mode == 5:
            # largest-hole reinsertion of worst point(s)
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            idx = int(np.argmax(rm))
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            Xk = Y[keep]
            cand = rng.standard_normal((3000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ Xk.T
            worst = dots.max(axis=1)
            bi = int(np.argmin(worst))
            Y[idx] = cand[bi]
            # small jitter around hole
            Y[idx] += 0.01 * rng.standard_normal(d)
        elif mode == 6:
            # double hole reinsert
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            idxs = np.argsort(rm)[-2:]
            keep = np.ones(n, dtype=bool)
            keep[idxs] = False
            Xk = Y[keep]
            for idx in idxs:
                cand = rng.standard_normal((2000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xk.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                Y[idx] = cand[bi]
                # extend kept set
                Xk = np.vstack([Xk, Y[idx:idx+1]])
        else:
            Y += 0.010 * rng.standard_normal(Y.shape)
            k, sig = 1, 0.30
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))

        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        # choose smoothing schedule by mode
        if mode in (0, 1, 7):
            steps, lr, ta0, ta1, mu = 250, 0.03, 0.008, 0.002, 0.85
        elif mode in (2, 3, 4):
            steps, lr, ta0, ta1, mu = 350, 0.04, 0.006, 0.0015, 0.88
        else:
            steps, lr, ta0, ta1, mu = 400, 0.045, 0.007, 0.0012, 0.88

        # quick budget check: skip long polish if little time
        Y = _smooth(Y, steps, lr, ta0, ta1, mu)
        v = _max_cos(Y)
        if v < best_val:
            best_val = float(v)
            best = Y.copy()
            cur = Y.copy()
            cur_val = float(v)
            probs, _, _ = probs_from(cur, cur_val)
        elif v < cur_val or rng.random() < np.exp((cur_val - v) / max(1e-9, temp)):
            cur = Y.copy()
            cur_val = float(v)

    # phase 2: final squeeze with LP (remaining time, leave 5s margin)
    remain = (t0 + 175.0) - time.time()
    if remain > 5:
        # short smooth with tiny tau to centre basin before LP
        try:
            best = _smooth(best, 200, 0.015, 0.003, 0.0008, 0.9)
            bv = _max_cos(best)
            if bv < best_val:
                best_val = bv
        except Exception:
            pass
        remain = (t0 + 175.0) - time.time()
        if remain > 5:
            best = _lp_polish(best, remain)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

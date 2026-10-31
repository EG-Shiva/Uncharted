import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_descent(X, steps, T0, T1, lr0, lr1, mu=0.9, ms0=0.04, ms1=0.004):
    X = X.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / float(max(1, steps))
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        ms = ms0 * (ms1 / ms0) ** frac
        if T < 1e-12:
            T = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp((G - gmax) / T)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        P = W / S
        Grad = P @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        raw = lr * V
        rn = np.sqrt(np.sum(raw * raw, axis=1, keepdims=True)) + 1e-12
        factor = np.minimum(1.0, ms / rn)
        step = raw * factor
        X = X - step
        nr = np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + 1e-12
        X = X / nr
    return X


def lp_polish(x, seconds, deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
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


def random_hole(cur_set, rng, n_samples=8000):
    S = rng.normal(size=(n_samples, 4))
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    D = S @ cur_set.T
    md = D.max(axis=1)
    bi = int(np.argmin(md))
    return S[bi]


def solve():
    t0 = time.time()
    LIMIT = 172.0
    n = 168
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (168, 4):
                cands.append(a)
        except Exception:
            pass
    X_best = None
    best = 1e9
    for c in cands:
        c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
        v = max_cos(c)
        if v < best:
            best = v
            X_best = c.copy()
    rng = np.random.default_rng(0)
    if X_best is None:
        X_best = rng.normal(size=(n, 4))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best = max_cos(X_best)

    it = 0
    PHASE1_END = t0 + 122.0
    while time.time() < PHASE1_END and (time.time() - t0) < LIMIT - 45:
        it += 1
        if it % 10 == 0:
            X = rng.normal(size=(n, 4))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            Xopt = smooth_descent(X, 800, 0.015, 0.0015, 0.30, 0.02)
            v = max_cos(Xopt)
            if v < best:
                best = v
                X_best = Xopt
            continue
        X = X_best.copy()
        G0 = X_best @ X_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        deg0 = np.sum(G0 > m0 - 0.005, axis=1)
        order = np.argsort(-deg0, kind="stable")
        r = rng.random()
        if r < 0.55:
            if rng.random() < 0.7:
                top = order[:40]
                center = int(rng.choice(top))
            else:
                center = int(rng.integers(n))
            dots = X_best[center] @ X_best.T
            nn = np.argsort(-dots, kind="stable")[:12]
            sigma = 0.02 + 0.08 * rng.random()
            X[nn] += rng.normal(0, sigma, size=(len(nn), 4))
            X += rng.normal(0, 0.005, size=X.shape)
        elif r < 0.85:
            k = int(rng.integers(2, 7))
            pool = order[:50]
            k = min(k, len(pool))
            chosen = rng.choice(pool, size=k, replace=False)
            # include rattler if present
            zero_idx = np.where(deg0 == 0)[0]
            if len(zero_idx) > 0 and rng.random() < 0.5:
                chosen[0] = int(zero_idx[0])
            remaining = np.delete(X, chosen, axis=0)
            cur_set = remaining.copy()
            for idx in chosen:
                h = random_hole(cur_set, rng, n_samples=8000)
                X[int(idx)] = h
                cur_set = np.vstack([cur_set, h[None, :]])
            X += rng.normal(0, 0.003, size=X.shape)
        else:
            sigma = 0.03 + 0.05 * rng.random()
            X += rng.normal(0, sigma, size=X.shape)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        steps = int(400 + rng.integers(0, 600))
        T0 = float(10.0 ** rng.uniform(-2.6, -1.7))
        T1 = T0 * 0.1
        lr0 = float(rng.uniform(0.15, 0.40))
        lr1 = lr0 * 0.08
        Xopt = smooth_descent(X, steps, T0, T1, lr0, lr1)
        v = max_cos(Xopt)
        if v < best:
            best = v
            X_best = Xopt

    remaining = LIMIT - (time.time() - t0) - 3.0
    if remaining > 8.0:
        lp_time = min(remaining - 2.0, 35.0)
        try:
            X_best = lp_polish(X_best, lp_time, deadline=t0 + LIMIT - 2.0)
        except Exception:
            pass
        try:
            best = max_cos(X_best)
        except Exception:
            pass
        # final tiny anneal to squeeze ties
        if time.time() < t0 + LIMIT - 5.0:
            try:
                X2 = smooth_descent(X_best, 300, 0.002, 0.0005, 0.10, 0.01)
                if max_cos(X2) < best:
                    X_best = X2
                    best = max_cos(X_best)
            except Exception:
                pass

    X_best = X_best / (np.linalg.norm(X_best, axis=1, keepdims=True) + 1e-12)
    # strict finite check
    X_best = np.where(np.isfinite(X_best), X_best, 0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr = np.where(nr < 1e-12, 1.0, nr)
    X_best = X_best / nr
    return X_best

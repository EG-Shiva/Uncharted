import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return best_x

def smooth_opt(X, steps, b0, b1, lr0, lr1, mu=0.9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    V = np.zeros_like(X)
    eye2 = 2.0 * np.eye(n)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        m = np.max(G - eye2)
        A = beta * (G - m)
        np.clip(A, -12, 0, out=A)
        np.fill_diagonal(A, -30.0)
        np.exp(A, out=A)
        Grad = A @ X
        d = np.sum(Grad * X, axis=1, keepdims=True)
        T = Grad - d * X
        s = float(np.mean(np.sqrt(np.sum(T * T, axis=1)))) + 1e-12
        T /= s
        V *= mu
        V += (1.0 - mu) * T
        X -= lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD = 176.0
    n, d = 153, 4
    rng = np.random.default_rng(20261031)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    # contact info of current best
    def contacts_of(X, thr_off=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        cnt = np.sum(G > mx - thr_off, axis=1)
        return cnt, mx
    trial = 0
    sigmas = [0.015, 0.03, 0.06, 0.10, 0.02, 0.05, 0.08]
    while time.time() - t0 < 132.0 and trial < 60:
        trial += 1
        if trial % 6 == 0:
            X = rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = 2500, 12.0, 250.0, 0.025, 0.002
        else:
            sigma = sigmas[(trial - 1) % len(sigmas)]
            X = best + sigma * rng.standard_normal((n, d))
            # targeted re-insertions
            try:
                cnt, mx = contacts_of(best)
                ratt = np.where(cnt == 0)[0]
                order = np.argsort(-cnt)
                hubs = order[:12]
            except Exception:
                ratt = np.array([], dtype=int)
                hubs = np.array([], dtype=int)
            # randomize rattlers fully
            for idx in ratt:
                if time.time() - t0 > 132.0:
                    break
                X[idx] = rng.standard_normal(d)
            # 1-3 extra replacements: mix of hubs and random
            krep = 1 + (trial % 3)
            cand = []
            if len(hubs) > 0:
                # pick deterministic subset using rng
                pi = rng.choice(hubs, size=min(krep, len(hubs)), replace=False)
                cand.extend(list(pi))
            # add one fully random index to diversify
            if rng.random() < 0.6:
                cand.append(int(rng.integers(0, n)))
            for idx in cand[:3]:
                # large jump for chosen index
                if rng.random() < 0.5:
                    X[idx] = rng.standard_normal(d)
                else:
                    X[idx] += 0.35 * rng.standard_normal(d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = 1800, 30.0, 320.0, 0.020, 0.002
            if sigma >= 0.08:
                steps = 2200
        X = smooth_opt(X, steps, b0, b1, lr0, lr1, mu=0.9)
        v = max_cos(X)
        # decide LP budget
        elapsed = time.time() - t0
        remain = HARD - elapsed - 16.0
        if remain < 1.5:
            if v < best_val - 1e-12:
                best, best_val = X, v
            break
        # only snap promising candidates
        if v < best_val + 0.02:
            if v < best_val + 0.002:
                budget = min(8.0, remain)
            else:
                budget = min(4.0, remain)
            Xp = lp_polish(X, budget)
            vp = max_cos(Xp)
            if vp < best_val - 1e-12:
                best, best_val = Xp, vp
        else:
            if v < best_val - 1e-12:
                best, best_val = X, v
    remain = HARD - (time.time() - t0) - 1.0
    if remain > 2.0:
        best = lp_polish(best, min(remain, 28.0))
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

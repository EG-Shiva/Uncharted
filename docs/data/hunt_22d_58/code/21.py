import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, a0, a1, lr0, lr1):
    n, d = X.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.linalg.norm(Gt, axis=1).mean() + 1e-12)
        X = X - (lr / gn) * Gt * 2.0
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (X @ X.T)[iu]
        m = float(C.max())
        sel = C > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = C[sel]
        # cap LP size for speed
        if len(I) > 900:
            k = np.argsort(-c)[:900]
            I = I[k]; J = J[k]; c = c[k]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    HOP_END = t0 + 120.0
    rng = np.random.default_rng(7)
    n, d = 58, 22
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (58, 22):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append(c)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((58, 22))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    # short clean of starting point
    try:
        if time.time() < HOP_END:
            best = _quench(best, 400, 400, 2000, 0.004, 0.001)
            best_val = _max_cos(best)
    except Exception:
        pass

    presets = [
        (80, 700, 0.03, 0.004, 700),
        (120, 1200, 0.025, 0.003, 900),
        (50, 500, 0.04, 0.005, 600),
        (150, 1800, 0.02, 0.002, 1000),
    ]
    it = 0
    while time.time() < HOP_END and it < 120:
        it += 1
        a0, a1, lr0, lr1, steps = presets[it % len(presets)]
        y = best.copy()
        mode = it % 5
        if mode == 0:
            sig = 0.012 + 0.008 * ((it // 5) % 3)
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            k = 2 + (it % 5)
            idx = rng.choice(n, size=k, replace=False)
            y = y + 0.005 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            r = rng.standard_normal((k, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r
        elif mode == 2:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(-cnt, kind="stable")
            k = int(rng.integers(0, 3))
            victim = int(order[k])
            # best-hole resampling for victim
            mask = np.ones(n, dtype=bool); mask[victim] = False
            Y = y[mask]
            Cc = rng.standard_normal((1500, d))
            Cc /= np.linalg.norm(Cc, axis=1, keepdims=True)
            dots = Cc @ Y.T
            worst = dots.max(axis=1)
            y[victim] = Cc[int(np.argmin(worst))]
            # jitter neighbours
            idx2 = rng.choice(n, size=4, replace=False)
            y[idx2] += 0.015 * rng.standard_normal((4, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 3:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            i0, j0 = ij // n, ij % n
            cand = [int(i0), int(j0)] + rng.choice(n, size=3, replace=False).tolist()
            y[cand] += 0.06 * rng.standard_normal((len(cand), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            k = 1 + (it % 2)
            idx = rng.choice(n, size=k, replace=False)
            y = y + 0.007 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            r = rng.standard_normal((k, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r
            idx2 = rng.choice(n, size=5, replace=False)
            y[idx2] += 0.02 * rng.standard_normal((5, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        if time.time() + 0.4 > HOP_END:
            break
        try:
            y = _quench(y, steps, a0, a1, lr0, lr1)
            y = _quench(y, 150, 1500, 2500, 0.002, 0.0008)
        except Exception:
            continue
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
        if time.time() > HOP_END:
            break

    # final LP polish with remaining budget
    remain_deadline = min(DEADLINE - 1.0, time.time() + 30.0)
    if time.time() < remain_deadline - 2.0:
        try:
            polished = _lp_polish(best, remain_deadline)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

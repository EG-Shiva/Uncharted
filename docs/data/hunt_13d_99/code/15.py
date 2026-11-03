import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_refine(v, alpha, steps, lr, beta=0.9, rng=None):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # shift for stability, clip exponent
        E = alpha * (G - m)
        # avoid overflow: E<=0 already
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - cur_lr * step
        nrm = np.linalg.norm(cur, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        cur /= nrm
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def far_reinsert(base, idx, rng, n_cand=2000):
    x0 = base.copy()
    n, d = base.shape
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 128.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        # try to salvage
        try:
            m = min(xb.shape[0], 99)
            tmp[:m, :min(xb.shape[1], 13)] = xb[:m, :min(xb.shape[1], 13)] if xb.ndim == 2 else 0
        except Exception:
            pass
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    sigmas = [0.015, 0.04, 0.09, 0.16, 0.28, 0.45]
    alphas = [25.0, 40.0, 60.0, 90.0, 130.0]
    it = 0
    while time.time() - t0 < EXPLORE and it < 400:
        it += 1
        remain = EXPLORE - (time.time() - t0)
        if remain < 2:
            break
        base = best_x
        mode = rng.random()
        if mode < 0.35:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            x0 = base + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            alpha = float(alphas[int(rng.integers(0, len(alphas)))])
            steps = int(rng.integers(400, 1000))
            lr = float(rng.uniform(0.03, 0.12))
        elif mode < 0.60:
            # subset shake on random subset
            m = int(rng.integers(2, 11))
            idx = rng.choice(n, size=m, replace=False)
            sigma = float(rng.uniform(0.1, 0.5))
            x0 = base.copy()
            x0[idx] += rng.standard_normal(size=(m, d)) * sigma
            x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
            alpha = float(alphas[int(rng.integers(0, len(alphas)))])
            steps = int(rng.integers(500, 1100))
            lr = float(rng.uniform(0.03, 0.12))
        elif mode < 0.85:
            # kill worst points and far reinsert
            ppm = per_point_max(base)
            q = int(rng.integers(1, 4))
            idx = np.argsort(ppm)[-q:]
            # sometimes also kill overloaded (high contact count)
            x0 = far_reinsert(base, idx, rng, n_cand=2000)
            # small jitter after
            x0 += rng.standard_normal(size=(n, d)) * 0.01
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            alpha = float(rng.choice([40.0, 60.0, 90.0]))
            steps = int(rng.integers(600, 1200))
            lr = float(rng.uniform(0.04, 0.10))
        else:
            # shake overloaded hubs: estimate contacts
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = G.max() - 1e-3
            cnt = (G > thr).sum(axis=1)
            # top overloaded
            q = int(rng.integers(3, 9))
            idx = np.argsort(cnt)[-q:]
            x0 = base.copy()
            sigma = float(rng.uniform(0.15, 0.4))
            x0[idx] += rng.standard_normal(size=(len(idx), d)) * sigma
            x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
            alpha = float(alphas[int(rng.integers(0, len(alphas)))])
            steps = int(rng.integers(500, 1000))
            lr = float(rng.uniform(0.03, 0.12))
        try:
            cand, sc = smooth_refine(x0, alpha, steps, lr)
        except Exception:
            continue
        if sc < best_val - 1e-12:
            best_val = sc
            best_x = cand.copy()
    # final polish with remaining time
    rem = BUDGET - (time.time() - t0)
    rem = max(5.0, min(rem, 45.0))
    try:
        polished = lp_polish(best_x, rem - 1.0)
        pv = max_cos(polished)
        if pv <= best_val:
            best_x = polished
            best_val = pv
        else:
            # keep polished if essentially equal (tighter contacts)
            if pv < best_val + 1e-9:
                best_x = polished
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_run(cur, alpha, steps, lr, beta, rng):
    n, d = cur.shape
    mom = np.zeros_like(cur)
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        cur = cur - lr * (mom / bc)
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
    return cur

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
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
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        v = _max_cos(y)
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
    LP_TIME = 35.0
    EXPLORE_UNTIL = BUDGET - LP_TIME
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    best_x = xb.copy()
    best_v = _max_cos(best_x)
    it = 0
    sigmas = [0.03, 0.07, 0.12, 0.20, 0.32, 0.50]
    while time.time() - t0 < EXPLORE_UNTIL:
        it += 1
        base = best_x
        u = rng.random()
        sigma = sigmas[rng.integers(0, len(sigmas))]
        if u < 0.40:
            x0 = base + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif u < 0.75:
            # worst-point surgery: pick k worst by max dot
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            wmax = G.max(axis=1)
            k = int(rng.integers(1, 5))
            idx = np.argsort(-wmax)[:k]
            # also sometimes include high-degree hub
            if rng.random() < 0.5:
                thr = np.sort(G.ravel())[-600] if 600 < G.size else best_v - 1e-9
                deg = (G > best_v - 1e-9).sum(axis=1)
                hub = int(np.argmax(deg))
                idx[0] = hub
            x0 = base.copy()
            for kk in idx:
                nc = 1500
                cand = rng.standard_normal(size=(nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                others = np.delete(x0, kk, axis=0)
                md = (cand @ others.T).max(axis=1)
                x0[kk] = cand[int(np.argmin(md))]
            # small jitter on the rest to break symmetry
            x0 = x0 + rng.standard_normal(size=(n, d)) * (0.3 * sigma)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # block shake: perturb random subset strongly
            x0 = base.copy()
            k = int(rng.integers(5, 20))
            idx = rng.choice(n, size=k, replace=False)
            x0[idx] += rng.standard_normal(size=(k, d)) * (sigma + 0.15)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        # annealed smooth params
        a_choice = rng.random()
        if a_choice < 0.3:
            alpha = float(rng.uniform(20, 45)); steps = 500; lr = 0.06
        elif a_choice < 0.65:
            alpha = float(rng.uniform(45, 90)); steps = 600; lr = 0.04
        else:
            alpha = float(rng.uniform(90, 160)); steps = 500; lr = 0.025
        beta = 0.9
        # adapt steps to remaining time
        if time.time() - t0 > EXPLORE_UNTIL - 10:
            steps = min(steps, 300)
        cur = _smooth_run(x0, alpha, steps, lr, beta, rng)
        # second short sharpening
        cur = _smooth_run(cur, alpha * 1.5 + 10, 200, lr * 0.7, beta, rng)
        v = _max_cos(cur)
        if v < best_v:
            best_v = v
            best_x = cur.copy()
        # occasional opportunistic LP if very promising (within 1e-3)
        if it % 25 == 0 and (time.time() - t0) < EXPLORE_UNTIL - 5:
            if v < best_v + 0.002:
                try:
                    lp = _lp_polish(cur, 2.0)
                    vlp = _max_cos(lp)
                    if vlp < best_v:
                        best_v = vlp
                        best_x = lp.copy()
                except Exception:
                    pass
    # final polish
    remain = BUDGET - (time.time() - t0)
    remain = max(5.0, min(LP_TIME + 5, remain))
    try:
        best_x = _lp_polish(best_x, remain)
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

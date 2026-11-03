import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

np.random.seed(0)

def max_cos_fn(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(x, seconds):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    def mc(a):
        G = a @ a.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    best = mc(x)
    radius = 3e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[sel]; J = iu1[sel]
        p = I.shape[0]
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        v = mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_anneal(x, steps=1200, T0=0.025, T1=0.004, lr=0.008):
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best_x = x.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    for t in range(1, steps + 1):
        frac = t / steps
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        G = x @ x.T
        g = G[iu0, iu1]
        mg = float(g.max())
        if mg < best_val:
            best_val = mg
            best_x = x.copy()
        e = np.exp((g - mg) / T)
        s = float(e.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        W = np.zeros((n, n), dtype=x.dtype)
        W[iu0, iu1] = e / s
        W = W + W.T
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * gt
        v1 = b2 * v1 + (1.0 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m1 / bc1
        vhat = v1 / bc2
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - delta
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        x = x / nr
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mg = float(G.max())
    if mg < best_val:
        return x
    return best_x

def solve():
    t00 = time.time()
    BUDGET = 170.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal(size=(131, 9))
    if xb.shape != (131, 9):
        # try to fix shape if transposed or different
        try:
            xb = np.reshape(xb, (131, 9))
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal(size=(131, 9))
    xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    best = max_cos_fn(xb)
    # initial short polish to ensure local optimum
    xb = lp_polish(xb, seconds=4.0)
    best = max_cos_fn(xb)
    rng = np.random.default_rng(0)
    n, d = xb.shape
    sigmas = [0.01, 0.022, 0.045, 0.008, 0.03]
    it = 0
    while time.time() - t00 < BUDGET - 14.0:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        do_kick = (it % 6 == 0)
        Xt = xb.copy()
        if do_kick:
            G = xb @ xb.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            thr = m - 5e-7
            cnt = np.sum(G > thr, axis=1)
            order = np.argsort(-cnt)
            pool = order[:12]
            kick_idx = rng.choice(pool, size=2, replace=False)
            Xt = Xt + (sigma * 0.5) * rng.standard_normal(size=Xt.shape)
            rnd = rng.standard_normal(size=(2, d))
            rnd = rnd / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
            Xt[kick_idx] = rnd
        else:
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
        Xt = Xt / np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        Xt = smooth_anneal(Xt, steps=1200, T0=0.025, T1=0.004, lr=0.008)
        if time.time() - t00 >= BUDGET - 14.0:
            v = max_cos_fn(Xt)
            if v < best:
                xb = Xt; best = v
            break
        Xt = lp_polish(Xt, seconds=2.0)
        v = max_cos_fn(Xt)
        if v < best - 1e-12:
            xb = Xt; best = v
    rem = BUDGET - (time.time() - t00) - 1.0
    if rem > 1.0:
        rem = min(rem, 12.0)
        xb = lp_polish(xb, seconds=float(rem))
    xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    return xb

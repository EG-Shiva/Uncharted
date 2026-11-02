import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for linprog
        if time.time() - start > seconds - 0.05:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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

def smooth_gd(x, beta, steps, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # weights, stable
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        F = W @ x
        F -= x * np.sum(F * x, axis=1, keepdims=True)
        rs = W.sum(axis=1, keepdims=True)
        F = F / (rs + 1.0)
        alpha = lr0 + (lr1 - lr0) * (s / max(steps - 1, 1))
        x = x - alpha * F
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    np.random.seed(0)
    t0 = time.time()
    HARD = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if xb.shape != (58, 22):
        # fallback pad/truncate deterministically
        rng0 = np.random.RandomState(0)
        tmp = rng0.randn(58, 22)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m = min(n, 58)
        tmp[:m, :min(d, 22)] = xb[:m, :min(d, 22)]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    best = max_cos(xb)
    cur = xb.copy()
    cur_val = best
    rng = np.random.RandomState(20261102)
    sigmas = [0.015, 0.04, 0.10, 0.30]
    trial = 0
    # exploration phase, reserve ~35s for final polish
    while time.time() - t0 < 125.0:
        if time.time() - t0 > HARD - 40:
            break
        trial += 1
        sigma = sigmas[trial % 4]
        y = cur + sigma * rng.randn(n, d)
        if trial % 4 == 3:
            k = 2 + (trial // 4) % 4  # 2..5 points fully re-inserted
            idx = rng.choice(n, k, replace=False)
            y[idx] = rng.randn(k, d)
        elif trial % 4 == 1 and trial % 8 == 1:
            # single worst point re-insertion
            G = cur @ cur.T
            np.fill_diagonal(G, -2)
            w = G.max(axis=1)
            wi = int(np.argmax(w))
            y[wi] = rng.randn(d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed repulsion
        y = smooth_gd(y, beta=70.0, steps=350, lr0=0.030, lr1=0.006)
        if time.time() - t0 > HARD - 40:
            break
        y = smooth_gd(y, beta=180.0, steps=350, lr0=0.020, lr1=0.004)
        if trial % 2 == 0:
            y = smooth_gd(y, beta=350.0, steps=250, lr0=0.010, lr1=0.002)
        v = max_cos(y)
        if v > cur_val + 0.04:
            # too bad, occasionally restart from best
            if trial % 30 == 0:
                cur = xb.copy()
                cur_val = best
            continue
        remain = HARD - (time.time() - t0) - 32.0
        if remain < 1.0:
            break
        sec = min(3.5, remain * 0.15 + 1.0)
        y2 = lp_polish(y, seconds=sec)
        v2 = max_cos(y2)
        if v2 < best - 1e-12:
            best = v2
            xb = y2.copy()
            cur = y2.copy()
            cur_val = v2
        else:
            # simulated-annealing walk to change basin
            T = 0.0025
            if v2 < cur_val or rng.rand() < np.exp((cur_val - v2) / T):
                cur = y2.copy()
                cur_val = v2
            # periodic return to best to intensify
            if trial % 40 == 0:
                cur = xb.copy()
                cur_val = best
    # final long polish of best
    rem = HARD - (time.time() - t0) - 1.0
    if rem > 2.0:
        xb = lp_polish(xb, seconds=min(rem, 38.0))
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    return xb

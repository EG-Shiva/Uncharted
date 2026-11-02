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
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_descent(x0, steps, T0, T1, lr0, mu=0.85):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.7 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        v = mu * v + (1.0 - mu) * grad
        x -= lr * v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def find_hole(x_fixed, rng, n_samples=20000):
    Y = rng.randn(n_samples, 4)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    return Y[idx], float(vals[idx])

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(0)
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(163, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (163, 4):
        bx = rng.randn(163, 4)
        bx /= np.linalg.norm(bx, axis=1, keepdims=True)
    bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
    best_x = bx.copy()
    best_val = max_cos(best_x)
    n = 163
    sigmas = [0.03, 0.08, 0.18, 0.01, 0.05, 0.12]
    trial = 0
    # search phase ~ up to 125s, reserve rest for final polish
    while time.time() - t0 < 122.0:
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        if trial % 13 == 12:
            # full random restart
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur, v = smooth_descent(cur, steps=8000, T0=0.02, T1=0.0012, lr0=0.035)
            if v < best_val:
                # short LP confirm
                if time.time() - t0 < 150:
                    pol = lp_polish(cur, seconds=3.0)
                    vp = max_cos(pol)
                    if vp < best_val:
                        best_x, best_val = pol, vp
                else:
                    if v < best_val:
                        best_x, best_val = cur, v
            trial += 1
            continue
        cur = best_x.copy()
        if mode == 0:
            cur = cur + rng.randn(*cur.shape) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 1:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            deg = (G > m - 0.002).sum(axis=1)
            idx = np.argsort(deg)[-8:]
            noise = rng.randn(*cur.shape) * (sigma * 0.2)
            noise[idx] += rng.randn(len(idx), 4) * (sigma * 2.0)
            cur = cur + noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 2:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            deg = (G > m - 0.002).sum(axis=1)
            k = 2 + (trial % 2)
            worst = np.argsort(deg)[-k:]
            keep = np.delete(np.arange(n), worst)
            xf = cur[keep].copy()
            for w in worst:
                h, _ = find_hole(xf, rng, n_samples=20000)
                h = h + rng.randn(4) * 0.01
                h /= np.linalg.norm(h)
                cur[w] = h
                xf = np.vstack([xf, h[None, :]])
            cur = cur + rng.randn(*cur.shape) * (sigma * 0.15)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 3:
            k = 20
            idx = rng.choice(n, k, replace=False)
            cur[idx] = cur[idx] + rng.randn(k, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        else:
            # single-point hole reinsertion of rattler-like point
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            deg = (G > m - 0.001).sum(axis=1)
            # pick lowest-degree point (rattler) if exists else random
            w = int(np.argmin(deg))
            keep = np.delete(np.arange(n), w)
            xf = cur[keep]
            h, _ = find_hole(xf, rng, n_samples=25000)
            h = h + rng.randn(4) * 0.005
            h /= np.linalg.norm(h)
            cur[w] = h
            # small global shake
            cur = cur + rng.randn(*cur.shape) * (sigma * 0.3)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        # anneal
        cur_r, v = smooth_descent(cur, steps=4500, T0=0.015, T1=0.0015, lr0=0.030)
        if v < best_val:
            # confirm with short LP if time allows
            if time.time() - t0 < 155:
                pol = lp_polish(cur_r, seconds=2.5)
                vp = max_cos(pol)
                if vp < best_val:
                    best_x, best_val = pol, vp
            else:
                best_x, best_val = cur_r, v
        trial += 1
        if time.time() - t0 > BUDGET - 30:
            break
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 4.0:
        remain = min(remain, 40.0)
        best_x = lp_polish(best_x, seconds=remain)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

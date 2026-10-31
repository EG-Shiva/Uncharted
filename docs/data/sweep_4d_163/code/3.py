import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline - 0.5:
            break
        c = (x @ x.T)[iu]
        m = c.max()
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

def repel(x, steps, lr, temp, margin, rng):
    n = x.shape[0]
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        thr = cmax - margin
        M = G > thr
        # weights, symmetric
        W = np.exp((G - cmax) / temp) * M
        scale = W.sum() * 0.5 + 1e-12
        G2 = W * G
        g = W @ x - (G2.sum(axis=1, keepdims=True) * x)
        g /= scale
        cur_lr = lr * (1.0 - 0.7 * s / max(steps, 1))
        x = x - cur_lr * g
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def largest_voids(x_keep, m, rng, S=20000):
    y = rng.standard_normal(size=(S, 4))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    # max dot to kept set: want minimal
    scores = (y @ x_keep.T).max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= m:
            break
        cand = y[idx]
        ok = True
        for q in picked:
            if float(cand @ q) > 0.85:
                ok = False
                break
        if ok:
            picked.append(cand)
        # if too strict, relax after half scan
    # fallback fill
    k = 0
    while len(picked) < m:
        picked.append(y[order[k]])
        k += 1
    return np.array(picked)

def solve():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal(size=(163, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (163, 4):
        # try to fix / random
        tmp = rng.standard_normal(size=(163, 4))
        m = min(xb.shape[0], 163) if xb.ndim == 2 else 0
        if m > 0:
            tmp[:m, :min(4, xb.shape[1])] = xb[:m, :min(4, xb.shape[1])]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur = best
    n, d = 163, 4
    it = 0
    # reserve ~18s for final polish
    while time.time() < DEADLINE - 20.0:
        it += 1
        # choose base: mostly best, sometimes cur for wandering
        if rng.random() < 0.7 or it < 3:
            base = x_best
            base_val = best
        else:
            base = x_cur
            base_val = cur
        r = rng.random()
        x_try = base.copy()
        if r < 0.38:
            # global shake, log-uniform sigma
            sigma = 10.0 ** rng.uniform(-3.2, -1.2)
            x_try = base + sigma * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.70:
            # patch shake around jammed or random center
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            # near-max degree to find jammed
            thr = base_val - 0.01
            deg = (G > thr).sum(axis=1)
            if rng.random() < 0.6:
                # jammed center
                center = int(np.argmax(deg + rng.random(n) * 0.5))
            else:
                center = int(rng.integers(n))
            k = int(rng.integers(8, 26))
            dots = base @ base[center]
            idx = np.argsort(-dots)[:k]
            sigma = float(rng.uniform(0.02, 0.09))
            x_try[idx] = base[idx] + sigma * rng.standard_normal(size=(k, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # rattler relocation to voids
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = base_val - 0.015
            deg = (G > thr).sum(axis=1)
            m = int(rng.integers(2, 5))
            reloc = np.argsort(deg + rng.random(n) * 0.3)[:m]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[reloc] = False
            voids = largest_voids(base[keep_mask], m, rng, S=20000)
            # small jitter on voids
            voids = voids + 0.005 * rng.standard_normal(size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            x_try[reloc] = voids
            # also slight shake of neighbours to open space
            if rng.random() < 0.5:
                sigma = 0.01
                x_try[keep_mask] = base[keep_mask] + sigma * rng.standard_normal(size=(n - m, d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # repel annealing with random hypers
        steps = int(rng.integers(600, 1800))
        lr = float(rng.uniform(0.04, 0.14))
        temp = float(10.0 ** rng.uniform(-2.6, -2.0))  # ~0.0025..0.01
        margin = float(rng.uniform(0.02, 0.06))
        # quick time check
        if time.time() > DEADLINE - 22.0:
            break
        x_try = repel(x_try, steps, lr, temp, margin, rng)
        if time.time() > DEADLINE - 20.5:
            # evaluate without polish
            v = max_cos(x_try / np.linalg.norm(x_try, axis=1, keepdims=True))
        else:
            # short LP polish 1.2-2.2s
            sec = float(rng.uniform(1.2, 2.4))
            # cap to remaining
            sec = min(sec, max(0.6, DEADLINE - 20.0 - time.time()))
            x_try = lp_polish(x_try, sec, deadline=DEADLINE - 19.0)
            v = max_cos(x_try)
        if v < best - 1e-12:
            x_best = x_try.copy()
            best = v
            x_cur = x_try.copy()
            cur = v
        elif v < cur + 0.0015 and rng.random() < 0.45:
            x_cur = x_try.copy()
            cur = v
        # opportunistic: if stuck long, reheat cur to best
        if it % 12 == 0 and cur > best + 0.002:
            x_cur = x_best.copy()
            cur = best
    # final long polish of best
    remain = DEADLINE - time.time() - 1.0
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 18.0), deadline=DEADLINE)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

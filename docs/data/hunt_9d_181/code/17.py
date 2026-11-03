import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_anneal(x, steps, a0, a1, lr, rng=None):
    n, d = x.shape
    for k in range(steps):
        frac = k / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        W = np.exp(alpha * (G - gm))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nm = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
        step = lr * (1.0 - 0.4 * frac) / nm
        x = x - step * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def find_void(x, drop, rng, n_cand=3000):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[drop] = False
    Xo = x[mask]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xo.T
    vals = dots.max(axis=1)
    return cand[int(np.argmin(vals))]

def lp_polish(x, seconds, init_radius=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 6e-4:
            break
        Gfull = x @ x.T
        ca = Gfull[iu]
        m = ca.max()
        sel = ca > m - max(20 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            kth = np.partition(ca, -2800)[-2800]
            sel = ca >= kth
            if int(sel.sum()) > 3200:
                idx = np.where(sel)[0]
                order = np.argsort(-ca[idx])[:3200]
                m2 = np.zeros_like(ca, dtype=bool)
                m2[idx[order]] = True
                sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        csel = ca[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    FINAL = 28.0
    rng = np.random.default_rng(0)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        a = rng.normal(size=(181, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    scores = [max_cos(a) for a in cand]
    xb = cand[int(np.argmin(scores))].copy()
    vb = min(scores)
    cur = xb.copy(); cur_v = vb
    best_x = xb.copy(); best_v = vb
    n, d = xb.shape
    it = 0
    T = 4e-04
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < FINAL + 2.0:
            break
        it += 1
        y = cur.copy()
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        vmax = G.max()
        worst = G.max(axis=1)
        mode = it % 8
        if mode == 0:
            y = y + rng.normal(size=y.shape) * 0.006
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_anneal(y, steps=300, a0=60, a1=250, lr=0.015)
        elif mode == 1:
            probs = np.maximum(worst - worst.min() + 0.002, 0)
            probs /= probs.sum()
            k = int(rng.integers(4, 10))
            idx = rng.choice(n, size=k, replace=False, p=probs)
            y[idx] += rng.normal(size=(k, d)) * rng.uniform(0.05, 0.13)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_anneal(y, steps=380, a0=80, a1=300, lr=0.018)
        elif mode == 2:
            j = int(np.argmax(worst))
            try:
                y[j] = find_void(y, j, rng, n_cand=3000)
            except Exception:
                y[j] = rng.normal(size=d); y[j] /= np.linalg.norm(y[j])
            y = smooth_anneal(y, steps=350, a0=70, a1=280, lr=0.016)
        elif mode == 3:
            y = y + rng.normal(size=y.shape) * 0.025
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_anneal(y, steps=300, a0=50, a1=200, lr=0.02)
        elif mode == 4:
            probs = np.maximum(worst - worst.min() + 0.002, 0)
            probs /= probs.sum()
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False, p=probs)
            y[idx] += rng.normal(size=(k, d)) * 0.08
            # teleport one rattler-like point to void
            j2 = int(np.argmin(worst + rng.random(n) * 0.005))
            try:
                y[j2] = find_void(y, j2, rng, n_cand=2500)
            except Exception:
                pass
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_anneal(y, steps=420, a0=90, a1=350, lr=0.014)
        elif mode == 5:
            y = y + rng.normal(size=y.shape) * 0.010
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_anneal(y, steps=250, a0=40, a1=150, lr=0.022)
        elif mode == 6:
            k = int(rng.integers(8, 18))
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += rng.normal(size=(k, d)) * 0.04
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_anneal(y, steps=380, a0=80, a1=320, lr=0.016)
        else:
            j = int(rng.integers(0, n))
            try:
                y[j] = find_void(y, j, rng, n_cand=3000)
            except Exception:
                y[j] = rng.normal(size=d); y[j] /= np.linalg.norm(y[j])
            y = smooth_anneal(y, steps=300, a0=60, a1=250, lr=0.015)
        v = max_cos(y)
        if v < best_v:
            best_v = v; best_x = y.copy()
            cur = y.copy(); cur_v = v
        elif v < cur_v:
            cur = y.copy(); cur_v = v
        else:
            if v < cur_v + T and rng.random() < 0.25:
                cur = y.copy(); cur_v = v
            # occasional restart to best
            if rng.random() < 0.05:
                cur = best_x.copy(); cur_v = best_v
                cur = cur + rng.normal(size=cur.shape) * 0.004
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        # decay temperature slightly
        T *= 0.9995
        T = max(T, 8e-05)
    rem = BUDGET - (time.time() - t0)
    ft = max(2.0, min(FINAL, rem - 0.5))
    if ft > 2.0:
        try:
            best_x = lp_polish(best_x, seconds=ft)
            best_v = max_cos(best_x)
        except Exception:
            pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

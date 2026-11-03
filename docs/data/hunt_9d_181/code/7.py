import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def contact_counts(x, thresh=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = np.sum(g > m - thresh, axis=1)
    return m, cnt, g

def lp_polish(x, seconds, init_radius=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        thresh = max(20 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            kth = np.partition(c_all, -2800)[-2800]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 3200:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3200]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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

def smooth_power(x, iters, p_start, p_end, lr=0.03):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p_start + (p_end - p_start) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Gp = np.where(G > 0, G, 0.0)
        m = Gp.max()
        if m <= 1e-12:
            break
        W = (Gp / (m + 1e-18)) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        nm = np.mean(np.linalg.norm(grad_t, axis=1)) + 1e-12
        step = lr * (1.0 - 0.5 * frac) / nm
        x = x - step * grad_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
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
    bi = int(np.argmin(scores))
    xb = cand[bi].copy()
    bb = scores[bi]
    n, d = xb.shape
    # current basin
    xc = xb.copy(); bc = bb
    # temperature for basin hopping
    T = 3e-05
    it = 0
    # precompute triu for hole search reuse
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 4.0:
            break
        it += 1
        y = xc.copy()
        mode = it % 6
        _, cnt, Gcur = contact_counts(y)
        order_hi = np.argsort(-cnt)
        if mode == 0:
            # global small shake + smooth
            y = y + rng.normal(size=y.shape) * 0.010
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_power(y, iters=250, p_start=25, p_end=70, lr=0.03)
            lp_time = min(1.6, max(0.6, rem - 3.0))
            y = lp_polish(y, seconds=lp_time, init_radius=0.008)
        elif mode == 1 or mode == 3:
            # surgery: remove 1-2 most overloaded points, reinsert in largest hole
            kremove = 1 if mode == 1 else 2
            victims = order_hi[:kremove]
            # hole search: sample random directions, maximize min-distance to remaining
            K = 4000
            R = rng.normal(size=(K, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            mask = np.ones(n, dtype=bool); mask[victims] = False
            Xrem = y[mask]
            dots = R @ Xrem.T
            worst = dots.max(axis=1)
            pick = np.argsort(worst)[:kremove]
            for k, v in enumerate(victims):
                y[v] = R[pick[k]]
            # small jitter others
            y = y + rng.normal(size=y.shape) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_power(y, iters=300, p_start=30, p_end=90, lr=0.035)
            lp_time = min(1.8, max(0.8, rem - 3.0))
            y = lp_polish(y, seconds=lp_time, init_radius=0.010)
        elif mode == 2:
            # attack worst pair directly
            G = y @ y.T; np.fill_diagonal(G, -2.0)
            ij = np.argmax(G); i, j = divmod(ij, n)
            # move both apart along difference + noise
            for v in (i, j):
                y[v] = y[v] + rng.normal(size=d) * 0.06
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_power(y, iters=250, p_start=35, p_end=100, lr=0.03)
            lp_time = min(1.6, max(0.6, rem - 3.0))
            y = lp_polish(y, seconds=lp_time, init_radius=0.008)
        elif mode == 4:
            # cluster shake of top-8 overloaded
            victims = order_hi[:8]
            y[victims] += rng.normal(size=(8, d)) * 0.035
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_power(y, iters=250, p_start=25, p_end=80, lr=0.03)
            lp_time = min(1.6, max(0.6, rem - 3.0))
            y = lp_polish(y, seconds=lp_time, init_radius=0.008)
        else:
            # random subset shake (basin hop)
            K = 20
            idx = rng.choice(n, size=K, replace=False)
            y[idx] += rng.normal(size=(K, d)) * 0.05
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_power(y, iters=200, p_start=20, p_end=60, lr=0.025)
            lp_time = min(1.4, max(0.6, rem - 3.0))
            y = lp_polish(y, seconds=lp_time, init_radius=0.006)
        v = max_cos(y)
        if v < bc - 1e-12:
            xc, bc = y, v
            if v < bb - 1e-12:
                xb, bb = y.copy(), v
            T *= 0.98
        else:
            # Metropolis accept to escape
            if rng.random() < np.exp(-(v - bc) / T):
                xc, bc = y, v
            # slowly cool / reheat
            T *= 0.999
            if T < 8e-06:
                T = 3e-05
            # restart from best occasionally if stuck
            if it % 25 == 0 and bc > bb + 5e-05:
                xc, bc = xb.copy(), bb
    # final long polish of best
    rem = BUDGET - (time.time() - t0)
    if rem > 1.0:
        xb = lp_polish(xb, seconds=min(rem - 0.5, 12.0), init_radius=2e-3)
        bb = max_cos(xb)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb

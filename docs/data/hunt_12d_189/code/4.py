import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, r_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for remaining
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        # cap constraints for speed
        p_sel = int(sel.sum())
        if p_sel == 0:
            radius *= 0.5
            continue
        if p_sel > 3500:
            # keep top 3500
            idx = np.argpartition(-c_all, 3500)[:3500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            # ensure we keep only those above thr? just take top
            sel = mask
            p_sel = 3500
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = p_sel
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def smooth_push(x, steps=500, lr0=0.02, lr1=0.002, d0=0.05, d1=0.004, rng=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 * (1 - frac) + lr1 * frac
        delta = d0 * (1 - frac) + d1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        Mask = G > (m - delta)
        grad = Mask.astype(np.float64) @ x
        # tangent projection
        dot = (grad * x).sum(axis=1, keepdims=True)
        tang = grad - dot * x
        tn = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        # scale hubs less aggressively by counts
        tang = tang / tn
        x = x - lr * tang
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def contacts_of(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return cnt, m, G

def deepest_holes(x, n_samples, rng):
    n, d = x.shape
    R = rng.standard_normal((n_samples, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ x.T
    vals = D.max(axis=1)
    order = np.argsort(vals)
    return R[order[:16]], vals[order[:16]]

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    # load start
    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (189, 12):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        r = rng.standard_normal((189, 12))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        x_best = r
        best_val = max_cos(x_best)
    # initial short polish to be at local optimum
    if BUDGET - (time.time() - t0) > 20:
        x_best = lp_polish(x_best, seconds=6.0, r_init=1e-3)
        best_val = max_cos(x_best)
    it = 0
    # main basin loop, reserve ~18s for final polish
    while time.time() - t0 < BUDGET - 18.0:
        it += 1
        remain = BUDGET - 18.0 - (time.time() - t0)
        if remain < 3.0:
            break
        secs = 1.6 if remain > 20 else 1.0
        rtype = rng.random()
        y = x_best.copy()
        n, d = y.shape
        if rtype < 0.40:
            sigma = float(rng.choice([0.004, 0.008, 0.015, 0.030, 0.050]))
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_push(y, steps=420, lr0=0.015, lr1=0.002, d0=0.04, d1=0.004)
            y = lp_polish(y, seconds=secs, r_init=1e-3)
        elif rtype < 0.72:
            # targeted kick of rattlers + hubs
            cnt, m, G = contacts_of(y, tol=1e-7)
            ratt = np.where(cnt == 0)[0]
            order = np.argsort(-cnt)
            hubs = order[:6]
            k = int(rng.integers(4, 13))
            cand_pool = list(ratt) + list(hubs) + list(rng.integers(0, n, size=k))
            # dedupe, take k
            idx = []
            for a in cand_pool:
                a = int(a)
                if a not in idx:
                    idx.append(a)
                if len(idx) >= k:
                    break
            while len(idx) < k:
                a = int(rng.integers(0, n))
                if a not in idx:
                    idx.append(a)
            idx = np.array(idx, dtype=int)
            sigma = float(rng.choice([0.08, 0.12, 0.20, 0.30]))
            y[idx] = y[idx] + sigma * rng.standard_normal((len(idx), d))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            # small global jitter
            y = y + 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_push(y, steps=500, lr0=0.018, lr1=0.002, d0=0.05, d1=0.004)
            y = lp_polish(y, seconds=secs, r_init=1.5e-3)
        else:
            # hole refill: replace worst points by deep holes
            cnt, m, G = contacts_of(y, tol=1e-7)
            order = np.argsort(-cnt)  # worst first
            holes, hvals = deepest_holes(y, 8000, rng)
            nr = int(rng.integers(2, 5))
            # pick nr points to replace: prefer hubs, else random
            repl = order[:nr].copy()
            # if rattlers exist, replace them instead of hubs half the time
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0 and rng.random() < 0.5:
                repl[:min(nr, len(ratt))] = ratt[:min(nr, len(ratt))]
            for k2 in range(nr):
                y[int(repl[k2])] = holes[k2 % len(holes)] + 0.005 * rng.standard_normal(d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_push(y, steps=450, lr0=0.012, lr1=0.002, d0=0.03, d1=0.003)
            y = lp_polish(y, seconds=secs, r_init=1e-3)
        try:
            v = max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
        # opportunistic: if time is running, break
        if time.time() - t0 > BUDGET - 19.0:
            break
    # final long polish
    frem = BUDGET - (time.time() - t0) - 1.0
    if frem > 2.0:
        x_best = lp_polish(x_best, seconds=min(frem, 15.0), r_init=8e-4)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, max_pairs=650):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and radius > 1e-13 and itc < 25:
        itc += 1
        G = x @ x.T
        c_full = G[iu]
        m = float(c_full.max())
        thr = m - max(20 * radius, 2e-9)
        sel = c_full > thr
        I = iu[0][sel]; J = iu[1][sel]
        # cap to tightest pairs to keep LP fast
        if len(I) > max_pairs:
            c_sel = c_full[sel]
            kp = np.argpartition(-c_sel, max_pairs)[:max_pairs]
            I = I[kp]; J = J[kp]
            bvals = -c_sel[kp]
        else:
            bvals = -c_full[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=bvals, A_eq=A_eq,
                          b_eq=np.zeros(n),
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def smooth_opt(x, steps=550, a0=25.0, a1=220.0, eta0=0.05):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        eta = eta0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W @ x
        wg = (W * G).sum(axis=1, keepdims=True)
        Tang = S - wg * x
        wsum = W.sum(axis=1, keepdims=True) + 1e-12
        step = Tang / wsum
        x = x - eta * step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
    return x

def deepest_hole(Xr, rng, K=8000):
    d = Xr.shape[1]
    best_c = None; best_v = 1e9
    # batch to bound memory
    for _ in range(2):
        cand = rng.normal(size=(K // 2, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = Xr @ cand.T
        cmax = dots.max(axis=0)
        i = int(np.argmin(cmax))
        if cmax[i] < best_v:
            best_v = float(cmax[i]); best_c = cand[i]
    return best_c

def solve():
    t0 = time.time()
    BUDGET = 165.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (189, 12) and np.all(np.isfinite(c)):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if not cands:
        rng0 = np.random.default_rng(0)
        c = rng0.normal(size=(189, 12))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = cands[0]; best_v = max_cos(best_x)
    for c in cands[1:]:
        v = max_cos(c)
        if v < best_v:
            best_x, best_v = c.copy(), v
    best_x = best_x.copy()
    rng = np.random.default_rng(7)
    n, d = best_x.shape
    it = 0
    # exploration loop; reserve ~22s for final polish
    while time.time() - t0 < BUDGET - 22:
        if it > 300:
            break
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        m_best = float(G.max())
        per_max = G.max(axis=1)
        deg = (G > m_best - 0.01).sum(axis=1)
        rattlers = np.where(per_max < m_best - 1e-3)[0]
        overload = np.argsort(deg)[-6:]
        worst = np.argsort(per_max)[-3:]
        y = best_x.copy()
        mode = it % 5
        if mode == 4:
            # hole re-insertion of worst points
            for wi in worst[:2]:
                mask = np.arange(n) != wi
                Xr = y[mask]
                hc = deepest_hole(Xr, rng, K=8000)
                y[wi] = hc
            y = y + rng.normal(scale=0.008, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_opt(y, steps=600, a0=35.0, a1=260.0, eta0=0.05)
        elif mode == 3:
            sigma = 0.09
            noise = rng.normal(scale=sigma, size=y.shape)
            if len(rattlers):
                noise[rattlers] *= 2.0
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_opt(y, steps=500, a0=20.0, a1=200.0, eta0=0.06)
        else:
            sigmas = [0.025, 0.05, 0.015]
            sigma = sigmas[it % len(sigmas)]
            noise = rng.normal(scale=sigma, size=y.shape)
            if len(rattlers):
                noise[rattlers] *= 2.5
            if len(overload):
                noise[overload] *= 1.8
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            a1 = 200.0 + (it % 3) * 40.0
            eta0 = 0.04 + (it % 2) * 0.015
            y = smooth_opt(y, steps=550, a0=25.0, a1=a1, eta0=eta0)
        v = max_cos(y)
        # selective LP: only polish promising basins
        if v < best_v + 0.004:
            remain = BUDGET - 22 - (time.time() - t0)
            if remain < 1.2:
                # no time for LP; accept cheap improvement only
                if v < best_v - 1e-12:
                    best_x, best_v = y, v
                it += 1
                continue
            y2 = lp_polish(y, seconds=min(2.2, remain))
            v2 = max_cos(y2)
            if v2 < best_v - 1e-12 and np.all(np.isfinite(y2)):
                best_x, best_v = y2, v2
        else:
            # far basin, accept only if outright better (rare)
            if v < best_v - 1e-12:
                best_x, best_v = y, v
        it += 1
    remain = BUDGET - (time.time() - t0)
    if remain > 4:
        best_x = lp_polish(best_x, seconds=min(remain - 1.5, 20.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

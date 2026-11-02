import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, init_radius=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time left check
        if time.time() - start > seconds:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        # cap size for speed
        if np.count_nonzero(sel) > 2500:
            # keep top 2500
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[-2500:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        c = c_all[sel]
        I = iu[0][sel]
        J = iu[1][sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # accept move if it improves linearized but not true? no, shrink
            # also allow lateral move to x if close? keep x=y if it improves m slightly?
            # check current max at y vs at x
            radius *= 0.5
            # if stuck, keep best x
            if v < max_cos(x) + 1e-12:
                x = y
    return best_x

def smooth_descent(x, steps=350, T0=0.012, T1=0.0025, lr=0.6, momentum=0.85):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    vel = np.zeros_like(x)
    for s in range(steps):
        a = s / max(steps - 1, 1)
        T = T0 * (1 - a) + T1 * a
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W.sum() / 2.0
        if not np.isfinite(S) or S < 1e-14:
            break
        Gr = (W @ x) / S
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        # clip extreme rows for stability
        gn = np.linalg.norm(Gt, axis=1, keepdims=True)
        # avoid huge steps
        scale = np.ones_like(gn)
        big = gn > 0.5
        scale[big] = 0.5 / (gn[big] + 1e-12)
        Gt = Gt * scale
        vel = momentum * vel - lr * Gt
        x = x + vel
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
        if not np.all(np.isfinite(x)):
            break
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def solve():
    t_start = time.time()
    BUDGET = 165.0
    rng = np.random.default_rng(12345)
    # load starting point
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (58, 22):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = max_cos(v)
                    if c < best_val:
                        best_val = c
                        best = v.copy()
        except Exception:
            pass
    if best is None:
        y = rng.normal(size=(58, 22))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        best = y
        best_val = max_cos(best)
    # quick initial LP to be at optimum (short)
    if BUDGET - (time.time() - t_start) > 10:
        best = lp_polish(best, seconds=3.0, init_radius=1e-3)
        best_val = max_cos(best)

    sigmas = [0.08, 0.03, 0.15, 0.05, 0.02, 0.10, 0.04, 0.06]
    it = 0
    # basin hopping
    while True:
        elapsed = time.time() - t_start
        # reserve at least 20s for final polish
        if elapsed > BUDGET - 20:
            break
        if it > 300:
            break
        sigma = sigmas[it % len(sigmas)]
        n, d = best.shape
        # contact counts for targeting
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        counts = np.sum(G > m - 0.002, axis=1)
        cand = best.copy()
        mode = it % 10
        if mode == 9:
            # full re-insert 2-3 points: pick worst contacts + random
            k = 2 + (it % 2)
            idx = np.argsort(counts)[-k:]
            # random choice also
            # replace with uniform random
            cand[idx] = rng.normal(size=(k, d))
            # small shake others
            cand += 0.01 * rng.normal(size=(n, d))
            steps = 600
        elif mode % 3 == 2:
            k = 4 + (it % 5)  # 4..8
            idx = np.argsort(counts)[-k:]
            cand[idx] += (sigma * 2.0) * rng.normal(size=(k, d))
            cand += (sigma * 0.25) * rng.normal(size=(n, d))
            steps = 400
        else:
            cand = cand + sigma * rng.normal(size=(n, d))
            steps = 350
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # smooth travel
        cand = smooth_descent(cand, steps=steps, T0=0.012, T1=0.0025, lr=0.6, momentum=0.85)
        # short LP
        left = BUDGET - 20 - (time.time() - t_start)
        if left < 0.8:
            break
        secs = min(1.6, left)
        cand = lp_polish(cand, seconds=secs, init_radius=0.005)
        v = max_cos(cand)
        if v < best_val - 1e-12 and np.all(np.isfinite(cand)):
            best = cand
            best_val = v
        it += 1

    # final long polish of best
    rem = BUDGET - (time.time() - t_start)
    if rem > 2:
        # alternate smooth fine + LP
        # one fine smooth with tiny T to squeeze
        try:
            tmp = smooth_descent(best, steps=200, T0=0.004, T1=0.0012, lr=0.3, momentum=0.8)
            tmp = lp_polish(tmp, seconds=min(2.0, rem / 3.0), init_radius=1e-3)
            if max_cos(tmp) < best_val:
                best = tmp
                best_val = max_cos(tmp)
        except Exception:
            pass
        rem2 = BUDGET - (time.time() - t_start)
        if rem2 > 1:
            best = lp_polish(best, seconds=rem2, init_radius=1e-3)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

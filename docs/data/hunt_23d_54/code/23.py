import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(x, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    vel = np.zeros_like(x)
    mu = 0.85
    for s in range(steps):
        f = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** f if a0 > 0 else a1
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True)
        Gn = Gt / (deg + 1.0)
        vel = mu * vel + Gn
        x = x - lr * vel
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _multi_hole(xb, idx, rng, n_cand=4000):
    n, d = xb.shape
    x = xb.copy()
    keep = np.ones(n, dtype=bool)
    keep[idx] = False
    fixed = xb[keep]
    # greedy farthest insertion among random candidates
    chosen = []
    for k, j in enumerate(idx):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ fixed.T
        smax = S.max(axis=1)
        if len(chosen) > 0:
            Ch = np.stack(chosen, axis=0)
            S2 = C @ Ch.T
            s2max = S2.max(axis=1)
            smax = np.maximum(smax, s2max)
        bi = int(np.argmin(smax))
        chosen.append(C[bi])
        x[j] = C[bi]
        fixed = np.vstack([fixed, C[bi][None, :]])
    return x

def _lp_polish(x, seconds, t_dead):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and time.time() < t_dead:
        if radius < 1e-13:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        tol = max(20 * radius, 2e-9)
        sel = c_all > m - tol
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 550:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:550]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
        I = iu0[0][sel]; J = iu0[1][sel]; c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        vals = np.empty((p, 2 * d + 1))
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            xb = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return xb

def solve():
    t_start = time.time()
    HARD = 178.0
    FINAL = 20.0
    rng = np.random.default_rng(7)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        r = min(xb.size, tmp.size)
        tmp.ravel()[:r] = xb.ravel()[:r]
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()
    t_dead = t_start + HARD - 1.0
    t_explore_end = t_start + HARD - FINAL - 1.0
    it = 0
    while time.time() < t_explore_end:
        it += 1
        u = rng.random()
        y = x_best.copy()
        if u < 0.35:
            sigma = float(rng.uniform(0.015, 0.30))
            y = x_best + sigma * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif u < 0.60:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            cnt = (G > best - 0.015).sum(axis=1)
            # mix of worst-max and low/high contacts
            ksub = int(rng.integers(2, 5))
            if rng.random() < 0.5:
                worst = np.argsort(-per)[:ksub]
            else:
                # target loose + overloaded
                order = np.argsort(cnt)
                worst = np.concatenate([order[:ksub//2+1], np.argsort(-cnt)[:ksub//2+1]])[:ksub]
                worst = np.unique(worst)[:ksub]
            y = _multi_hole(x_best, worst, rng)
        elif u < 0.80:
            ksub = int(rng.integers(1, 5))
            idx = rng.choice(n, size=ksub, replace=False)
            R = rng.standard_normal((ksub, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[idx] = R
        elif u < 0.92:
            # block rotation kick: random subspace jitter of a cluster
            ksub = int(rng.integers(4, 10))
            idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(rng.uniform(0.05, 0.25))
            y[idx] = y[idx] + sigma * rng.standard_normal((ksub, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # fresh random restart (rare)
            y = rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        steps = int(rng.integers(250, 750))
        a0 = float(rng.uniform(30, 100))
        a1 = float(rng.uniform(250, 700))
        lr0 = float(rng.uniform(0.004, 0.010))
        lr1 = float(rng.uniform(0.0005, 0.003))
        # ensure lr1 < lr0 for annealing
        if lr1 > lr0:
            lr0, lr1 = lr1, lr0
        y2 = _smooth(y, steps, a0, a1, lr0, lr1)
        v2 = _max_cos(y2)
        if v2 > best + 0.012:
            continue
        # promising: short LP
        remain_explore = t_explore_end - time.time()
        if remain_explore < 1.0:
            # keep raw if it is actually better (rare without LP)
            if v2 < best:
                best = v2
                x_best = y2
            break
        sec = 1.2 if v2 > best else 2.0
        sec = min(sec, remain_explore)
        y3 = _lp_polish(y2, sec, t_dead)
        v3 = _max_cos(y3)
        if v3 < best - 1e-12:
            best = v3
            x_best = y3.copy()
        if time.time() >= t_explore_end:
            break
    rem = t_dead - time.time()
    if rem > 2.0:
        rem = min(rem, FINAL)
        x_best = _lp_polish(x_best, rem, t_dead)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

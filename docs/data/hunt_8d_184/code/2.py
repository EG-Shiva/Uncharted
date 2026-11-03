import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _smooth_refine(x, steps=350, a0=40.0, a1=320.0, lr0=0.03, lr1=0.003):
    n, d = x.shape
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = np.max(G)
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        direction = gt / S
        x = x - lr * direction
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy_refine(x, steps=150, m0=0.008, m1=0.0008, lr=0.008):
    n, d = x.shape
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        thr = m - margin
        M = (G > thr)
        # gradient from thresholded neighbours
        grad = M.astype(np.float64) @ x
        deg = M.sum(axis=1, keepdims=True).astype(np.float64)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        direction = gt / np.maximum(deg, 1.0)
        # decay lr slightly
        x = x - lr * direction
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _refine_hole(x_fixed, y, steps=250, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = x_fixed @ y
        mx = np.max(s)
        w = np.exp(alpha * (s - mx))
        sw = np.sum(w) + 1e-30
        g = w @ x_fixed
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def _best_hole_sample(x_fixed, rng, n_sample=8000):
    d = x_fixed.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    S = Y @ x_fixed.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx], float(m[idx])

def _lp_polish(x, seconds, t_start=None, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    if t_start is None:
        t_start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    # ensure shape (184,8)
    if xb.shape != (184, 8):
        # try to reshape / truncate
        xb = xb.reshape(184, 8)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    gbest = xb.copy()
    gval = best_val

    NTRIAL = 110
    for trial in range(NTRIAL):
        if time.time() > DEADLINE - 60:
            break
        # time guard for hopping phase (leave 55s for LP)
        if time.time() - t0 > 115:
            break
        mode = trial % 5
        xt = gbest.copy()
        if mode == 0:
            # small global jitter
            xt = xt + rng.standard_normal(size=xt.shape) * 0.010
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif mode == 1:
            # shake worst 6
            pm = _per_max(xt)
            worst = np.argsort(-pm)[:6]
            xt[worst] += rng.standard_normal(size=(6, 8)) * 0.07
            xt[worst] /= np.linalg.norm(xt[worst], axis=1, keepdims=True)
        elif mode == 2:
            # shake random 10 medium
            idx = rng.choice(184, size=10, replace=False)
            xt[idx] += rng.standard_normal(size=(10, 8)) * 0.12
            xt[idx] /= np.linalg.norm(xt[idx], axis=1, keepdims=True)
        elif mode == 3:
            # relocate 2 lowest to largest holes
            pm = _per_max(xt)
            lowest = np.argsort(pm)[:2]
            keep = [i for i in range(184) if i not in set(lowest.tolist())]
            xfix = xt[keep].copy()
            for li in lowest:
                y0, _ = _best_hole_sample(xfix, rng, n_sample=8000)
                y0 = _refine_hole(xfix, y0, steps=200)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            # tiny jitter others to break symmetry
            xt[keep] += rng.standard_normal(size=(len(keep), 8)) * 0.003
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        else:
            # block shake around random center: 14 nearest
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            c = int(rng.integers(0, 184))
            nearest = np.argsort(-G[c])[:14]
            xt[nearest] += rng.standard_normal(size=(14, 8)) * 0.055
            xt[nearest] /= np.linalg.norm(xt[nearest], axis=1, keepdims=True)

        # annealed repulsion + greedy
        # vary steps slightly by mode for diversity
        if mode == 3:
            xt = _smooth_refine(xt, steps=380, a0=40.0, a1=320.0, lr0=0.032, lr1=0.003)
            xt = _greedy_refine(xt, steps=170, m0=0.008, m1=0.0007, lr=0.008)
        else:
            xt = _smooth_refine(xt, steps=320, a0=45.0, a1=300.0, lr0=0.028, lr1=0.0025)
            xt = _greedy_refine(xt, steps=130, m0=0.007, m1=0.0008, lr=0.007)
        v = _max_cos(xt)
        if v < gval:
            gval = float(v)
            gbest = xt.copy()

    # final polish with remaining time
    remain = DEADLINE - time.time() - 1.0
    if remain > 8:
        lp_t = min(remain, 55.0)
        gbest = _lp_polish(gbest, lp_t, t_start=time.time(), deadline=DEADLINE)
    gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(gbest)):
        gbest = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return gbest

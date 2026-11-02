import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        sel = G > (m - w)
        W = np.zeros_like(G)
        # stable exp: G-m in [-w,0]
        W[sel] = np.exp(k * (G[sel] - m))
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = (F * x).sum(axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt((Ft * Ft).sum() / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt((v * v).sum() / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _best_void(x_keep, rng, n_try=3000):
    d = x_keep.shape[1]
    C = rng.standard_normal((n_try, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ x_keep.T
    m = S.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        c = c_all[sel]
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_blocks = np.empty((p, 2 * d + 1), dtype=int)
        cols_blocks[:, :d] = I[:, None] * d + np.arange(d)
        cols_blocks[:, d:2*d] = J[:, None] * d + np.arange(d)
        cols_blocks[:, 2*d] = n * d
        cols = cols_blocks.ravel()
        vals_blocks = np.empty((p, 2 * d + 1))
        vals_blocks[:, :d] = x[J]
        vals_blocks[:, d:2*d] = x[I]
        vals_blocks[:, 2*d] = -1.0
        vals = vals_blocks.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 5.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 175.0
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
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # also consider record
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    it = 0
    # hopping phase, reserve 22s for final LP
    while time.time() - t0 < BUDGET - 22.0:
        it += 1
        remain = BUDGET - 22.0 - (time.time() - t0)
        if remain < 1.5:
            break
        steps = 500
        if remain < 15:
            steps = 300
        mode = it % 6
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        ppm = G.max(axis=1)
        cnt = (G > best - 0.015).sum(axis=1)
        order_worst = np.argsort(-ppm)
        order_lo = np.argsort(cnt)
        x = x_best.copy()
        if mode == 0:
            sig = float(rng.uniform(0.006, 0.014))
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=80, k1=2500, lr0=0.012, lr1=0.0007, w0=0.05, w1=0.005, mu=0.85)
        elif mode == 1:
            sig = float(rng.uniform(0.015, 0.035))
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=50, k1=1800, lr0=0.016, lr1=0.001, w0=0.07, w1=0.007, mu=0.8)
        elif mode == 2:
            ksel = int(rng.integers(1, 4))
            idx = order_worst[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _best_void(x[mask], rng, n_try=2500)
                x[j] = nv + rng.standard_normal(d) * 0.01
                x[j] /= np.linalg.norm(x[j])
            x = _anneal(x, steps=steps, k0=70, k1=3000, lr0=0.014, lr1=0.0007, w0=0.06, w1=0.004, mu=0.85)
        elif mode == 3:
            ksel = int(rng.integers(2, 5))
            idx = order_lo[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _best_void(x[mask], rng, n_try=2500)
                x[j] = nv + rng.standard_normal(d) * 0.015
                x[j] /= np.linalg.norm(x[j])
            x = _anneal(x, steps=steps, k0=60, k1=2200, lr0=0.015, lr1=0.0009, w0=0.06, w1=0.005, mu=0.82)
        elif mode == 4:
            # collective jitter of most-jammed cluster + anneal
            ksel = int(rng.integers(4, 9))
            idx = order_worst[:ksel]
            x[idx] += rng.standard_normal((ksel, d)) * float(rng.uniform(0.02, 0.06))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=40, k1=1500, lr0=0.018, lr1=0.0012, w0=0.08, w1=0.008, mu=0.8)
        else:
            # full random subset reinsertion
            ksel = int(rng.integers(2, 5))
            idx = rng.choice(n, size=ksel, replace=False)
            R = rng.standard_normal((ksel, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            x[idx] = R
            x = _anneal(x, steps=steps, k0=60, k1=2000, lr0=0.015, lr1=0.001, w0=0.07, w1=0.006, mu=0.85)
        v = _max_cos(x)
        if v < best - 1e-12:
            best = v
            x_best = x.copy()
        # opportunistic extra tighten of new best with short high-k anneal
        if v < best + 0.002 and (it % 10 == 0):
            try:
                xt = _anneal(x.copy(), steps=200, k0=500, k1=6000, lr0=0.004, lr1=0.0003, w0=0.02, w1=0.003, mu=0.9)
                vt = _max_cos(xt)
                if vt < best - 1e-12:
                    best = vt
                    x_best = xt.copy()
            except Exception:
                pass
    # final tighten + LP
    try:
        x_best = _anneal(x_best, steps=400, k0=200, k1=8000, lr0=0.003, lr1=0.0002, w0=0.015, w1=0.003, mu=0.9)
        best = _max_cos(x_best)
    except Exception:
        pass
    rem = BUDGET - (time.time() - t0) - 1.0
    if rem > 3:
        rem = min(rem, 20.0)
        try:
            x_best = _lp_polish(x_best, rem)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

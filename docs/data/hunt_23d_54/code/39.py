import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        W = np.zeros_like(G)
        sel = D > -w
        np.fill_diagonal(sel, False)
        if np.any(sel):
            W[sel] = np.exp(k * D[sel])
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _best_void(x, keep_mask, rng, n_try=2000):
    kept = x[keep_mask]
    C = rng.randn(n_try, x.shape[1])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(x, seconds, t0=None, overall_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and overall_deadline is not None and time.time() > overall_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if int(np.count_nonzero(sel)) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:900]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if v < best + 1e-12:
                x = y
    return xb

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()
    iu = np.triu_indices(n, 1)

    it = 0
    # hopping phase: reserve ~30s for final LP
    while time.time() < deadline - 32.0 and it < 60:
        it += 1
        if deadline - 32.0 - (time.time()) < 1.0:
            break
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        ppm = G.max(axis=1)
        wide = G > (best - 0.015)
        cnt = wide.sum(axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-ppm)
        mode = it % 7
        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.007
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=450, k0=120, k1=3000, lr0=0.010, lr1=0.0007, w0=0.04, w1=0.004)
        elif mode == 1:
            x += rng.randn(n, d) * 0.018
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=450, k0=60, k1=2000, lr0=0.013, lr1=0.0009, w0=0.06, w1=0.006)
        elif mode == 2:
            ksel = 2 + (it % 3)
            idx = order_worst[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _best_void(x, mask, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.015
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=500, k0=80, k1=2800, lr0=0.012, lr1=0.0007, w0=0.05, w1=0.005)
        elif mode == 3:
            ksel = 3 + (it % 3)
            idx = order_lo[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _best_void(x, mask, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.02
                x[j] /= np.linalg.norm(x[j])
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=500, k0=70, k1=2500, lr0=0.012, lr1=0.0008, w0=0.05, w1=0.005)
        elif mode == 4:
            # collective jitter of jammed cluster: pick high-contact point + neighbors
            j0 = int(order_worst[0])
            gg = x_best @ x_best[j0]
            gg[j0] = -2
            nbr = np.argsort(-gg)[:6]
            x[nbr] += rng.randn(len(nbr), d) * 0.03
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=500, k0=60, k1=2200, lr0=0.014, lr1=0.0008, w0=0.055, w1=0.005)
        elif mode == 5:
            x += rng.randn(n, d) * 0.011
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=600, k0=200, k1=4000, lr0=0.006, lr1=0.0004, w0=0.025, w1=0.0035)
        else:
            # single-point void reinsert of worst
            j = int(order_worst[(it // 7) % n])
            mask = np.ones(n, dtype=bool); mask[j] = False
            nv = _best_void(x, mask, rng, n_try=2500)
            x[j] = nv
            x += rng.randn(n, d) * 0.002
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=500, k0=100, k1=3000, lr0=0.010, lr1=0.0006, w0=0.04, w1=0.004)
        v = _max_cos(x)
        if v < best:
            best = v
            x_best = x.copy()
    # final polish: short anneal sharpening then LP
    try:
        if time.time() < deadline - 28.0:
            xs = _anneal(x_best.copy(), steps=300, k0=400, k1=6000, lr0=0.003, lr1=0.0002, w0=0.012, w1=0.0025)
            vs = _max_cos(xs)
            if vs < best:
                best, x_best = vs, xs
    except Exception:
        pass
    remain = deadline - time.time() - 2.0
    if remain > 5.0:
        secs = min(remain, 28.0)
        try:
            xp = _lp_polish(x_best.copy(), seconds=secs, t0=t0, overall_deadline=deadline - 1.0)
            vp = _max_cos(xp)
            if vp < best:
                best, x_best = vp, xp
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

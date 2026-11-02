import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t0=None, budget=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        it += 1
        if it > 60:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:900]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # keep x at bx, shrink
            radius *= 0.5
            # if stuck, stay at best
            x = bx.copy()
            if radius < 5e-4 and v > best + 1e-12:
                # small random tangential nudge to escape LP stall
                break
    return bx


def _smooth(x_init, steps, k0, k1, lr0, lr1, d0, d1):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        # max over off-diag
        m = float(G[iu].max())
        thr = m - d1 if d1 > d0 else m - (d0 + (d1 - d0) * frac)
        # weight matrix
        # W = exp(k*(G-m)) for G>thr, 0 else, zero diag
        W = np.zeros_like(G)
        mask = G > thr
        np.fill_diagonal(mask, False)
        if np.any(mask):
            W[mask] = np.exp(k * (G[mask] - m))
            Gr = W @ x
            dot = np.sum(Gr * x, axis=1, keepdims=True)
            Gt = Gr - dot * x
            deg = np.sum(W, axis=1, keepdims=True) + 1.0
            x = x - lr * Gt / deg
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
    return x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()
    iu = np.triu_indices(n, 1)

    it = 0
    # reserve final polish
    while time.time() - t0 < BUDGET - 10.0:
        it += 1
        remain = BUDGET - 10.0 - (time.time() - t0)
        if remain < 2.0:
            break
        mode = it % 8
        # contact analysis on x_best
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        wide = Gb > best - 0.02
        cnt = wide.sum(axis=1)
        order_lo = np.argsort(cnt)
        order_hi = np.argsort(-cnt)
        # worst max per point
        per_max = Gb.max(axis=1)
        order_worst = np.argsort(-per_max)

        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.008
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=600, k0=150, k1=1200, lr0=0.02, lr1=0.005, d0=0.05, d1=0.008)
        elif mode == 1:
            x += rng.randn(n, d) * 0.022
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=900, k0=200, k1=2000, lr0=0.03, lr1=0.004, d0=0.05, d1=0.006)
        elif mode == 2:
            ksel = 5
            idx = order_lo[:ksel]
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.15
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=900, k0=200, k1=1800, lr0=0.03, lr1=0.004, d0=0.06, d1=0.007)
        elif mode == 3:
            ksel = 3
            idx = order_hi[:ksel]
            x += rng.randn(n, d) * 0.004
            x[idx] += rng.randn(ksel, d) * 0.10
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=800, k0=200, k1=1500, lr0=0.025, lr1=0.004, d0=0.05, d1=0.007)
        elif mode == 4:
            # hole re-insert worst point
            idx = order_worst[0]
            v = rng.randn(d)
            v /= np.linalg.norm(v)
            x[idx] = v
            # small shake neighbours
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=1000, k0=250, k1=2200, lr0=0.03, lr1=0.004, d0=0.06, d1=0.006)
        elif mode == 5:
            ksel = 2
            idx = np.concatenate([order_worst[:1], order_lo[:1]])
            for j in idx:
                v = rng.randn(d)
                v /= np.linalg.norm(v)
                x[j] = v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=1100, k0=250, k1=2200, lr0=0.03, lr1=0.004, d0=0.07, d1=0.006)
        elif mode == 6:
            x += rng.randn(n, d) * 0.05
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=1000, k0=150, k1=1500, lr0=0.035, lr1=0.005, d0=0.08, d1=0.008)
        else:
            # focused: shake low-contact points mildly + global tiny
            ksel = 7
            idx = order_lo[:ksel]
            x += rng.randn(n, d) * 0.006
            x[idx] += rng.randn(ksel, d) * 0.05
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, steps=700, k0=180, k1=1600, lr0=0.02, lr1=0.004, d0=0.04, d1=0.006)

        v = _max_cos(x)
        # polish only if promising to save time
        thresh = best + 0.012
        if v < thresh:
            # short LP, time-aware
            lp_time = 1.6 if remain > 20 else 0.9
            xp = _lp_polish(x, seconds=lp_time, t0=t0, budget=BUDGET - 9.0)
            vp = _max_cos(xp)
            if vp < best - 1e-12:
                best = vp
                x_best = xp.copy()
                continue
            # even if not better than best, if smooth alone beat best, keep it
            if v < best - 1e-12:
                # polish it a bit more next round; store
                best = v
                x_best = x.copy()
                # quick extra polish if very good
                if v < vp:
                    pass
        else:
            if v < best - 1e-12:
                best = v
                x_best = x.copy()

    # final polish of best
    final_time = max(2.0, BUDGET - (time.time() - t0) - 1.0)
    final_time = min(final_time, 25.0)
    if final_time > 0.5:
        x_best = _lp_polish(x_best, seconds=final_time, t0=t0, budget=174.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

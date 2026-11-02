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
    radius = 2e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(25 * radius, 2e-9)
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
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x_init, steps, k0, k1, lr0, lr1, d0, d1):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        delta = d0 + (d1 - d0) * frac
        G = x @ x.T
        m = float(G[iu].max())
        # stable exp weights only for near-max pairs
        # use full matrix but thresholded
        thr = m - delta
        # compute E = exp(k*(G-m)) where G>thr
        E = np.zeros_like(G)
        mask = G > thr
        np.fill_diagonal(mask, False)
        if np.any(mask):
            E[mask] = np.exp(k * (G[mask] - m))
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        x = x - lr * Gt / deg
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _best_hole(x, j, rng, ncand=6000):
    d = x.shape[1]
    cand = rng.randn(ncand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T
    dots[:, j] = -10.0
    md = dots.max(axis=1)
    bj = int(np.argmin(md))
    return cand[bj]

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(11)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(54, 23)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 54, 23
    if xb.shape != (54, 23):
        tmp = rng.randn(54, 23)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    best = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
        if vr < best:
            best, x_best = vr, xr.copy()
    except Exception:
        pass

    iu = np.triu_indices(n, 1)
    it = 0
    # reserve final polish
    while time.time() - t0 < BUDGET - 14.0:
        it += 1
        remaining = BUDGET - 14.0 - (time.time() - t0)
        if remaining < 3.0:
            break
        mode = it % 8
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        wide = (Gb > best - 0.015)
        cnt = wide.sum(axis=1)
        order_lo = np.argsort(cnt)
        order_hi = np.argsort(-cnt)

        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.010
        elif mode == 1:
            x += rng.randn(n, d) * 0.030
        elif mode == 2:
            # shake jammed cluster hard
            ksel = 5
            idx = order_hi[:ksel]
            x += rng.randn(n, d) * 0.006
            x[idx] += rng.randn(ksel, d) * 0.10
        elif mode == 3:
            # shake loose valley
            ksel = 7
            idx = order_lo[:ksel]
            x += rng.randn(n, d) * 0.006
            x[idx] += rng.randn(ksel, d) * 0.12
        elif mode == 4:
            # block resample of loose valley + jitter
            ksel = 6
            idx = order_lo[:ksel].copy()
            # add 2 random to diversify
            extra = rng.choice(n, size=2, replace=False)
            idx = np.unique(np.concatenate([idx, extra]))[:8]
            for j in idx:
                # randomize then hole-pick against current x
                x[j] = rng.randn(d)
                xn = np.linalg.norm(x[j])
                if xn == 0:
                    xn = 1.0
                x[j] /= xn
            x += rng.randn(n, d) * 0.005
        elif mode == 5:
            # replace 1-2 worst by holes
            nj = 2 if rng.rand() < 0.5 else 1
            # pick from most jammed
            pool = order_hi[:10]
            idx = pool[rng.choice(len(pool), size=nj, replace=False)]
            for j in idx:
                x[j] = _best_hole(x, int(j), rng, ncand=6000)
            x += rng.randn(n, d) * 0.004
        elif mode == 6:
            # replace 3 points sequentially by holes
            idx = rng.choice(n, size=3, replace=False)
            # bias one to loose
            if rng.rand() < 0.7:
                idx[0] = int(order_lo[rng.randint(0, 10)])
            for j in idx:
                x[int(j)] = _best_hole(x, int(j), rng, ncand=5000)
            x += rng.randn(n, d) * 0.003
        else:
            # large global kick + subspace twist: random orthogonal-ish rotation of subset
            x += rng.randn(n, d) * 0.05
            # random direction flip of a few points
            ksel = 3
            idx = rng.choice(n, size=ksel, replace=False)
            v = rng.randn(d); v /= np.linalg.norm(v)
            for j in idx:
                # reflect component along v partially
                proj = float(x[j] @ v)
                x[j] = x[j] - 1.5 * proj * v + rng.randn(d) * 0.05

        x /= np.linalg.norm(x, axis=1, keepdims=True)

        if mode in (0, 1):
            steps, k0, k1, lr0, lr1, d0, d1 = 500, 30.0, 600.0, 0.30, 0.02, 0.10, 0.008
        elif mode in (2, 3, 7):
            steps, k0, k1, lr0, lr1, d0, d1 = 650, 20.0, 500.0, 0.35, 0.02, 0.12, 0.010
        else:
            steps, k0, k1, lr0, lr1, d0, d1 = 750, 25.0, 700.0, 0.40, 0.02, 0.12, 0.008

        try:
            xs = _smooth(x, steps, k0, k1, lr0, lr1, d0, d1)
        except Exception:
            continue
        vs = _max_cos(xs)
        if vs > best + 0.020:
            continue
        # selective LP polish: always polish if promising or periodically
        do_lp = (vs < best + 0.002) or (it % 4 == 0 and vs < best + 0.008)
        if not do_lp:
            # accept without polish if outright better (rare)
            if vs < best - 1e-12:
                # quick snap
                try:
                    xp = _lp_polish(xs, 1.5, t0, BUDGET - 14.0)
                except Exception:
                    xp = xs
                vp = _max_cos(xp)
                if vp < best - 1e-12:
                    best = vp
                    x_best = xp.copy()
            continue
        lp_sec = 2.0 if remaining > 25 else 1.2
        if it % 12 == 0:
            lp_sec = 3.0
        try:
            xp = _lp_polish(xs, lp_sec, t0, BUDGET - 14.0)
        except Exception:
            continue
        vp = _max_cos(xp)
        if vp < best - 1e-12:
            best = vp
            x_best = xp.copy()

    try:
        left = BUDGET - (time.time() - t0)
        if left > 2:
            x_best = _lp_polish(x_best, min(left - 0.5, 13.0), t0, BUDGET)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(25.0 * radius, 2e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
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
            obj = np.zeros(nv); obj[-1] = 1.0
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

def _lp_kick(x, radius, window, keep_frac, rng):
    # one trust-region LP on random subset of tight pairs -> topology change
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    Gfull = x @ x.T
    c_all = Gfull[iu0]
    m = c_all.max()
    sel = c_all > m - window
    idx = np.where(sel)[0]
    if len(idx) > 900:
        order = np.argsort(c_all[idx])[::-1][:900]
        idx = idx[order]
    if len(idx) < 10:
        return None
    keep = rng.rand(len(idx)) < keep_frac
    if keep.sum() < 10:
        return None
    idx = idx[keep]
    I, J = iu0[0][idx], iu0[1][idx]
    c = c_all[idx]
    p = len(I)
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
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                      b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                      method="highs")
    except Exception:
        return None
    if res.status != 0:
        return None
    y = x + res.x[:-1].reshape(n, d)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _anneal(x_init, steps, k0, k1, lr0, lr1, win0, win1, mu=0.85):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        win = win0 + (win1 - win0) * frac
        G = x @ x.T
        m = float(G[iu].max())
        thr = m - win
        # weights
        W = np.zeros_like(G)
        mask = G > thr
        np.fill_diagonal(mask, False)
        if np.any(mask):
            # stable exp
            W[mask] = np.exp(k * (G[mask] - m))
        # symmetrize already symmetric
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(W, axis=1, keepdims=True) + 1.0
        Gt = Gt / deg
        vel = mu * vel + (1.0 - mu) * Gt
        # bias correction-ish scale
        x = x - lr * vel / (1.0 - mu ** (s + 2))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _best_hole(x, j, rng, ncand=3000):
    d = x.shape[1]
    cand = rng.randn(ncand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T
    dots[:, j] = -10.0
    md = dots.max(axis=1)
    return cand[int(np.argmin(md))]

def solve():
    t0 = time.time()
    BUDGET = 171.0
    T_END = t0 + BUDGET
    rng = np.random.RandomState(20261102)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(54, 23)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 54, 23
    if xb.shape != (54, 23):
        xb = rng.randn(54, 23)
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
    it = 0
    # quick contact stats helper inline
    while time.time() < T_END - 11.0:
        it += 1
        rem = T_END - 11.0 - (time.time() - t0)
        if rem < 2.5:
            break
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        cnt_wide = (Gb > best - 0.02).sum(axis=1)
        order_lo = np.argsort(cnt_wide)
        order_hi = np.argsort(-cnt_wide)
        mode = it % 7
        x = x_best.copy()
        kick_done = False
        if mode == 0:
            # LP dropout hop small-medium
            r = 0.04 + 0.04 * rng.rand()
            w = 0.015 + 0.02 * rng.rand()
            kf = 0.5 + 0.3 * rng.rand()
            y = _lp_kick(x, r, w, kf, rng)
            if y is not None:
                x = y
                kick_done = True
            else:
                x += rng.randn(n, d) * 0.010
        elif mode == 1:
            # coherent cluster shift of jammed neighborhood
            seed = int(order_hi[rng.randint(0, 10)])
            Grow = (x_best @ x_best.T)[seed].copy()
            Grow[seed] = -10
            nn = np.argsort(-Grow)[:6 + int(rng.randint(0, 3))]
            common = rng.randn(d) * 0.035
            # project common to tangent approx per point later via renorm
            x[nn] += common * 0.6 + rng.randn(len(nn), d) * 0.012
            x += rng.randn(n, d) * 0.004
        elif mode == 2:
            # loose hole re-insertion (1-2 pts)
            if rng.rand() < 0.6:
                js = order_lo[:10][rng.choice(10, size=2, replace=False)]
            else:
                js = rng.choice(n, size=2, replace=False)
            for j in js:
                x[int(j)] = _best_hole(x, int(j), rng, ncand=3000)
            x += rng.randn(n, d) * 0.003
        elif mode == 3:
            # single hole + medium jitter
            j = int(order_lo[rng.randint(0, 8)]) if rng.rand() < 0.7 else int(rng.randint(0, n))
            x[j] = _best_hole(x, j, rng, ncand=4000)
            x += rng.randn(n, d) * 0.004
        elif mode == 4:
            x += rng.randn(n, d) * 0.020
        elif mode == 5:
            # larger LP hop
            r = 0.07 + 0.05 * rng.rand()
            w = 0.02 + 0.03 * rng.rand()
            kf = 0.4 + 0.3 * rng.rand()
            y = _lp_kick(x, r, w, kf, rng)
            if y is not None:
                x = y
                kick_done = True
            else:
                ksel = 5
                idx = order_lo[:ksel]
                x[idx] += rng.randn(ksel, d) * 0.06
                x += rng.randn(n, d) * 0.005
        else:
            x += rng.randn(n, d) * 0.008
            ksel = 4
            idx = order_hi[:ksel]
            x[idx] += rng.randn(ksel, d) * 0.03
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # anneal params depend on kick or not
        if kick_done:
            steps, k0, k1, lr0, lr1, w0, w1 = 650, 25.0, 600.0, 0.35, 0.02, 0.12, 0.008
        elif mode in (2, 3):
            steps, k0, k1, lr0, lr1, w0, w1 = 750, 20.0, 500.0, 0.32, 0.02, 0.12, 0.008
        elif mode == 4:
            steps, k0, k1, lr0, lr1, w0, w1 = 550, 30.0, 500.0, 0.25, 0.02, 0.10, 0.008
        else:
            steps, k0, k1, lr0, lr1, w0, w1 = 600, 25.0, 550.0, 0.30, 0.02, 0.11, 0.008
        # adapt steps to remaining time
        if rem < 12:
            steps = min(steps, 300)
        try:
            xs = _anneal(x, steps, k0, k1, lr0, lr1, w0, w1)
        except Exception:
            continue
        vs = _max_cos(xs)
        if vs > best + 0.022:
            continue
        # selective LP polish
        lp_sec = 1.2
        if it % 9 == 0:
            lp_sec = 2.2
        if rem < 8:
            lp_sec = 0.8
        try:
            xp = _lp_polish(xs, lp_sec, T_END - 10.0)
        except Exception:
            continue
        vp = _max_cos(xp)
        if vp < best - 1e-12:
            best = vp
            x_best = xp.copy()
    # final polish
    try:
        left = T_END - (time.time() - t0)
        if left > 2:
            x_best = _lp_polish(x_best, min(left - 0.6, 10.0), T_END)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

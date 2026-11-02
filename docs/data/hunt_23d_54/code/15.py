import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t0=None, deadline=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t0 is not None and (time.time() - t0) > deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x_init, steps, k0, k1, lr0, lr1, d0, d1, mu=0.85):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        delta = d0 + (d1 - d0) * frac
        G = x @ x.T
        m = float(G[iu].max())
        thr = m - delta
        mask = G > thr
        np.fill_diagonal(mask, False)
        E = np.zeros_like(G)
        if np.any(mask):
            E[mask] = np.exp(k * (G[mask] - m))
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        grad = Gt / deg
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _best_hole(x, j, cand):
    dots = cand @ x.T
    dots[:, j] = -10.0
    md = dots.max(axis=1)
    return int(np.argmin(md))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(54, 23)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 54, 23
    if xb.shape != (54, 23):
        xb = rng.randn(54, 23)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
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
    it = 0
    while time.time() - t0 < BUDGET - 11.0:
        it += 1
        remaining = BUDGET - 11.0 - (time.time() - t0)
        if remaining < 2.5:
            break
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        pw = Gb.max(axis=1)
        cnt = (Gb > best - 0.015).sum(axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-pw)
        # PCA peripheral score
        try:
            xc = x_best - x_best.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(xc, full_matrices=False)
            proj = xc @ Vt[:2].T
            rad = np.linalg.norm(proj, axis=1)
            order_peri = np.argsort(-rad)
        except Exception:
            order_peri = order_lo
        mode = it % 7
        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.009
        elif mode == 1:
            # jammed-block kick around low-contact seed + neighbors
            s = int(order_lo[(it // 7) % 10])
            g = Gb[s].copy()
            g[s] = -10
            nn = np.argsort(-g)[:7]
            blk = np.concatenate([[s], nn])
            x += rng.randn(n, d) * 0.004
            x[blk] += rng.randn(len(blk), d) * 0.09
        elif mode == 2:
            # worst-point hole reinsertion
            j = int(order_worst[0] if rng.rand() < 0.6 else order_lo[rng.randint(0, 8)])
            cand = rng.randn(7000, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            bj = _best_hole(x, j, cand)
            x[j] = cand[bj]
            x += rng.randn(n, d) * 0.003
        elif mode == 3:
            # two-hole reinsertion (one worst, one peripheral/loose)
            j1 = int(order_worst[rng.randint(0, 4)])
            j2 = int(order_peri[rng.randint(0, 10)])
            if j2 == j1:
                j2 = int(rng.randint(0, n))
            for j in (j1, j2):
                cand = rng.randn(5000, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                bj = _best_hole(x, j, cand)
                x[j] = cand[bj]
            x += rng.randn(n, d) * 0.004
        elif mode == 4:
            # peripheral cluster kick
            ksel = 6
            idx = order_peri[:ksel]
            # rotate seed choice
            shift = (it // 7) % 3
            if shift:
                idx = np.roll(idx, shift)
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.075
        elif mode == 5:
            x += rng.randn(n, d) * 0.020
        else:
            # single loose hole + global jitter
            j = int(order_lo[rng.randint(0, 6)])
            cand = rng.randn(6000, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            bj = _best_hole(x, j, cand)
            x[j] = cand[bj]
            x += rng.randn(n, d) * 0.005
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if mode in (0, 5):
            steps, k0, k1, lr0, lr1, d0, d1 = 450, 30.0, 500.0, 0.5, 0.04, 0.10, 0.008
        elif mode in (1, 4):
            steps, k0, k1, lr0, lr1, d0, d1 = 650, 18.0, 450.0, 0.6, 0.04, 0.13, 0.010
        else:
            steps, k0, k1, lr0, lr1, d0, d1 = 600, 25.0, 650.0, 0.65, 0.04, 0.12, 0.008
        try:
            xs = _smooth(x, steps, k0, k1, lr0, lr1, d0, d1)
        except Exception:
            continue
        vs = _max_cos(xs)
        if vs > best + 0.020:
            continue
        lp_sec = 1.6 if remaining > 25 else 1.0
        if it % 9 == 0:
            lp_sec = 3.0
        try:
            xp = _lp_polish(xs, lp_sec, t0, BUDGET - 11.0)
        except Exception:
            continue
        vp = _max_cos(xp)
        if vp < best - 1e-12:
            best = vp
            x_best = xp.copy()
    try:
        left = BUDGET - (time.time() - t0)
        if left > 2:
            x_best = _lp_polish(x_best, min(left - 0.5, 10.0), t0, BUDGET)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

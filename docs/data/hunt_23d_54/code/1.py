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
    radius = 1e-3
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
        sel = c_all > m - max(20 * radius, 1e-9)
        # cap constraints to keep LP fast
        if np.count_nonzero(sel) > 800:
            # keep tightest 800
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
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
        thr = m - delta
        mask = G > thr
        np.fill_diagonal(mask, False)
        E = np.zeros_like(G)
        # exp in stable range
        sel = mask
        if np.any(sel):
            E[sel] = np.exp(k * (G[sel] - m))
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        x = x - lr * Gt / deg
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(54, 23)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 54, 23
    if xb.shape != (54, 23):
        # fallback resize by random
        tmp = rng.randn(54, 23)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    best = _max_cos(xb)
    x_best = xb.copy()
    # consider record file too
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
    # reserve ~12s for final polish
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        remaining = BUDGET - 12.0 - (time.time() - t0)
        if remaining < 3.0:
            break
        mode = it % 6
        # contact analysis on current best
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        # wide window to rank looseness
        wide = (Gb > best - 0.02)
        cnt = wide.sum(axis=1)
        order_lo = np.argsort(cnt)  # loose first
        order_hi = np.argsort(-cnt)  # jammed first

        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.008
        elif mode == 1:
            x += rng.randn(n, d) * 0.022
        elif mode == 2:
            ksel = 6
            idx = order_lo[:ksel]
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.08
        elif mode == 3:
            ksel = 4
            idx = order_hi[:ksel]
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.07
        elif mode == 4:
            # replace 2 points by best-of-random holes
            nrep = 2
            idx = rng.choice(n, size=nrep, replace=False)
            for j in idx:
                cand = rng.randn(4000, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ x.T
                dots[:, j] = -10.0  # ignore self
                # also ignore? use max over others
                md = dots.max(axis=1)
                bj = np.argmin(md)
                x[j] = cand[bj]
            x += rng.randn(n, d) * 0.004
        else:
            # replace 1 point (often a loose one)
            if rng.rand() < 0.7:
                j = int(order_lo[rng.randint(0, 8)])
            else:
                j = int(rng.randint(0, n))
            cand = rng.randn(5000, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ x.T
            dots[:, j] = -10.0
            md = dots.max(axis=1)
            bj = np.argmin(md)
            x[j] = cand[bj]
            x += rng.randn(n, d) * 0.003

        x /= np.linalg.norm(x, axis=1, keepdims=True)

        # smooth params vary by mode
        if mode in (0, 1):
            steps, k0, k1, lr0, lr1, d0, d1 = 700, 30.0, 500.0, 0.25, 0.02, 0.10, 0.008
        elif mode in (2, 3):
            steps, k0, k1, lr0, lr1, d0, d1 = 900, 20.0, 400.0, 0.30, 0.02, 0.12, 0.010
        else:
            steps, k0, k1, lr0, lr1, d0, d1 = 1000, 25.0, 600.0, 0.35, 0.02, 0.12, 0.008

        try:
            xs = _smooth(x, steps, k0, k1, lr0, lr1, d0, d1)
        except Exception:
            continue
        vs = _max_cos(xs)
        # only LP-polish promising ones to save time
        if vs > best + 0.025:
            continue
        lp_sec = 2.5 if remaining > 20 else 1.5
        # alternate longer polish occasionally
        if it % 10 == 0:
            lp_sec = 4.0
        try:
            xp = _lp_polish(xs, lp_sec, t0, BUDGET - 12.0)
        except Exception:
            continue
        vp = _max_cos(xp)
        if vp < best - 1e-12:
            best = vp
            x_best = xp.copy()

    # final long polish of champion
    try:
        left = BUDGET - (time.time() - t0)
        if left > 2:
            x_best = _lp_polish(x_best, min(left - 0.5, 12.0), t0, BUDGET)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

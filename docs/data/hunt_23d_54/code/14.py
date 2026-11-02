import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth(x_init, steps, k0, k1, lr0, lr1, rng=None):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        # max over off-diagonal
        m = float(G[iu].max())
        # stable exp weights, zero diagonal
        E = np.exp(np.clip(k * (G - m), -50.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        x = x - lr * Gt / deg
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _hole_vector(x_keep, rng, n_cand=2000):
    # find direction minimizing max cosine to x_keep
    cand = rng.randn(n_cand, x_keep.shape[1])
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_keep.T
    vals = dots.max(axis=1)
    return cand[int(np.argmin(vals))]


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


def solve():
    t0 = time.time()
    BUDGET = 170.0
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
    best = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            vr = _max_cos(xr)
            if vr < best:
                best, x_best = vr, xr.copy()
    except Exception:
        pass

    iu = np.triu_indices(n, 1)
    it = 0
    # basin hopping with only first-order smoothing (fast, no LP inside)
    while time.time() - t0 < BUDGET - 18.0:
        it += 1
        if time.time() - t0 > BUDGET - 18.0:
            break
        mode = it % 7
        x = x_best.copy()
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        rowmax = Gb.max(axis=1)
        worst = np.argsort(-rowmax)
        cnt = (Gb > best - 0.015).sum(axis=1)
        loose = np.argsort(cnt)

        if mode == 0:
            x += rng.randn(n, d) * 0.010
        elif mode == 1:
            x += rng.randn(n, d) * 0.030
        elif mode == 2:
            # 1-point hole re-insert of worst point
            wi = int(worst[0])
            keep = np.delete(x, wi, axis=0)
            hv = _hole_vector(keep, rng, n_cand=2000)
            hv = hv + rng.randn(d) * 0.01
            hv /= np.linalg.norm(hv)
            x[wi] = hv
            x += rng.randn(n, d) * 0.004
        elif mode == 3:
            # 2-point hole re-insert
            for wi in worst[:2]:
                keep_idx = [k for k in range(n) if k != wi]
                keep = x[keep_idx]
                hv = _hole_vector(keep, rng, n_cand=1500)
                hv = hv + rng.randn(d) * 0.01
                hv /= np.linalg.norm(hv)
                x[int(wi)] = hv
            x += rng.randn(n, d) * 0.004
        elif mode == 4:
            # shake loosest points hard
            ksel = 5
            idx = loose[:ksel]
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.06
        elif mode == 5:
            # shake most jammed points
            ksel = 4
            idx = loose[-ksel:]
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.05
        else:
            x += rng.randn(n, d) * 0.015
            # random rotation of a small subset
            ksel = 8
            idx = rng.choice(n, ksel, replace=False)
            x[idx] += rng.randn(ksel, d) * 0.03

        x /= np.linalg.norm(x, axis=1, keepdims=True)

        # annealed smoothing; vary schedule by iteration
        if mode in (2, 3):
            steps, k0, k1, lr0, lr1 = 700, 150.0, 2500.0, 0.03, 0.003
        elif mode == 1:
            steps, k0, k1, lr0, lr1 = 900, 200.0, 3000.0, 0.02, 0.002
        else:
            steps, k0, k1, lr0, lr1 = 600, 150.0, 2000.0, 0.025, 0.003
        # shrink steps if little time left
        remain = BUDGET - 18.0 - (time.time() - t0)
        if remain < 10:
            steps = min(steps, 250)
        x = _smooth(x, steps, k0, k1, lr0, lr1)
        v = _max_cos(x)
        if v < best:
            best = v
            x_best = x.copy()

    # final short LP polish on the best
    try:
        x_best = _lp_polish(x_best, seconds=14.0, t0=t0, budget=BUDGET - 1.0)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

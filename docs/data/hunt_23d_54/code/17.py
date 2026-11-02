import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth(x, steps, a0, a1, lr0, lr1, mu=0.85):
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 > 0 else a1
        lr = lr0 * (lr1 / lr0) ** frac if steps > 1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights focused on near-max pairs
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True)
        Gn = Gt / (deg + 1.0)
        v = mu * v + Gn
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _hole_reinsert(x, idx_list, rng, n_cand=3000):
    n, d = x.shape
    for idx in idx_list:
        cand = rng.randn(n_cand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # score vs all other points
        G = cand @ x.T
        # ignore self column
        G[:, idx] = -10.0
        scores = G.max(axis=1)
        bi = int(np.argmin(scores))
        x[idx] = cand[bi]
    return x


def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    x_best = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
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
        vy = _max_cos(y)
        if vy < best:
            x, best = y, vy
            x_best = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x_best


def solve():
    t0 = time.time()
    BUDGET = 176.0
    FINAL_LP = 22.0
    rng = np.random.RandomState(12345)
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
    # basin hopping with first-order repulsion only (no LP inside loop)
    while time.time() - t0 < BUDGET - FINAL_LP - 1.0:
        it += 1
        if time.time() - t0 > BUDGET - FINAL_LP - 1.0:
            break
        mode = it % 7
        # contact ranking on current best
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        cnt = np.sum(Gb > best - 0.015, axis=1)
        order_lo = np.argsort(cnt)
        order_hi = np.argsort(-cnt)

        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.010
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 700, 80, 1200, 0.03, 0.003)
        elif mode == 1:
            k = 1
            idx = order_lo[:k].tolist()
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _hole_reinsert(x, idx, rng, n_cand=4000)
            x = _smooth(x, 900, 100, 1800, 0.025, 0.0025)
        elif mode == 2:
            k = 2
            idx = order_lo[:k].tolist()
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _hole_reinsert(x, idx, rng, n_cand=3000)
            x = _smooth(x, 900, 100, 1800, 0.025, 0.0025)
        elif mode == 3:
            k = 3
            idx = order_hi[:k].tolist()
            x[idx] += rng.randn(k, d) * 0.06
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 800, 60, 1000, 0.03, 0.003)
        elif mode == 4:
            x += rng.randn(n, d) * 0.028
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 1000, 50, 1500, 0.028, 0.002)
        elif mode == 5:
            k = 4
            idx = rng.choice(n, size=k, replace=False).tolist()
            x = _hole_reinsert(x, idx, rng, n_cand=2500)
            x = _smooth(x, 900, 90, 1600, 0.025, 0.0025)
        else:
            # strong global kick + hole for loosest
            x += rng.randn(n, d) * 0.045
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 700, 40, 800, 0.035, 0.004)

        v = _max_cos(x)
        if v < best:
            best = v
            x_best = x.copy()
        # opportunistic sharp tighten if very close
        if v < best + 1e-7 and v <= best and False:
            pass
        # safety: stop if time low
        if time.time() - t0 > BUDGET - FINAL_LP - 1.0:
            break

    # short sharp descent before LP
    try:
        if time.time() - t0 < BUDGET - FINAL_LP + 4.0:
            x_best = _smooth(x_best.copy(), 300, 800, 4000, 0.008, 0.0015, mu=0.7)
            vb = _max_cos(x_best)
            if vb < best:
                best = vb
    except Exception:
        pass

    # final trust-region LP polish
    t_end = t0 + BUDGET
    remain = t_end - time.time() - 0.5
    if remain > 3.0:
        try:
            x_best = _lp_polish(x_best, min(remain, FINAL_LP + 5.0), t_end - 0.3)
        except Exception:
            pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best[~np.isfinite(x_best)] = 0.0
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    bad = (nr.squeeze() == 0)
    if np.any(bad):
        x_best[bad] = rng.randn(np.count_nonzero(bad), d)
        x_best[bad] /= np.linalg.norm(x_best[bad], axis=1, keepdims=True)
    return x_best

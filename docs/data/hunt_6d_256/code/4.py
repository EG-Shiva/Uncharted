import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _smooth_polish(x, steps=800, beta0=25.0, beta1=130.0, lr0=0.06, lr1=0.004):
    n, d = x.shape
    for k in range(steps):
        f = k / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = np.max(G)
        # soft-max weights, focus on near-max pairs
        W = np.exp(beta * (G - m))
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        F = (W @ x) / rs
        proj = (F * x).sum(axis=1, keepdims=True)
        Ft = F - proj * x
        x = x - lr * Ft
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _lp_polish(x, seconds, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    # precompute iu arrays
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        Ioff = (I[:, None] * d + np.arange(d)).ravel()
        Joff = (J[:, None] * d + np.arange(d)).ravel()
        cols_IJ = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols_IJ[0::(2 * d + 1)] = 0  # placeholder, we build directly below
        # build cols vectorized
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)).ravel(),
            (J[:, None] * d + np.arange(d)).ravel(),
        ])
        # interleave per row: need rows repeat pattern, easier build full arrays
        cols_full = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals_full = np.empty(p * (2 * d + 1), dtype=np.float64)
        # fill row by row using vectorization
        # I part
        rpt = np.repeat(np.arange(p), d)
        cols_full[0:p * d] = 0  # temp
        # simpler: construct via tiling
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cT = np.full((p, 1), n * d)
        cols_mat = np.concatenate([cI, cJ, cT], axis=1)
        vI = x[J]
        vJ = x[I]
        vT = -np.ones((p, 1))
        vals_mat = np.concatenate([vI, vJ, vT], axis=1)
        cols_full = cols_mat.ravel()
        vals_full = vals_mat.ravel()
        rows_full = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals_full, (rows_full, cols_full)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (np.tile(x.ravel(), 1), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def _teleport_to_holes(y, idx_move, rng, n_cand=6000):
    n = y.shape[0]
    for r in idx_move:
        cands = rng.standard_normal(size=(n_cand, 6))
        cn = np.linalg.norm(cands, axis=1, keepdims=True)
        cn[cn == 0] = 1.0
        cands = cands / cn
        D = cands @ y.T
        D[:, r] = -10.0
        md = D.max(axis=1)
        bi = int(np.argmin(md))
        y[r] = cands[bi]
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return y


def solve():
    t0 = time.time()
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (256, 6):
        # try to reshape / truncate if needed
        xb = xb.reshape(256, 6)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_x = xb.copy()
    best = _max_cos(best_x)
    rng = np.random.default_rng(0)

    it = 0
    # main basin-hopping loop, leave ~25s for final polish
    while time.time() - t0 < 148.0:
        it += 1
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        cnt = (G > m - 1e-9).sum(axis=1)
        ratt = np.where(cnt == 0)[0]
        hubs = np.argsort(-cnt)[:12]
        mode = it % 7
        y = best_x.copy()
        if mode == 0:
            y = y + rng.standard_normal(y.shape) * 0.006
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_polish(y, steps=650, beta0=30, beta1=140, lr0=0.05, lr1=0.003)
        elif mode == 1:
            y = y + rng.standard_normal(y.shape) * 0.018
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_polish(y, steps=800, beta0=25, beta1=130, lr0=0.06, lr1=0.004)
        elif mode == 2:
            y[hubs] = y[hubs] + rng.standard_normal((len(hubs), 6)) * 0.07
            y = y + rng.standard_normal(y.shape) * 0.002
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_polish(y, steps=800, beta0=25, beta1=150, lr0=0.06, lr1=0.003)
        elif mode == 3:
            if len(ratt) > 0:
                y = _teleport_to_holes(y, ratt, rng, n_cand=8000)
                y = y + rng.standard_normal(y.shape) * 0.004
            else:
                y = _teleport_to_holes(y, hubs[:3], rng, n_cand=8000)
                y = y + rng.standard_normal(y.shape) * 0.004
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_polish(y, steps=900, beta0=25, beta1=140, lr0=0.055, lr1=0.003)
        elif mode == 4:
            sel = np.unique(np.concatenate([hubs[:4], rng.choice(256, 4, replace=False)]))
            y = _teleport_to_holes(y, sel, rng, n_cand=5000)
            y = y + rng.standard_normal(y.shape) * 0.003
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_polish(y, steps=800, beta0=30, beta1=140, lr0=0.05, lr1=0.003)
        elif mode == 5:
            sel = rng.choice(256, 20, replace=False)
            y[sel] = y[sel] + rng.standard_normal((20, 6)) * 0.12
            y = y + rng.standard_normal(y.shape) * 0.002
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_polish(y, steps=900, beta0=22, beta1=130, lr0=0.065, lr1=0.004)
        else:
            y = y + rng.standard_normal(y.shape) * 0.011
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_polish(y, steps=700, beta0=35, beta1=160, lr0=0.045, lr1=0.002)

        v = _max_cos(y)
        if v < best - 1e-12:
            t_rem = 148.0 - (time.time() - t0)
            if t_rem < 1.0:
                if v < best:
                    best_x, best = y, v
                continue
            lp_t = min(2.5, max(0.8, t_rem * 0.04))
            y2 = _lp_polish(y, seconds=lp_t, r0=2e-3)
            v2 = _max_cos(y2)
            if v2 < best:
                best_x, best = y2, v2
            elif v < best:
                best_x, best = y, v
        elif v < best + 0.002 and (it % 12 == 0):
            # opportunistic short LP on near-miss basins
            t_rem = 148.0 - (time.time() - t0)
            if t_rem > 2.0:
                y2 = _lp_polish(y, seconds=0.8, r0=1.5e-3)
                v2 = _max_cos(y2)
                if v2 < best:
                    best_x, best = y2, v2
        if time.time() - t0 >= 148.0:
            break

    t_left = 178.0 - (time.time() - t0)
    if t_left > 4.0:
        best_x = _lp_polish(best_x, seconds=min(t_left - 1.0, 22.0), r0=1.5e-3)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, init_radius=5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    while time.time() - start < seconds and radius > 1e-14:
        # remaining time check
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3500)[-3500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3500
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        # A_ub rows: xJ*dxI + xI*dxJ - t <= -c
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        # interleave: need vals aligned with rows order: [xJ block, xI block, -1]
        cols_first = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        # fill per row block
        # faster vectorized:
        # positions 0:d -> I block with xJ, d:2d -> J block with xI, 2d -> t
        rbase = (np.arange(p)[:, None] * (2 * d + 1))
        # I block cols
        cols_first.reshape(p, 2 * d + 1)[:, :d] = I[:, None] * d + np.arange(d)
        cols_first.reshape(p, 2 * d + 1)[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols_first.reshape(p, 2 * d + 1)[:, 2 * d] = n * d
        vals.reshape(p, 2 * d + 1)[:, :d] = x[J]
        vals.reshape(p, 2 * d + 1)[:, d:2 * d] = x[I]
        vals.reshape(p, 2 * d + 1)[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols_first)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _worst_per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    w = G.max(axis=1)
    return w


def _contact_counts(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    cnt = np.sum(G > m - tol, axis=1)
    return cnt


def _greedy_reinsert(y, to_replace, rng, n_cand=12000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[to_replace] = False
    X_fixed = y[fixed_mask]
    new_pts = []
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if new_pts:
            Xa = np.vstack([X_fixed] + new_pts)
        else:
            Xa = X_fixed
        dots = cand @ Xa.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        bp = cand[bi]
        new_pts.append(bp[None, :])
        y[k] = bp
    # small hill climb on new points
    for _ in range(2):
        for idx_i, k in enumerate(to_replace):
            cur = y[k].copy()
            # current worst
            others = np.delete(y, k, axis=0)
            cur_w = float((others @ cur).max())
            trial = rng.standard_normal(size=(400, d)) * 0.05 + cur[None, :]
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            dw = (trial @ others.T).max(axis=1)
            bi = int(np.argmin(dw))
            if dw[bi] < cur_w:
                y[k] = trial[bi]
    return y


def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    # load start
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b0:
                    b0 = v
                    x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    x_best = x0.copy()
    best = float(b0)

    # quick initial tight polish if time allows (1s) - already polished, skip mostly
    it = 0
    no_improve = 0
    # main basin hopping, reserve ~25s for final
    while True:
        elapsed = time.time() - t0
        if elapsed > 148:
            break
        # choose strategy
        s = it % 8
        x = x_best.copy()
        n, d = x.shape
        if s == 0:
            sig = 0.020 * (1.0 + 0.1 * min(no_improve, 10))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            budget = 4.0
            init_r = 0.008
        elif s == 1:
            w = _worst_per_point(x)
            k = 40
            idx = np.argpartition(w, -k)[-k:]
            sig = 0.06
            x[idx] += rng.standard_normal(size=(k, d)) * sig
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            budget = 4.0
            init_r = 0.008
        elif s == 2:
            m = best
            cc = _contact_counts(x, m, tol=1e-9)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-cc)
            torep = []
            for r in ratt:
                torep.append(int(r))
            for o in order:
                if len(torep) >= 6:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:6]
            x = _greedy_reinsert(x, torep, rng, n_cand=12000)
            # tiny shake rest
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            budget = 5.0
            init_r = 0.008
        elif s == 3:
            sig = 0.035
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            budget = 4.0
            init_r = 0.008
        elif s == 4:
            k = 30
            idx = rng.choice(n, size=k, replace=False)
            sig = 0.05
            x[idx] += rng.standard_normal(size=(k, d)) * sig
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            budget = 4.0
            init_r = 0.008
        elif s == 5:
            sig = 0.008
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            budget = 3.5
            init_r = 0.004
        elif s == 6:
            w = _worst_per_point(x)
            k = 20
            idx = np.argpartition(w, -k)[-k:]
            sig = 0.10
            x[idx] += rng.standard_normal(size=(k, d)) * sig
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            budget = 4.0
            init_r = 0.01
        else:
            m = best
            w = _worst_per_point(x)
            k = 10
            idx = np.argpartition(w, -k)[-k:].tolist()
            x = _greedy_reinsert(x, [int(v) for v in idx], rng, n_cand=8000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            budget = 5.0
            init_r = 0.008
        # adjust budget to remaining
        remain = 148 - (time.time() - t0)
        if remain < 1.0:
            break
        budget = min(budget, remain)
        try:
            y = _lp_polish(x, seconds=budget, init_radius=init_r)
        except Exception:
            it += 1
            no_improve += 1
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            best = v
            x_best = y
            no_improve = 0
        else:
            no_improve += 1
        it += 1

    # final squeeze on best
    remain = 178 - (time.time() - t0)
    if remain > 2:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain, 25), init_radius=2e-3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

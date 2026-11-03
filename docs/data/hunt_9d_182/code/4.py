import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _reinsert_try(x, cur_best, rng_K=1500):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    row_max = G.max(axis=1)
    order = np.argsort(-row_max)[:3]
    improved = False
    for w in order:
        R = np.random.randn(rng_K, d)
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        S = R @ x.T
        S[:, w] = -2.0
        s_max = S.max(axis=1)
        k = int(np.argmin(s_max))
        if s_max[k] < cur_best - 1e-12:
            x[w] = R[k]
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            cur_best = float(G2.max())
            improved = True
            break
    return x, cur_best, improved

def _smooth_stage(x, best, deadline):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    t_start = time.time()
    best_x = x.copy()
    T0, T1 = 0.015, 0.0012
    lr0, lr1 = 0.003, 0.0004
    while time.time() < deadline:
        el = time.time() - t_start
        total = max(deadline - t_start, 1e-9)
        frac = min(max(el / total, 0.0), 1.0)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        # max off diag
        m = -1e99
        # fast max without diag pollution
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights
        # clip for overflow
        Z = (G - m) / T
        # Z <= 0, exp safe
        W = np.exp(np.clip(Z, -30, 0.0))
        np.fill_diagonal(W, 0.0)
        grad = (W @ x) / T
        # tangent projection
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        step += 1
        m_ad = b1 * m_ad + (1 - b1) * gt
        v_ad = b2 * v_ad + (1 - b2) * (gt * gt)
        m_hat = m_ad / (1 - b1 ** step)
        v_hat = v_ad / (1 - b2 ** step)
        dx = lr * m_hat / (np.sqrt(v_hat) + eps)
        # limit per-point norm to keep trust
        y = x - dx
        y = _norm(y)
        x = y
        if step % 100 == 0:
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
        if step % 400 == 0:
            # try reinsertion on current x
            v = _max_cos(x)
            x, v2, _ = _reinsert_try(x, v)
            if v2 < best:
                best = v2
                best_x = x.copy()
            # re-init adam slightly to avoid stale momentum
            m_ad *= 0.5
            v_ad *= 0.5
            if time.time() >= deadline:
                break
    # return best found, but continue from best
    return best_x, best

def _contact_stage(x, best, deadline):
    lr = 2e-3
    tol = 2e-3
    it = 0
    while time.time() < deadline:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        A = (G > m - tol).astype(np.float64)
        deg = A.sum(axis=1)
        AG = A * G
        F = A @ x - AG.sum(axis=1)[:, None] * x
        scale = np.maximum(deg, 1.0)[:, None]
        F = F / scale
        # guard large forces
        fn = np.linalg.norm(F, axis=1, keepdims=True)
        # optional clip
        # propose
        y = x - lr * F
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            lr = min(lr * 1.08, 5e-3)
            tol = max(tol * 0.998, 2e-7)
        else:
            lr *= 0.6
            if lr < 1e-7:
                tol = max(tol * 0.5, 1e-9)
                lr = 5e-4
                # small random tangent jiggle of worst points to escape flat
                if it % 20 == 0:
                    Gw = x @ x.T
                    np.fill_diagonal(Gw, -2.0)
                    rm = Gw.max(axis=1)
                    w = int(np.argmax(rm))
                    rnd = np.random.randn(9)
                    rnd = rnd - (rnd @ x[w]) * x[w]
                    rn = np.linalg.norm(rnd)
                    if rn > 1e-12:
                        rnd = rnd / rn * 1e-4
                        y2 = x.copy()
                        y2[w] = y2[w] + rnd
                        y2 = _norm(y2)
                        v2 = _max_cos(y2)
                        if v2 < best + 1e-7:
                            x = y2
                            if v2 < best:
                                best = v2
        # slow anneal of tol with iterations
        if it % 500 == 0:
            tol = max(tol * 0.8, 1e-9)
    return x, best

def _lp_polish(x, best, deadline):
    n, d = x.shape
    x = _norm(x)
    radius = 3e-4
    iu = np.triu_indices(n, 1)
    maxK = 2500
    while time.time() < deadline and radius > 1e-13:
        Gfull = x @ x.T
        cfull = Gfull[iu]
        m = float(cfull.max())
        if m < best:
            best = m
        thresh = max(5.0 * radius, 1e-10)
        sel = cfull > m - thresh
        # cap to largest maxK
        if int(sel.sum()) > maxK:
            # keep largest maxK
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-maxK:]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 1.5
            continue
        csel = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.2, 1e-3)
        else:
            radius *= 0.5
    return x, best

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 168.0
    # load start
    best_x = None
    best_v = 1e99
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (182, 9):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = np.random.randn(182, 9)
        best_x = _norm(best_x)
        best_v = _max_cos(best_x)
    x = best_x.copy()
    # stages with deadlines
    # smooth till +65s, contact till +130s, lp till budget
    t_smooth_end = t0 + 65.0
    t_contact_end = t0 + 130.0
    t_lp_end = t0 + BUDGET
    # stage 1
    if time.time() < t_smooth_end:
        x, best_v = _smooth_stage(x, best_v, min(t_smooth_end, t0 + BUDGET - 40))
        # keep best
    # stage 2
    if time.time() < t_contact_end:
        # restart from best tracker? x is already near best; ensure x is best-like
        x, best_v = _contact_stage(x, best_v, min(t_contact_end, t0 + BUDGET - 20))
    # stage 3 lp fine polish
    if time.time() < t_lp_end - 2.0:
        x, best_v = _lp_polish(x, best_v, t_lp_end - 1.0)
    x = _norm(x)
    return x

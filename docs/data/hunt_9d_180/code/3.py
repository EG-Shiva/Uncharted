import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        # cap number of constraints for speed
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            radius *= 0.5
            continue
        if p0 > 3500:
            # keep largest 3500
            csel = c[sel]
            part = np.argpartition(csel, -3500)[-3500:]
            idx_all = np.flatnonzero(sel)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx_all[part]] = True
            sel = keep
        I = iu[0][sel]
        J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn = np.maximum(yn, 1e-12)
        y = y / yn
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_opt(x, steps, T0, T1, lr0):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.9 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mval = float(np.max(G))
        # weights, max is 1
        W = np.exp((G - mval) / T)
        # diag is 0 because exp(-inf)=0
        S = float(np.sum(W) * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            break
        Grad = (W @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1 ** t)
        vhat = v_ad / (1.0 - b2 ** t)
        upd = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * upd
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x = x / nrm
    return x


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    TIME_BUDGET = 172.0

    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = xb
    xb = np.asarray(xb, dtype=np.float64)
    xr = np.asarray(xr, dtype=np.float64)
    # pick better of the two as start
    def normed(a):
        a = np.asarray(a, dtype=np.float64)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        return a
    xb = normed(xb)
    xr = normed(xr)
    vb = _max_cos(xb)
    vr = _max_cos(xr)
    if vr < vb:
        xb, vb = xr, vr
    x_best = xb.copy()
    best_val = vb
    n, d = x_best.shape

    it = 0
    # reserve ~32s for final polish
    while time.time() - t0 < TIME_BUDGET - 32.0:
        it += 1
        r = rng.random()
        x_cand = x_best.copy()
        if r < 0.55:
            # targeted shake of high-contact points
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnts = np.sum(G > m - 0.003, axis=1)
            k = int(rng.integers(5, 20))
            idx = np.argsort(cnts)[-k:]
            s_all = 0.006 + rng.random() * 0.012
            s_big = 0.10 + rng.random() * 0.18
            noise = rng.standard_normal(size=(n, d)) * s_all
            noise[idx] += rng.standard_normal(size=(k, d)) * s_big
            x_cand = x_best + noise
            x_cand = x_cand / np.maximum(np.linalg.norm(x_cand, axis=1, keepdims=True), 1e-12)
        elif r < 0.80:
            # global shake
            s = 0.02 + rng.random() * 0.07
            noise = rng.standard_normal(size=(n, d)) * s
            x_cand = x_best + noise
            x_cand = x_cand / np.maximum(np.linalg.norm(x_cand, axis=1, keepdims=True), 1e-12)
        else:
            # re-insert 1-2 worst points into voids + small global jitter
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnts = np.sum(G > m - 0.003, axis=1)
            k = int(rng.integers(1, 3))
            worst = np.argsort(cnts)[-k:]
            s_all = 0.005 + rng.random() * 0.008
            x_cand = x_best + rng.standard_normal(size=(n, d)) * s_all
            x_cand = x_cand / np.maximum(np.linalg.norm(x_cand, axis=1, keepdims=True), 1e-12)
            for wi in worst:
                mask = np.ones(n, dtype=bool)
                mask[wi] = False
                others = x_cand[mask]
                # sample many random directions, pick most isolated
                nt = 3000
                cand_dirs = rng.standard_normal(size=(nt, d))
                cand_dirs /= np.maximum(np.linalg.norm(cand_dirs, axis=1, keepdims=True), 1e-12)
                dots = cand_dirs @ others.T
                np.fill_diagonal(np.zeros((1, 1)), 0)  # no-op
                worst_dot = np.max(dots, axis=1)
                bi = int(np.argmin(worst_dot))
                x_cand[wi] = cand_dirs[bi]
            x_cand = x_cand / np.maximum(np.linalg.norm(x_cand, axis=1, keepdims=True), 1e-12)

        T0 = 0.015 + rng.random() * 0.025
        T1 = 0.002 + rng.random() * 0.004
        lr0 = 0.004 + rng.random() * 0.009
        steps = int(600 + rng.random() * 900)
        x_sm = _smooth_opt(x_cand, steps, T0, T1, lr0)
        v = _max_cos(x_sm)
        if v < best_val:
            # promising: short LP refine
            budget = 3.0
            if time.time() - t0 > TIME_BUDGET - 35:
                budget = 1.0
            x_ref = _lp_polish(x_sm, budget)
            v2 = _max_cos(x_ref)
            if v2 < best_val:
                x_best, best_val = x_ref, v2
            elif v < best_val:
                # keep smooth even if LP failed to improve? keep best smooth
                # only update if LP didn't worsen beyond smooth
                if v < best_val:
                    # re-evaluate: v < best already, so update to better of the two
                    if v2 < v:
                        x_best, best_val = x_ref, v2
                    else:
                        x_best, best_val = x_sm, v
        elif v < best_val + 0.004 and rng.random() < 0.22:
            # occasional exploitation
            if time.time() - t0 > TIME_BUDGET - 40:
                continue
            x_ref = _lp_polish(x_sm, 1.2)
            v2 = _max_cos(x_ref)
            if v2 < best_val:
                x_best, best_val = x_ref, v2
        # safety time break
        if time.time() - t0 >= TIME_BUDGET - 32.0:
            break

    remain = TIME_BUDGET - (time.time() - t0) - 1.5
    if remain < 5:
        remain = 5
    # final long polish, capped to not exceed budget
    remain = min(remain, 40.0)
    x_best = _lp_polish(x_best, remain)
    x_best = x_best / np.maximum(np.linalg.norm(x_best, axis=1, keepdims=True), 1e-12)
    return x_best

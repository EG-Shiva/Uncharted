import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        if it >= 120:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = c > thr
        # always include the max pair(s)
        if np.count_nonzero(sel) < 1:
            sel = c >= m
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        # A_ub * [dx,t] <= -c_sel
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def contact_degree(x, thresh=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    B = G >= m - thresh
    # each edge counted twice, degree per point
    deg = B.sum(axis=1)
    return deg, float(m)


def best_void_against(Y, rng, n_sample=12000, refine=150):
    # find point on S^3 minimizing max dot to Y
    cand = rng.standard_normal(size=(n_sample, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ Y.T
    ms = s.max(axis=1)
    idx = int(np.argmin(ms))
    y = cand[idx].copy()
    cur = float(ms[idx])
    step = 0.02
    for _ in range(refine):
        prop = y + step * rng.standard_normal(4)
        prop /= np.linalg.norm(prop)
        v = float(np.max(prop @ Y.T))
        if v < cur:
            y = prop
            cur = v
        else:
            step *= 0.97
            if step < 1e-4:
                step = 1e-4
    return y


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    # load starting points
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (163, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        a = rng.standard_normal(size=(163, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best = a
        best_val = max_cos(a)
    n, d = best.shape

    # quick initial polish to lock baseline (short)
    if HARD_END - time.time() > 20:
        best = lp_polish(best, seconds=3.0, t_deadline=HARD_END - 5)
        best_val = max_cos(best)

    deg, _ = contact_degree(best, thresh=1e-9)

    trial = 0
    # main basin hopping: mix of global kicks, targeted kicks, dropout
    # reserve last ~12s for final polish
    while time.time() < HARD_END - 12:
        trial += 1
        remaining = HARD_END - 12 - time.time()
        if remaining <= 0:
            break
        mode = trial % 5
        x_try = None
        if mode in (0, 1):
            # global gaussian kick, cycling sigma
            sigmas = [0.025, 0.012, 0.006, 0.003, 0.0015]
            sigma = sigmas[(trial // 5) % len(sigmas)]
            noise = rng.standard_normal(size=(n, d))
            x_try = best + sigma * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode in (2, 3):
            # targeted kick on overloaded + rattlers
            # recompute deg occasionally
            if trial % 10 == 2:
                deg, _ = contact_degree(best, thresh=1e-9)
            order = np.argsort(-deg)
            k = int(rng.integers(4, 10))
            idx = order[:k]
            # also include rattlers (deg==0)
            ratt = np.where(deg == 0)[0]
            idx = np.unique(np.concatenate([idx, ratt])) if len(ratt) else idx
            sigma = 0.02 if mode == 2 else 0.008
            noise = np.zeros((n, d))
            noise[idx] = rng.standard_normal(size=(len(idx), d))
            # small background shake to allow topology change
            noise += 0.0015 * rng.standard_normal(size=(n, d))
            x_try = best + sigma * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # dropout-reinsertion of jammed points
            if trial % 10 == 4:
                deg, _ = contact_degree(best, thresh=1e-7)
            order = np.argsort(-deg)
            k = int(rng.integers(2, 6))
            drop = order[:k]
            # sometimes drop a random point too
            if rng.random() < 0.3:
                drop = np.unique(np.append(drop, rng.integers(0, n)))
                k = len(drop)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[drop] = False
            Y = best[keep_mask]
            new_pts = []
            curY = Y.copy()
            for _ in range(k):
                yv = best_void_against(curY, rng, n_sample=8000, refine=120)
                new_pts.append(yv)
                curY = np.vstack([curY, yv[None, :]])
            x_try = np.empty_like(best)
            x_try[keep_mask] = Y
            x_try[~keep_mask] = np.array(new_pts)
            # tiny jitter
            x_try = x_try + 0.002 * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # short LP polish for this trial
        budget = 1.2
        if trial % 15 == 0:
            budget = 2.0
        # don't exceed reserve
        budget = min(budget, max(0.4, HARD_END - 12 - time.time()))
        if budget <= 0.05:
            break
        y = lp_polish(x_try, seconds=budget, t_deadline=HARD_END - 12)
        v = max_cos(y)
        if v < best_val:
            best = y
            best_val = v
            deg, _ = contact_degree(best, thresh=1e-9)

    # final long polish on champion
    fbudget = max(0.5, HARD_END - time.time() - 1.0)
    fbudget = min(fbudget, 25.0)
    if fbudget > 0.3:
        best = lp_polish(best, seconds=fbudget)
        best_val = max_cos(best)

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

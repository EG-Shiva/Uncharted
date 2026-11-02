import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _contact_counts(x, tol=2e-6):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = np.sum(g > m - tol, axis=1).astype(int)
    return cnt, m

def _lp_polish(x, deadline, r_init=0.01, r_max=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    while time.time() < deadline and radius > 1e-13:
        if time.time() + 0.35 > deadline:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return x, best

def _refine_void(y0, Xk, beta=300.0, iters=150, lr=0.15):
    y = y0 / (np.linalg.norm(y0) + 1e-18)
    for it in range(iters):
        s = Xk @ y
        mx = s.max()
        e = np.exp(beta * (s - mx))
        w = e / (e.sum() + 1e-30)
        grad = Xk.T @ w
        gt = grad - float(grad @ y) * y
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14:
            break
        step = lr / (1.0 + it * 0.02)
        y = y - step * gt
        n = float(np.linalg.norm(y))
        if not np.isfinite(n) or n < 1e-12:
            return y0 / np.linalg.norm(y0)
        y = y / n
    return y

def _void_points(X_keep, k, rng, S=15000):
    d = X_keep.shape[1]
    keep = X_keep.copy()
    outs = []
    for _ in range(k):
        cand = rng.standard_normal((S, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        scores = cand @ keep.T
        mx = scores.max(axis=1)
        bidx = int(np.argmin(mx))
        y0 = cand[bidx].copy()
        # also try a couple of runners-up refined and pick best
        # refine best only for speed
        y = _refine_void(y0, keep)
        outs.append(y)
        keep = np.vstack([keep, y[None, :]])
    return np.array(outs)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    cand = []
    try:
        b = np.load("best.npy")
        if b.shape == (n, d):
            cand.append(b.astype(float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        if r.shape == (n, d):
            cand.append(r.astype(float))
    except Exception:
        pass
    if len(cand) == 0:
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    best_v = 1e9
    best_x = None
    for c in cand:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    x_best = best_x.copy()
    v_best = best_v
    # initial squeeze
    if time.time() < HARD_END - 40:
        x_best, v_best = _lp_polish(x_best, min(time.time() + 6.0, HARD_END - 35), r_init=0.002, r_max=0.02)
    stagn = 0
    trial = 0
    while time.time() < HARD_END - 28:
        trial += 1
        remaining = HARD_END - 28 - time.time()
        if remaining < 3.0:
            break
        budget = min(7.0, max(3.0, remaining / 3.0))
        deadline = time.time() + budget
        mode = trial % 6
        # adapt: if stagnant, force void moves
        if stagn >= 10 and mode in (2, 4):
            mode = 1
        x_try = x_best.copy()
        if mode == 0:
            # void replace 1 lowest-contact
            cnt, _ = _contact_counts(x_try, tol=2e-6)
            order = np.argsort(cnt)
            k = 1
            repl = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[repl] = False
            Xk = x_try[keep_mask]
            nv = _void_points(Xk, k, rng, S=15000)
            x_try[repl] = nv
            x_try[~keep_mask] = x_try[~keep_mask] / np.linalg.norm(x_try[~keep_mask], axis=1, keepdims=True)
            # tiny jitter to rest
            jm = np.zeros((n, d))
            jm[keep_mask] = 0.002 * rng.standard_normal((np.count_nonzero(keep_mask), d))
            x_try = x_try + jm
            r_init = 0.015
        elif mode == 1:
            cnt, _ = _contact_counts(x_try, tol=2e-6)
            order = np.argsort(cnt)
            k = 2 if (stagn < 6) else 3
            if trial % 12 == 1:
                k = 3
            repl = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[repl] = False
            Xk = x_try[keep_mask]
            nv = _void_points(Xk, k, rng, S=12000)
            x_try[repl] = nv
            jm = np.zeros((n, d))
            jm[keep_mask] = 0.003 * rng.standard_normal((np.count_nonzero(keep_mask), d))
            x_try = x_try + jm
            r_init = 0.018
        elif mode == 2:
            # small global jitter
            sigma = 0.008 if stagn < 5 else 0.012
            x_try = x_try + sigma * rng.standard_normal((n, d))
            r_init = 0.008
        elif mode == 3:
            pm = _personal_max(x_try)
            order = np.argsort(-pm)
            k = 6
            worst = order[:k]
            noise = 0.005 * rng.standard_normal((n, d))
            noise[worst] += 0.06 * rng.standard_normal((k, d))
            x_try = x_try + noise
            r_init = 0.015
        elif mode == 4:
            sigma = 0.025
            x_try = x_try + sigma * rng.standard_normal((n, d))
            r_init = 0.015
        else:
            cnt, _ = _contact_counts(x_try, tol=2e-6)
            order = np.argsort(cnt)
            tail = order[:14]
            noise = 0.004 * rng.standard_normal((n, d))
            noise[tail] += 0.05 * rng.standard_normal((len(tail), d))
            x_try = x_try + noise
            r_init = 0.015
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        x_try, v_try = _lp_polish(x_try, deadline, r_init=r_init, r_max=0.05)
        if v_try < v_best - 1e-13:
            x_best = x_try
            v_best = v_try
            stagn = 0
        else:
            stagn += 1
        if time.time() >= HARD_END - 28:
            break
    if time.time() < HARD_END - 1.0:
        x_best, v_best = _lp_polish(x_best, HARD_END - 0.6, r_init=0.003, r_max=0.02)
        if time.time() < HARD_END - 0.4:
            x_best, v_best = _lp_polish(x_best, HARD_END - 0.2, r_init=0.0006, r_max=0.006)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return x_best

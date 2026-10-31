import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_TRIU = None

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def lp_polish(x, t_end, r0=0.003):
    x = _norm_rows(x.copy())
    n, d = x.shape
    global _TRIU
    if _TRIU is None or _TRIU[0].shape[0] != n * (n - 1) // 2:
        _TRIU = np.triu_indices(n, 1)
    iu = _TRIU
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = float(r0)
    nv = n * d + 1
    it = 0
    while time.time() < t_end and radius > 1e-14 and it < 80:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        # cap number of constraints for speed
        p_all = int(np.sum(sel))
        if p_all == 0:
            radius *= 0.5
            continue
        idx_all = np.where(sel)[0]
        if p_all > 2200:
            # keep largest 2200
            part = np.argpartition(c[idx_all], -2200)[-2200:]
            idx = idx_all[part]
        else:
            idx = idx_all
        p = idx.shape[0]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bnds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = x + dx
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            # early exit if time low
        else:
            radius *= 0.5
    return x

def smooth_relax(x, steps, beta, lr):
    x = _norm_rows(x.copy())
    n, d = x.shape
    for _ in range(int(steps)):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # weights in (0,1]
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        proj = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - proj * x
        step = lr * gt
        # clip per-row step to 0.02 for stability
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.02
        if np.any(over):
            scale = np.ones_like(sn)
            scale[over] = 0.02 / (sn[over] + 1e-15)
            step = step * scale
        x = x - step
        x = _norm_rows(x)
    return x

def _row_maxes(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    x_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm_rows(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(x_best)
    # quick initial polish
    try:
        x_best = lp_polish(x_best, min(time.time() + 6.0, T_END - 10.0), r0=0.002)
        best_val = _max_cos(x_best)
    except Exception:
        pass
    x_cur = x_best.copy()
    cur_val = best_val
    temp = 0.003
    it = 0
    while time.time() < T_END - 3.0:
        it += 1
        r = rng.random()
        y = None
        try:
            if r < 0.50:
                # global shake, log-uniform sigma
                sigma = 10.0 ** float(rng.uniform(-3.0, -1.15))
                y = x_cur + sigma * rng.standard_normal(size=(n, d))
                y = _norm_rows(y)
                # light smooth to settle
                y = smooth_relax(y, steps=80, beta=60.0, lr=0.008)
            elif r < 0.70:
                # subset reinit, bias to worst points
                k = int(rng.integers(1, 5))
                y = x_cur.copy()
                mrow, _ = _row_maxes(x_cur)
                if rng.random() < 0.6:
                    order = np.argsort(-mrow)
                    pool = order[:max(12, 3 * k)]
                    idx = rng.choice(pool, size=k, replace=False)
                else:
                    idx = rng.choice(n, size=k, replace=False)
                y[idx] = rng.standard_normal(size=(k, d))
                y = _norm_rows(y)
            elif r < 0.85:
                # targeted max-pair nudge + small noise
                G = x_cur @ x_cur.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                ii, jj = flat // n, flat % n
                y = x_cur.copy()
                eps = float(rng.uniform(0.01, 0.06))
                # push apart tangentially
                vi = x_cur[ii] - x_cur[jj]
                vj = x_cur[jj] - x_cur[ii]
                # remove radial component
                vi = vi - np.dot(vi, x_cur[ii]) * x_cur[ii]
                vj = vj - np.dot(vj, x_cur[jj]) * x_cur[jj]
                ni = np.linalg.norm(vi) + 1e-12
                nj = np.linalg.norm(vj) + 1e-12
                y[ii] = y[ii] + eps * vi / ni
                y[jj] = y[jj] + eps * vj / nj
                # add tiny global noise
                y = y + 0.002 * rng.standard_normal(size=(n, d))
                y = _norm_rows(y)
                y = smooth_relax(y, steps=60, beta=70.0, lr=0.008)
            else:
                # fresh random + repulsive relaxation
                y = _norm_rows(rng.standard_normal(size=(n, d)))
                y = smooth_relax(y, steps=400, beta=22.0, lr=0.02)
                y = smooth_relax(y, steps=400, beta=65.0, lr=0.01)
        except Exception:
            continue
        # LP polish with short budget
        budget = 2.2
        # fresh starts get a bit more
        if r >= 0.85:
            budget = 2.8
        t_end = min(time.time() + budget, T_END - 3.0)
        if t_end - time.time() < 0.3:
            break
        try:
            y2 = lp_polish(y, t_end, r0=0.004)
            v2 = _max_cos(y2)
        except Exception:
            continue
        if v2 < best_val - 1e-13:
            best_val = v2
            x_best = y2.copy()
            x_cur = y2.copy()
            cur_val = v2
        else:
            # annealing acceptance for cur
            if v2 < cur_val or rng.random() < np.exp((cur_val - v2) / max(temp, 1e-4)):
                x_cur = y2.copy()
                cur_val = v2
            # reset wandering cur if too bad
            if cur_val > best_val + 0.012:
                x_cur = x_best.copy()
                cur_val = best_val
        temp = max(0.0004, temp * 0.995)
        # opportunistic: if very close, try single-point sampling on rattlers
        # (cheap, no LP) every ~20 iters
        if it % 25 == 0 and time.time() < T_END - 6.0:
            try:
                mrow, Gfull = _row_maxes(x_best)
                # rattlers: far below max
                thresh = best_val - 0.002
                ratt = np.where(mrow < thresh)[0]
                if ratt.size > 0:
                    for _t in range(30):
                        i = int(rng.choice(ratt))
                        cand = _norm_rows(rng.standard_normal(size=(1, d)))[0]
                        # keep if improves its own max and stays below best
                        dots = x_best @ cand
                        dots[i] = -2.0
                        mi = float(np.max(dots))
                        if mi < mrow[i] - 1e-4 and mi < best_val:
                            x_best[i] = cand
                            # re-evaluate
                            nv_ = _max_cos(x_best)
                            if nv_ < best_val:
                                best_val = nv_
                                x_cur = x_best.copy()
                                cur_val = best_val
                            break
            except Exception:
                pass
    # final squeeze of best with remaining time
    try:
        remain = T_END - time.time()
        if remain > 1.0:
            x_best = lp_polish(x_best, time.time() + min(remain, 8.0), r0=0.001)
    except Exception:
        pass
    x_best = _norm_rows(x_best)
    return x_best

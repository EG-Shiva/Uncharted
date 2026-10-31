import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

_TRIU = None
_IU_I = None
_IU_J = None

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _ensure_triu(n):
    global _TRIU, _IU_I, _IU_J
    if _TRIU is None or _TRIU[0].shape[0] != n * (n - 1) // 2:
        _TRIU = np.triu_indices(n, 1)
        _IU_I = _TRIU[0].astype(np.int64)
        _IU_J = _TRIU[1].astype(np.int64)
    return _TRIU

def lp_polish(x, t_end, r0=0.003):
    x = _norm_rows(x.copy())
    n, d = x.shape
    iu = _ensure_triu(n)
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
        p_all = int(np.sum(sel))
        if p_all == 0:
            radius *= 0.5
            continue
        idx_all = np.where(sel)[0]
        if p_all > 2500:
            part = np.argpartition(c[idx_all], -2500)[-2500:]
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
        y = _norm_rows(x + dx)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, steps, beta, lr):
    x = _norm_rows(x.copy())
    for _ in range(int(steps)):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        proj = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - proj * x
        step = lr * gt
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.02
        if np.any(over):
            scale = np.ones_like(sn)
            scale[over] = 0.02 / (sn[over] + 1e-15)
            step = step * scale
        x = _norm_rows(x - step)
    return x

def lbfgs_smooth(x_init, tau=0.008, maxiter=120):
    x_init = _norm_rows(x_init.copy())
    n, d = x_init.shape
    iu = _ensure_triu(n)
    I = _IU_I
    J = _IU_J
    y0 = x_init.ravel().copy()
    def fun(yflat):
        Y = yflat.reshape(n, d)
        nr = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        nr[nr < 1e-12] = 1e-12
        X = Y / nr
        G = X @ X.T
        dots = G[iu]
        m = float(np.max(dots))
        e = np.exp((dots - m) / tau)
        s = float(np.sum(e))
        if s <= 0:
            s = 1e-300
        f = m + tau * np.log(s)
        w = e / s
        gx = np.zeros((n, d), dtype=np.float64)
        # accumulate: gx[I] += w*X[J], gx[J] += w*X[I]
        np.add.at(gx, I, w[:, None] * X[J])
        np.add.at(gx, J, w[:, None] * X[I])
        dot = np.sum(gx * X, axis=1, keepdims=True)
        gy = (gx - dot * X) / nr
        return float(f), gy.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        return _norm_rows(Y)
    except Exception:
        return x_init

def hole_candidate(x, idx, rng, n_sample=12000, beta=90.0):
    n, d = x.shape
    C = rng.standard_normal(size=(n_sample, d))
    C = _norm_rows(C)
    D = C @ x.T
    # ignore self column
    if 0 <= idx < n:
        D[:, idx] = -2.0
    m_c = np.max(D, axis=1)
    order = np.argpartition(m_c, 5)[:5]
    best_c = C[order[0]]
    best_v = float(m_c[order[0]])
    # refine top candidates with projected gradient on LSE
    for o in order:
        c = C[o].copy()
        for _ in range(80):
            dots = x @ c
            dots[idx] = -10.0
            m = float(np.max(dots))
            w = np.exp(beta * (dots - m))
            w[idx] = 0.0
            sw = float(np.sum(w))
            if sw <= 0:
                break
            w = w / sw
            g = w @ x  # (d,)
            g = g - np.dot(g, c) * c
            gn = float(np.linalg.norm(g))
            if gn > 1e-12:
                g = g / gn
                step = 0.05
                c = c - step * g
                c = c / (np.linalg.norm(c) + 1e-15)
            else:
                break
        dots = x @ c
        dots[idx] = -2.0
        v = float(np.max(dots))
        if v < best_v:
            best_v = v
            best_c = c
    return best_c.copy()

def _row_maxes(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng = np.random.default_rng(7)
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
    try:
        x_best = lp_polish(x_best, min(time.time() + 6.0, T_END - 12.0), r0=0.002)
        best_val = _max_cos(x_best)
    except Exception:
        pass
    x_cur = x_best.copy()
    cur_val = best_val
    temp = 0.0025
    it = 0
    sigmas = [0.005, 0.01, 0.02, 0.05, 0.10, 0.22]
    while time.time() < T_END - 3.0:
        it += 1
        r = rng.random()
        y = None
        y_mode = ""
        try:
            if r < 0.28:
                sigma = float(sigmas[int(rng.integers(0, len(sigmas)))])
                y = x_cur + sigma * rng.standard_normal(size=(n, d))
                y = _norm_rows(y)
                y = smooth_relax(y, steps=70, beta=65.0, lr=0.008)
                y_mode = "shake"
            elif r < 0.42:
                k = int(rng.integers(1, 5))
                y = x_cur.copy()
                mrow, _ = _row_maxes(x_cur)
                if rng.random() < 0.65:
                    order = np.argsort(-mrow)
                    pool = order[:max(14, 3 * k)]
                    idx = rng.choice(pool, size=k, replace=False)
                else:
                    idx = rng.choice(n, size=k, replace=False)
                y[idx] = rng.standard_normal(size=(k, d))
                y = _norm_rows(y)
                y = smooth_relax(y, steps=60, beta=60.0, lr=0.008)
                y_mode = "subset"
            elif r < 0.56:
                # L-BFGS from small shake with two taus
                sigma = float(10.0 ** rng.uniform(-3.0, -1.3))
                y = _norm_rows(x_cur + sigma * rng.standard_normal(size=(n, d)))
                tau = float(rng.choice([0.015, 0.008, 0.004]))
                mi = 60 if tau > 0.01 else 90
                y = lbfgs_smooth(y, tau=tau, maxiter=mi)
                # light smooth to stabilise
                y = smooth_relax(y, steps=30, beta=80.0, lr=0.005)
                y_mode = "lbfgs"
            elif r < 0.70:
                # hole reinsertion for worst points
                mrow, _ = _row_maxes(x_cur)
                order = np.argsort(-mrow)
                k = int(rng.integers(1, 3))
                y = x_cur.copy()
                for t_ in range(k):
                    idx = int(order[t_ if rng.random() < 0.8 else int(rng.integers(0, 10))])
                    hc = hole_candidate(y, idx, rng, n_sample=10000)
                    y[idx] = hc
                y = _norm_rows(y)
                y = smooth_relax(y, steps=50, beta=70.0, lr=0.007)
                y_mode = "hole"
            elif r < 0.82:
                G = x_cur @ x_cur.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                ii, jj = flat // n, flat % n
                y = x_cur.copy()
                eps = float(rng.uniform(0.015, 0.07))
                vi = x_cur[ii] - x_cur[jj]
                vj = x_cur[jj] - x_cur[ii]
                vi = vi - np.dot(vi, x_cur[ii]) * x_cur[ii]
                vj = vj - np.dot(vj, x_cur[jj]) * x_cur[jj]
                ni = float(np.linalg.norm(vi)) + 1e-12
                nj = float(np.linalg.norm(vj)) + 1e-12
                y[ii] = y[ii] + eps * vi / ni
                y[jj] = y[jj] + eps * vj / nj
                y = y + 0.0025 * rng.standard_normal(size=(n, d))
                y = _norm_rows(y)
                y = smooth_relax(y, steps=60, beta=75.0, lr=0.008)
                y_mode = "nudge"
            elif r < 0.92:
                # rattler shuffle
                mrow, _ = _row_maxes(x_cur)
                thresh = float(np.max(mrow)) - 0.0025
                ratt = np.where(mrow < thresh)[0]
                y = x_cur.copy()
                if ratt.size > 0:
                    k = min(int(ratt.size), int(rng.integers(1, max(2, min(10, ratt.size + 1)))))
                    pick = rng.choice(ratt, size=k, replace=False)
                    # half random, half hole-optimised
                    for idx in pick:
                        if rng.random() < 0.5:
                            y[idx] = rng.standard_normal(size=(d,))
                        else:
                            y[idx] = hole_candidate(y, int(idx), rng, n_sample=6000)
                    y = _norm_rows(y)
                    y = smooth_relax(y, steps=50, beta=70.0, lr=0.007)
                else:
                    sigma = 0.01
                    y = _norm_rows(x_cur + sigma * rng.standard_normal(size=(n, d)))
                y_mode = "rattler"
            else:
                # fresh staged restart (rare)
                y = _norm_rows(rng.standard_normal(size=(n, d)))
                y = smooth_relax(y, steps=250, beta=22.0, lr=0.02)
                y = lbfgs_smooth(y, tau=0.012, maxiter=60)
                y = smooth_relax(y, steps=200, beta=65.0, lr=0.01)
                y_mode = "fresh"
        except Exception:
            continue
        if y is None:
            continue
        budget = 1.7
        if y_mode in ("fresh",):
            budget = 2.6
        elif y_mode in ("lbfgs", "hole"):
            budget = 2.0
        t_end = min(time.time() + budget, T_END - 3.0)
        if t_end - time.time() < 0.25:
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
            if v2 < cur_val or rng.random() < np.exp((cur_val - v2) / max(temp, 1e-4)):
                x_cur = y2.copy()
                cur_val = v2
            if cur_val > best_val + 0.012:
                x_cur = x_best.copy()
                cur_val = best_val
        temp = max(0.00035, temp * 0.995)
        if it % 22 == 0 and time.time() < T_END - 6.0:
            try:
                mrow, _ = _row_maxes(x_best)
                thresh = best_val - 0.002
                ratt = np.where(mrow < thresh)[0]
                if ratt.size > 0:
                    for _t in range(25):
                        i = int(rng.choice(ratt))
                        cand = _norm_rows(rng.standard_normal(size=(1, d)))[0]
                        dots = x_best @ cand
                        dots[i] = -2.0
                        mi = float(np.max(dots))
                        if mi < mrow[i] - 1e-4 and mi < best_val:
                            x_best[i] = cand
                            nv_ = _max_cos(x_best)
                            if nv_ < best_val:
                                best_val = nv_
                                x_cur = x_best.copy()
                                cur_val = best_val
                            break
            except Exception:
                pass
    try:
        remain = T_END - time.time()
        if remain > 1.2:
            # final L-BFGS squeeze then LP
            try:
                x_best = lbfgs_smooth(x_best, tau=0.004, maxiter=80)
            except Exception:
                pass
            x_best = lp_polish(x_best, time.time() + min(remain - 0.2, 7.0), r0=0.001)
    except Exception:
        pass
    x_best = _norm_rows(x_best)
    return x_best

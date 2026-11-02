import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lbfgs_refine(X0, alphas, iters):
    n, d = X0.shape
    Y = X0.copy()
    for alpha, mit in zip(alphas, iters):
        # closure for this alpha
        def fun(Yf, _alpha=alpha):
            Ym = Yf.reshape(n, d)
            norms = np.linalg.norm(Ym, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = Ym / norms
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            # exp
            E = np.exp(_alpha * (G - M))
            np.fill_diagonal(E, 0.0)
            Ssum = float(np.sum(E))
            S = Ssum * 0.5
            if S <= 0 or not np.isfinite(S):
                f = M
                grad_X = np.zeros_like(X)
            else:
                f = M + np.log(S) / _alpha
                W = E / S  # symmetric, zero diag
                grad_X = W @ X
            dots = np.sum(grad_X * X, axis=1, keepdims=True)
            grad_Y = (grad_X - dots * X) / norms
            if not np.all(np.isfinite(grad_Y)):
                grad_Y = np.zeros_like(grad_Y)
            return f, grad_Y.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mit), "ftol": 1e-14, "gtol": 1e-9, "maxfun": 100000})
            Y = res.x.reshape(n, d)
            Y = _norm_rows(Y)
        except Exception:
            break
    return _norm_rows(Y)

def _find_hole(Y, rng, n_cand=4000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = _norm_rows(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, seconds, t0=None):
    if t0 is None:
        t0 = time.time()
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    TIME_LIMIT = 172.0
    SEARCH_BUDGET = 148.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal(size=(n, d)))
        best_score = _max_cos(best)
    cur = best.copy()
    cur_score = best_score
    trial = 0
    T0 = 0.002
    while time.time() - t_start < SEARCH_BUDGET:
        trial += 1
        if trial % 30 == 0:
            cur = best.copy()
            cur_score = best_score
        frac = (time.time() - t_start) / SEARCH_BUDGET
        T = T0 * (1.0 - frac) + 0.0003 * frac
        u = rng.random()
        prop = cur.copy()
        # default lbfgs schedule
        if u < 0.05:
            prop = _norm_rows(rng.standard_normal(size=(n, d)))
            alphas = [30.0, 150.0, 500.0]
            iters = [120, 120, 120]
        elif u < 0.35:
            sigma = 0.015 + 0.02 * rng.random()
            prop = _norm_rows(prop + sigma * rng.standard_normal(size=(n, d)))
            alphas = [50.0, 300.0, 800.0]
            iters = [80, 80, 100]
        elif u < 0.55:
            sigma = 0.06 + 0.06 * rng.random()
            prop = _norm_rows(prop + sigma * rng.standard_normal(size=(n, d)))
            alphas = [40.0, 200.0, 600.0]
            iters = [100, 100, 100]
        elif u < 0.70:
            sigma = 0.20 + 0.15 * rng.random()
            prop = _norm_rows(prop + sigma * rng.standard_normal(size=(n, d)))
            alphas = [30.0, 150.0, 500.0]
            iters = [120, 120, 120]
        elif u < 0.85:
            # hole reinsertion of worst points
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            # worst pair
            ij = np.argmax(G)
            wi, wj = divmod(int(ij), n)
            k = 1 if rng.random() < 0.6 else 2
            victims = [wi]
            if k == 2:
                victims.append(wj)
            if rng.random() < 0.25:
                victims = list(rng.choice(n, size=k, replace=False))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Y = prop[keep_mask]
            for vi in victims:
                h = _find_hole(Y, rng, n_cand=4000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm_rows(prop + 0.01 * rng.standard_normal(size=(n, d)))
            alphas = [40.0, 250.0, 700.0]
            iters = [100, 100, 100]
        else:
            k = int(rng.integers(4, 9))
            idx = rng.choice(n, size=k, replace=False)
            for ii in idx:
                v = rng.standard_normal(size=(d,))
                v /= max(np.linalg.norm(v), 1e-12)
                prop[ii] = v
            prop = _norm_rows(prop + 0.015 * rng.standard_normal(size=(n, d)))
            alphas = [30.0, 180.0, 600.0]
            iters = [110, 110, 110]
        # shrink iters if little time left
        remain_search = SEARCH_BUDGET - (time.time() - t_start)
        if remain_search < 12:
            iters = [min(v, 60) for v in iters]
        # jitter alphas slightly for diversity
        alphas = [a * (0.9 + 0.2 * rng.random()) for a in alphas]
        try:
            new = _lbfgs_refine(prop, alphas, iters)
        except Exception:
            continue
        try:
            s = _max_cos(new)
        except Exception:
            continue
        if not np.isfinite(s):
            continue
        if s < best_score - 1e-12:
            # quick squeeze with short LP before accepting as best
            try:
                if time.time() - t_start < SEARCH_BUDGET - 2:
                    new2 = _lp_polish(new, 0.8, t0=time.time())
                    s2 = _max_cos(new2)
                    if s2 < s:
                        new, s = new2, s2
            except Exception:
                pass
            best, best_score = new.copy(), s
            cur, cur_score = new.copy(), s
        elif s < cur_score:
            cur, cur_score = new.copy(), s
        else:
            try:
                if rng.random() < np.exp(-(s - cur_score) / max(T, 1e-12)):
                    cur, cur_score = new.copy(), s
            except Exception:
                pass
    # final polish
    remain = TIME_LIMIT - (time.time() - t_start)
    if remain > 6:
        try:
            best = _lbfgs_refine(best, [100.0, 500.0, 1200.0], [80, 80, 80])
        except Exception:
            pass
        remain2 = TIME_LIMIT - (time.time() - t_start)
        if remain2 > 4:
            try:
                best = _lp_polish(best, min(remain2 - 1.0, 20.0), t0=time.time())
            except Exception:
                pass
    best = _norm_rows(best)
    return best

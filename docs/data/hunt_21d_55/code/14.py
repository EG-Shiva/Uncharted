import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _adam_quench(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (E @ X) / S
        # tangent projection
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm_rows(X)
    return X

def _find_hole(Y, rng, n_cand=2000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = _norm_rows(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, deadline):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 80:
        it += 1
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0 or p > 2000:
            radius *= 0.5
            continue
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    T_END = t_start + 170.0
    T_SEARCH_END = t_start + 143.0
    rng = np.random.default_rng(7)
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
    T = 0.0012
    while time.time() < T_SEARCH_END:
        trial += 1
        if trial % 30 == 0:
            cur = best.copy()
            cur_score = best_score
        prop = cur.copy()
        mode = trial % 6
        if mode == 0 or mode == 1:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = np.max(G)
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 1 if mode == 0 else 2
            victims = np.argsort(-cnt)[:k]
            if rng.random() < 0.3:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _find_hole(Y, rng, n_cand=2000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm_rows(prop + 0.008 * rng.standard_normal(size=prop.shape))
            steps, a0, a1, lr0, lr1 = 900, 30.0, 450.0, 0.02, 0.002
        elif mode == 2:
            prop = _norm_rows(prop + 0.06 * rng.standard_normal(size=prop.shape))
            steps, a0, a1, lr0, lr1 = 700, 20.0, 380.0, 0.015, 0.002
        elif mode == 3:
            prop = _norm_rows(prop + 0.22 * rng.standard_normal(size=prop.shape))
            steps, a0, a1, lr0, lr1 = 800, 15.0, 350.0, 0.02, 0.002
        elif mode == 4:
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            for ii in idx:
                vv = rng.standard_normal(size=(d,))
                vv /= np.linalg.norm(vv)
                prop[ii] = vv
            prop = _norm_rows(prop + 0.015 * rng.standard_normal(size=prop.shape))
            steps, a0, a1, lr0, lr1 = 800, 30.0, 500.0, 0.015, 0.002
        else:
            # subset shake of 5-8 points
            k = int(rng.integers(5, 9))
            idx = rng.choice(n, size=k, replace=False)
            prop[idx] = _norm_rows(prop[idx] + 0.35 * rng.standard_normal(size=(k, d)))
            prop = _norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 700, 25.0, 400.0, 0.018, 0.002
        # shrink work if time is short
        if T_SEARCH_END - time.time() < 12:
            steps = min(steps, 400)
        new = _adam_quench(prop, steps, a0, a1, lr0, lr1)
        s = _max_cos(new)
        if s < best_score - 1e-12:
            best, best_score = new.copy(), s
            cur, cur_score = new.copy(), s
        elif s < cur_score:
            cur, cur_score = new.copy(), s
        else:
            if rng.random() < np.exp(-(s - cur_score) / T):
                cur, cur_score = new.copy(), s
        if trial > 150 and best_score < 0.0666:
            T = 0.0006
    # short high-alpha quench of global best
    if time.time() < T_END - 14:
        best = _adam_quench(best, 400, 100.0, 700.0, 0.006, 0.0008)
    # LP polish with remaining time
    if time.time() < T_END - 3:
        best = _lp_polish(best, min(T_END - 1.0, time.time() + 12.0))
    best = _norm_rows(best)
    return best

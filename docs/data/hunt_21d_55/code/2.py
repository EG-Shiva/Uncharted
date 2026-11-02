import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def anneal_relax(X0, steps, a0, a1, lr0, lr1):
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
        if S <= 0 or not np.isfinite(S):
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X = norm_rows(X)
    return X

def find_hole(Y, rng, n_cand=2000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = norm_rows(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def lp_polish(x, seconds, t0=None):
    if t0 is None:
        t0 = time.time()
    x = norm_rows(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    TIME_LIMIT = 172.0
    SEARCH_BUDGET = 148.0
    rng = np.random.default_rng(12345)
    # load init
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = norm_rows(a)
                s = max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = norm_rows(rng.standard_normal(size=(55, 21)))
        best_score = max_cos(best)
    cur = best.copy()
    cur_score = best_score
    n, d = 55, 21
    trial = 0
    T = 0.0015
    while time.time() - t_start < SEARCH_BUDGET:
        trial += 1
        # periodic restart from global best
        if trial % 25 == 0:
            cur = best.copy()
            cur_score = best_score
        prop = cur.copy()
        mode = trial % 5
        if mode == 0 or mode == 1:
            # remove overloaded point(s) and reinsert in hole
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = np.max(G)
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 1 if mode == 0 else 2
            victims = np.argsort(-cnt)[:k]
            # also randomize victim sometimes
            if rng.random() < 0.3:
                victims = rng.choice(n, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Y = prop[keep_mask]
            for vi in victims:
                h = find_hole(Y, rng, n_cand=3000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            # small jitter
            prop = prop + 0.01 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1600, 25.0, 400.0, 0.02, 0.002
        elif mode == 2:
            prop = prop + 0.07 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1200, 20.0, 350.0, 0.015, 0.002
        elif mode == 3:
            prop = prop + 0.25 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1500, 15.0, 350.0, 0.02, 0.002
        else:
            # completely randomize 1-3 points
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            for ii in idx:
                v = rng.standard_normal(size=(d,))
                v /= np.linalg.norm(v)
                prop[ii] = v
            prop = prop + 0.02 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 1400, 25.0, 450.0, 0.015, 0.002
        # adapt steps if little time left
        if time.time() - t_start > SEARCH_BUDGET - 10:
            steps = min(steps, 600)
        new = anneal_relax(prop, steps, a0, a1, lr0, lr1)
        s = max_cos(new)
        if s < best_score - 1e-12:
            best, best_score = new.copy(), s
            cur, cur_score = new.copy(), s
        elif s < cur_score:
            cur, cur_score = new.copy(), s
        else:
            if rng.random() < np.exp(-(s - cur_score) / T):
                cur, cur_score = new.copy(), s
        # opportunistic quick check: if very good, tighten T
        if trial > 200 and best_score < 0.0666:
            T = 0.0008
    # final polish of global best with remaining time
    remain = TIME_LIMIT - (time.time() - t_start)
    if remain > 5:
        # first a short high-alpha smooth polish
        best = anneal_relax(best, 800, 80.0, 600.0, 0.008, 0.001)
        remain2 = TIME_LIMIT - (time.time() - t_start)
        if remain2 > 4:
            best = lp_polish(best, min(remain2 - 1.0, 22.0), t0=time.time())
    best = norm_rows(best)
    return best

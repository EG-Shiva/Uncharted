import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
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
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def _find_hole(Y, rng, n_cand=4000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)].copy()

def _make_simplex22():
    M = np.eye(22) - np.ones((22, 22)) / 22.0
    w, V = np.linalg.eigh(M)
    E = V[:, 1:]  # 22x21, skip zero eigenvalue
    X = E * np.sqrt(22.0 / 21.0)
    return _norm(X)

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        nz = int(np.count_nonzero(sel))
        if nz == 0:
            break
        if nz > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_SEARCH_END = t0 + 148.0
    T_LP_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    x_best = best.copy()

    # algebraic seed: simplex + anti-simplex + 11 holes
    try:
        S22 = _make_simplex22()
        X44 = np.vstack([S22, -S22])
        cur = X44.copy()
        rng_h = np.random.default_rng(1)
        for _ in range(11):
            h = _find_hole(cur, rng_h, n_cand=20000)
            cur = np.vstack([cur, h[None, :]])
        simplex55 = _norm(cur)
        # quick quench of simplex seed to get competitive starting point
        sq = _adam_quench(simplex55, steps=1200, a0=40.0, a1=600.0, lr0=0.025, lr1=0.002)
        s = _max_cos(sq)
        if s < best_score:
            best_score = s
            x_best = sq.copy()
    except Exception:
        simplex55 = None

    trial = 0
    while time.time() < T_SEARCH_END:
        trial += 1
        remain = T_SEARCH_END - time.time()
        if remain < 2:
            break
        # choose steps based on remaining time
        if trial <= 4 and simplex55 is not None:
            # symmetric variants
            X0 = simplex55 + 0.01 * rng.standard_normal((n, d))
            X0 = _norm(X0)
            steps = 1000
            a0 = 30.0 + 20.0 * rng.random()
            a1 = 500.0 + 200.0 * rng.random()
            lr0 = 0.02 + 0.01 * rng.random()
            lr1 = 0.002
            Xq = _adam_quench(X0, steps, a0, a1, lr0, lr1)
        elif trial % 4 == 0:
            # pure random different basin
            X0 = _norm(rng.standard_normal((n, d)))
            steps = 800 + int(rng.integers(0, 400))
            a0 = 20.0 + 60.0 * rng.random()
            a1 = 300.0 + 500.0 * rng.random()
            lr0 = 0.015 + 0.015 * rng.random()
            lr1 = 0.001 + 0.003 * rng.random()
            Xq = _adam_quench(X0, steps, a0, a1, lr0, lr1)
        else:
            # basin hop from elite: hole-reinsert worst + kicks
            X0 = x_best.copy()
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            cnt = np.sum(G > G.max() - 1e-3, axis=1)
            score_pick = rowmax + 0.002 * cnt
            k = int(rng.integers(1, 6))  # 1..5 points
            if rng.random() < 0.75:
                victims = np.argsort(-score_pick)[:k]
            else:
                victims = rng.choice(n, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            # sequential hole fill
            Y = X0[keep_mask]
            for vi in victims:
                h = _find_hole(Y, rng, n_cand=3000)
                X0[vi] = h
                Y = np.vstack([Y, h[None, :]])
            # add kicks: global small + subset large
            sigma = 0.01 + 0.10 * rng.random()
            X0 = X0 + sigma * rng.standard_normal((n, d))
            if rng.random() < 0.5:
                sub = rng.choice(n, size=int(rng.integers(5, 15)), replace=False)
                X0[sub] += 0.15 * rng.standard_normal((len(sub), d))
            X0 = _norm(X0)
            steps = 700 + int(rng.integers(0, 500))
            a0 = 20.0 + 60.0 * rng.random()
            a1 = 300.0 + 500.0 * rng.random()
            lr0 = 0.015 + 0.015 * rng.random()
            lr1 = 0.001 + 0.003 * rng.random()
            Xq = _adam_quench(X0, steps, a0, a1, lr0, lr1)
        s = _max_cos(Xq)
        if s < best_score:
            best_score = s
            x_best = Xq.copy()

    # final short LP squeeze on elite
    try:
        if time.time() < T_LP_END:
            x_best = _lp_polish(x_best, T_LP_END)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

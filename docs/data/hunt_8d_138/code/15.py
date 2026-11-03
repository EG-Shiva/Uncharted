import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), float(g.max())


def _anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if np.count_nonzero(sel) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _void_for_point(X_other, rng, n_try=8000):
    d = X_other.shape[1]
    cand = rng.normal(size=(n_try, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_other.T
    scores = dots.max(axis=1)
    return cand[np.argmin(scores)]


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 170.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # quick squeeze of incumbent
    try:
        best = _lp_polish(best, min(deadline - 8.0, time.time() + 4.0))
        best_val = _max_cos(best)
    except Exception:
        pass

    cur = best.copy()
    cur_val = best_val
    stall = 0
    it = 0
    while time.time() < deadline - 7.0 and it < 200:
        it += 1
        now = time.time()

        if it % 17 == 0:
            # rare fresh basin
            y = rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 900, 10.0, 300.0, 0.03, 0.002)
            y = _lp_polish(y, min(deadline - 7.0, now + 4.0))
            v = _max_cos(y)
            if v < best_val - 1e-12:
                best_val = v
                best = y.copy()
                cur = y.copy()
                cur_val = v
                stall = 0
            continue

        # choose kick size cyclically
        cyc = it % 3
        if cyc == 0:
            sigma = 0.25
        elif cyc == 1:
            sigma = 0.08
        else:
            sigma = 0.02

        y = cur + sigma * rng.normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)

        mode = it % 5
        try:
            rm, gm = _row_max(y)
            if mode == 4:
                # rattler rebirth: points far below max -> move to voids
                order = np.argsort(rm)  # smallest first = loosest
                # rattler threshold
                k = 6
                chosen = order[:k]
                for idx in chosen:
                    mask = np.ones(n, dtype=bool)
                    mask[idx] = False
                    y[idx] = _void_for_point(y[mask], rng, n_try=8000)
            elif mode == 3:
                # kill worst points into voids (greedy sequential)
                k = 4
                order = np.argsort(-rm)[:k]
                for idx in order:
                    mask = np.ones(n, dtype=bool)
                    mask[idx] = False
                    y[idx] = _void_for_point(y[mask], rng, n_try=8000)
            elif mode == 1 and sigma > 0.2:
                # random re-insertion of 8 points to voids
                k = 8
                idxs = rng.choice(n, size=k, replace=False)
                for idx in idxs:
                    mask = np.ones(n, dtype=bool)
                    mask[idx] = False
                    # half void-optimal, half random for diversity
                    if rng.random() < 0.7:
                        y[idx] = _void_for_point(y[mask], rng, n_try=6000)
                    else:
                        v0 = rng.normal(size=(d,))
                        v0 /= np.linalg.norm(v0)
                        y[idx] = v0
            elif mode == 2:
                # targeted shake of overloaded points
                thr = gm - 0.01
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                cnt = (G > thr).sum(axis=1)
                order = np.argsort(-cnt)
                pool = order[:10]
                # extra kick to top 4
                ek = pool[rng.choice(len(pool), size=4, replace=False)]
                y[ek] += 0.20 * rng.normal(size=(4, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
        except Exception:
            pass

        steps = 700 if sigma > 0.05 else 400
        y = _anneal(y, steps, 20.0, 250.0, 0.02, 0.002)
        y = _lp_polish(y, min(deadline - 7.0, now + 2.2))
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
            stall = 0
        else:
            if v < cur_val:
                cur = y.copy()
                cur_val = v
                stall = 0
            else:
                stall += 1
                if stall >= 10:
                    cur = best.copy()
                    cur_val = best_val
                    stall = 0

    try:
        best = _lp_polish(best, deadline)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

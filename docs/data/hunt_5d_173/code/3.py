import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _void_fill(rem, k, rng, n_cand=3000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # max dot to remaining
    s_max = (cand @ rem.T).max(axis=1)
    chosen = []
    for _ in range(k):
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        chosen.append(pt)
        # update to keep new points apart from each other
        upd = cand @ pt
        # max of old and new
        np.maximum(s_max, upd, out=s_max)
        # avoid picking same again
        s_max[idx] = 2.0
    return np.stack(chosen, axis=0)


def _repel(x, alphas, iters_list, lrs, mu, decay, t_deadline):
    n, d = x.shape
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for s in range(int(iters)):
            if (s & 15) == 0 and time.time() > t_deadline:
                break
            G = x @ x.T
            # fixed shift 0.75 cancels in weighted average, avoids overflow
            # exp(alpha*(G-0.75)), diag killed
            W = np.exp(alpha * (G - 0.75))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-300
            A = (W @ x) / rs
            dot = (A * x).sum(axis=1, keepdims=True)
            Ft = A - dot * x
            v = mu * v + (1.0 - mu) * Ft
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            lr *= decay
        if time.time() > t_deadline:
            break
    return x


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    deadline_search = deadline - 30.0
    rng = np.random.RandomState(0)

    # load starting points
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (173, 5):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if len(cands) == 0:
        tmp = rng.standard_normal((173, 5))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    # also random candidate
    tmp2 = rng.standard_normal((173, 5))
    tmp2 /= np.linalg.norm(tmp2, axis=1, keepdims=True)
    cands.append(tmp2)

    best = None
    best_val = 2.0
    for c in cands[:2]:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    n, d = 173, 5
    trial = 0
    # quick evaluate random repel on random start to get diversity early
    while time.time() < deadline_search:
        trial += 1
        r = rng.rand()
        x_try = best.copy()
        if r < 0.65:
            k = int(rng.randint(2, 7))  # 2..6
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            mi = G.max(axis=1)
            order = np.argsort(-mi)
            pool = order[:25]
            sel = rng.choice(pool, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem = x_try[mask]
            try:
                new_pts = _void_fill(rem, k, rng, n_cand=3000)
            except Exception:
                new_pts = rng.standard_normal((k, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x_try[sel] = new_pts
            sigma = 0.005 + 0.015 * rng.rand()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            sigma = 0.03 + 0.12 * rng.rand()
            # occasionally very large kick
            if rng.rand() < 0.1:
                sigma = 0.25 + 0.15 * rng.rand()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        a1 = 30.0 + rng.rand() * 20.0
        a2 = 80.0 + rng.rand() * 60.0
        a3 = 180.0 + rng.rand() * 140.0
        alphas = [a1, a2, a3]
        iters_list = [200, 250, 350]
        # adapt iters if little time left
        remain = deadline_search - time.time()
        if remain < 10:
            iters_list = [120, 150, 200]
        l1 = 0.06 + rng.rand() * 0.10
        l2 = 0.03 + rng.rand() * 0.05
        l3 = 0.015 + rng.rand() * 0.025
        lrs = [l1, l2, l3]
        mu = 0.5 + rng.rand() * 0.35
        decay = 0.997 + rng.rand() * 0.002  # 0.997..0.999
        if decay > 0.9995:
            decay = 0.9995

        x_opt = _repel(x_try, alphas, iters_list, lrs, mu, decay, deadline_search)
        v = _max_cos(x_opt)
        if v < best_val - 1e-12:
            best = x_opt
            best_val = v
        # early exit if big win
        if best_val < 0.74:
            break

    # final polish with remaining time
    remain = deadline - time.time() - 2.0
    if remain < 5.0:
        remain = 5.0
    if remain > 35.0:
        remain = 35.0
    try:
        best = _lp_polish(best, remain)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

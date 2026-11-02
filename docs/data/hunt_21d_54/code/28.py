import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    x = np.nan_to_num(x)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    if seconds <= 0.05:
        return x
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2200:
            idx = np.argpartition(-c[sel], 2200)[:2200]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2200
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def _single_opt(v, Y, steps=160, lr0=0.08):
    v = v / (np.linalg.norm(v) + 1e-12)
    m = Y.shape[0]
    if m == 0:
        return v
    for s in range(steps):
        dots = Y @ v
        mx = float(dots.max())
        T = 0.015 * (1.0 - s / steps) + 0.002
        E = (dots - mx) / T
        np.clip(E, -20, 0, out=E)
        W = np.exp(E) + 1e-12
        grad = W @ Y
        grad -= float(grad @ v) * v
        gn = float(np.linalg.norm(grad)) + 1e-12
        grad /= gn
        step = lr0 * (1.0 - 0.6 * s / steps)
        v = v - step * grad
        n = float(np.linalg.norm(v))
        if n == 0:
            break
        v /= n
    return v

def _largest_hole(Y, rng, K=20000):
    d = Y.shape[1]
    C = rng.normal(size=(K, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    S = C @ Y.T
    scores = S.max(axis=1)
    i = int(np.argmin(scores))
    return C[i]

def _rebuild(base, cluster, rng, budget):
    t0 = time.time()
    n = base.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(cluster, dtype=int)] = False
    Y = base[mask].copy()
    k = int(n - Y.shape[0])
    if k <= 0:
        return base
    # LP-polish remainder to gain slack (~25% of budget)
    rem = budget * 0.25
    if time.time() - t0 < budget - 0.5:
        Y = _lp_polish(Y, max(0.5, min(rem, budget - (time.time() - t0) - k * 0.8 - 0.3)))
    for t in range(k):
        if time.time() - t0 >= budget - 0.2:
            break
        v = _largest_hole(Y, rng, K=20000)
        v = _single_opt(v, Y, steps=140, lr0=0.07)
        Y = np.vstack([Y, v[None, :]])
        Y = _proj(Y)
        # brief LP repair after each insertion
        left = budget - (time.time() - t0)
        # reserve ~0.7s per remaining insertion
        per = min(1.0, max(0.3, left / (k - t)))
        if per > 0.25 and time.time() - t0 < budget - 0.2:
            Y = _lp_polish(Y, per)
    return _proj(Y)

def solve():
    T0 = time.time()
    LIMIT = 160.0
    def left():
        return LIMIT - (time.time() - T0)
    rng = np.random.default_rng(20260717)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    best = b0.copy()
    bv = _max_cos(best)
    # light touch-up of incumbent (keep short)
    if left() > 30:
        best = _lp_polish(best, min(3.0, left() - 27.0))
        bv = _max_cos(best)
    # contact structure of incumbent
    G = best @ best.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    cnts = (G > m0 - 1e-9).sum(axis=1)
    order = np.argsort(-cnts)
    n_try = 7
    for a in range(n_try):
        if left() < 24:
            break
        k = 5 + (a % 6)  # 5..10
        seed = int(order[a % 7])
        # strongest mutual contacts of seed
        neigh = np.argsort(-G[seed])
        neigh = [int(j) for j in neigh if int(j) != seed][:k-1]
        cluster = [seed] + neigh
        # insertion-order diversity: different rng stream per attempt
        rng_a = np.random.default_rng(1000 + a * 7919)
        budget = min(12.0, max(5.0, left() - 13.0))
        # ensure we leave room for remaining tries + final polish
        # cap so we don't eat everything on first tries
        if n_try - a > 1 and left() - budget < 12:
            budget = max(5.0, (left() - 12) / 2.0)
        if budget < 4.0:
            break
        cand = _rebuild(best, cluster, rng_a, budget)
        v = _max_cos(cand)
        if v < 0.070 and left() > 14:
            cand = _lp_polish(cand, min(4.0, left() - 11.0))
            v = _max_cos(cand)
        if v < bv - 1e-12:
            best, bv = cand, v
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            cnts = (G > m0 - 1e-9).sum(axis=1)
            order = np.argsort(-cnts)
        if left() < 14:
            break
    if left() > 2.0:
        best = _lp_polish(best, max(0.5, left() - 1.0))
    return _proj(best)

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    return np.nan_to_num(x)

def _maxcos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _maxcos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20 * radius, 1e-9)
        sel = c > thr
        I = iu0[0][sel]; J = iu0[1][sel]
        csv = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            k = 2500
            idx = np.argpartition(-csv, k)[:k]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.concatenate([ci, cj, np.full(p, n * d)])
        # interleave per row: need order [xJ, xI, -1] per row
        vals = np.empty(p * (2 * d + 1))
        vals[0::2*d+1] = 0  # placeholder
        # build correctly row-major
        V = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1)
        # rows already repeat, cols must be row-major too
        cols2 = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals2 = V.ravel()
        try:
            A_ub = coo_matrix((vals2, (rows, cols2)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _contacts(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - 1e-9).sum(axis=1)
    return C, m, G

def _kick(x, rng, strength, k):
    n, d = x.shape
    cnts, m, G = _contacts(x)
    low = np.argsort(cnts)
    high = np.argsort(-cnts)
    targets = set()
    # half from high (overloaded core), half from low (periphery)
    nh = k // 2 + 1
    nl = k // 2 + 1
    for i in range(nh):
        targets.add(int(high[i % n]))
    for i in range(nl):
        targets.add(int(low[i % n]))
    while len(targets) < k:
        targets.add(int(rng.integers(0, n)))
    targets = list(targets)[:k]
    y = x.copy()
    for i in targets:
        v = rng.normal(size=d)
        v -= (v @ y[i]) * y[i]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v /= nv
        ang = strength * (0.5 + rng.random())
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
    # small global tangent jitter
    nz = rng.normal(size=(n, d)) * (strength * 0.15)
    nz -= (nz * y).sum(axis=1, keepdims=True) * y
    y = _proj(y + nz)
    return y

def _spread(x, steps, lr):
    y = x.copy()
    n, d = y.shape
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        T = 0.02 - 0.015 * (s / max(1, steps))
        T = max(T, 0.004)
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        grad -= (grad * y).sum(axis=1, keepdims=True) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= gn
        step = lr * (1.0 - 0.5 * s / steps)
        y = _proj(y - step * grad)
    return y

def solve():
    T0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _proj(np.array(x0, dtype=float))
    best = x0
    best_val = _maxcos(best)
    el = lambda: time.time() - T0
    # brief initial polish
    if el() < LIMIT - 60:
        best = _lp_polish(best, min(7.0, LIMIT - 60 - el()))
        best_val = _maxcos(best)
    strengths = [0.06, 0.15, 0.30, 0.04, 0.10, 0.22]
    ks = [6, 9, 12, 5, 8, 10]
    it = 0
    while el() < LIMIT - 28:
        s = strengths[it % len(strengths)]
        k = ks[it % len(ks)]
        it += 1
        cand = _kick(best, rng, strength=s, k=int(k))
        cand = _spread(cand, steps=60, lr=0.02)
        budget = min(5.0, (LIMIT - 28 - el()) / 1.2)
        if budget < 1.0:
            break
        cand = _lp_polish(cand, budget)
        v = _maxcos(cand)
        if v < best_val - 1e-12:
            best, best_val = cand, v
        # occasional second try from same kick with different spread
        if it % 3 == 0 and el() < LIMIT - 32:
            cand2 = _spread(_kick(best, rng, strength=s * 0.6, k=int(k)), steps=40, lr=0.03)
            b2 = min(4.0, LIMIT - 28 - el())
            if b2 >= 1.0:
                cand2 = _lp_polish(cand2, b2)
                v2 = _maxcos(cand2)
                if v2 < best_val - 1e-12:
                    best, best_val = cand2, v2
    if el() < LIMIT - 1:
        best = _lp_polish(best, max(0.5, LIMIT - el() - 0.5))
    return _proj(best)

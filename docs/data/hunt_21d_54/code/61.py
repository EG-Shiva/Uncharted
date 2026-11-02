import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    return x

def _quench(y, steps, T0, T1, lr0, lr1, mu=0.88):
    n, d = y.shape
    vel = np.zeros_like(y)
    # geometric schedules in log space
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = np.exp(lt0 + (lt1 - lt0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -25.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ y) / rs
        grad -= (np.sum(grad * y, axis=1, keepdims=True)) * y
        vel = mu * vel + (1.0 - mu) * grad
        y = y - lr * vel
        # normalize
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
    return y

def _lp_polish(x, t_deadline):
    x = _proj(x.copy())
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_deadline - 0.4 and radius > 1e-14:
        remain = t_deadline - time.time()
        if remain < 0.5:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            idx = np.argpartition(-c[sel], 1500)[:1500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = 0  # placeholder
        # build cols vectorized
        base = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        cols = base
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(2.5, max(0.5, remain - 0.3))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 176.0
    deadline = T0 + LIMIT
    rng = np.random.default_rng(7)
    try:
        b0 = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        b0 = np.array(np.load("record.npy"), dtype=float)
    b0 = np.nan_to_num(b0, nan=0.0, posinf=0.0, neginf=0.0)
    if b0.shape != (54, 21):
        b0 = rng.normal(size=(54, 21))
    b0 = _proj(b0)
    try:
        r0 = _proj(np.array(np.load("record.npy"), dtype=float))
        if r0.shape != (54, 21):
            r0 = b0
    except Exception:
        r0 = b0
    best = b0.copy()
    bv = _mc(best)
    rv = _mc(r0)
    if rv < bv:
        best, bv = r0.copy(), rv
    n, d = 54, 21
    # quick evaluate: keep best as fallback
    # Phase 1: diverse quench hopping, NO LP, until deadline-25s
    phase1_end = deadline - 25.0
    it = 0
    # precompute for speed: random restart quench params varied
    while time.time() < phase1_end:
        it += 1
        mode = it % 10
        y = None
        if mode == 0:
            # pure random restart with sharp quench
            y = _proj(rng.normal(size=(n, d)))
            y = _quench(y, steps=900, T0=0.02, T1=0.0012, lr0=0.35, lr1=0.03)
        elif mode == 1:
            # small jitter of best + low-T quench
            sig = 0.012
            e = rng.normal(size=(n, d)) * sig
            e -= (np.sum(e * best, axis=1, keepdims=True)) * best
            y = _proj(best + e)
            y = _quench(y, steps=700, T0=0.008, T1=0.0008, lr0=0.20, lr1=0.02)
        elif mode == 2:
            # medium jitter
            sig = 0.05
            e = rng.normal(size=(n, d)) * sig
            e -= (np.sum(e * best, axis=1, keepdims=True)) * best
            y = _proj(best + e)
            y = _quench(y, steps=800, T0=0.015, T1=0.001, lr0=0.30, lr1=0.03)
        elif mode == 3:
            # large jitter / partial resample: replace k points randomly
            y = best.copy()
            k = 3 + (it % 4)  # 3..6
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = _proj(rng.normal(size=(k, d)))
            y = _quench(y, steps=1000, T0=0.025, T1=0.0012, lr0=0.35, lr1=0.03)
        elif mode == 4:
            # teleport worst point to best void (best-of-many random, no refine)
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            pm = G.max(axis=1)
            w = int(np.argmax(pm))
            y = best.copy()
            K = 2500
            C = _proj(rng.normal(size=(K, d)))
            A = np.delete(best, w, axis=0)
            sc = (C @ A.T).max(axis=1)
            bi = int(np.argmin(sc))
            y[w] = C[bi]
            y = _quench(y, steps=900, T0=0.012, T1=0.0009, lr0=0.28, lr1=0.025)
        elif mode == 5:
            # teleport 2 worst to voids
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            pm = G.max(axis=1)
            order = np.argsort(-pm)[:2]
            y = best.copy()
            for w in order:
                K = 1500
                C = _proj(rng.normal(size=(K, d)))
                A = np.delete(y, w, axis=0)
                sc = (C @ A.T).max(axis=1)
                y[w] = C[int(np.argmin(sc))]
            y = _quench(y, steps=1000, T0=0.015, T1=0.001, lr0=0.32, lr1=0.03)
        elif mode == 6:
            # global shake + high-T anneal (basin escape)
            sig = 0.10
            e = rng.normal(size=(n, d)) * sig
            e -= (np.sum(e * best, axis=1, keepdims=True)) * best
            y = _proj(best + e)
            y = _quench(y, steps=1100, T0=0.035, T1=0.002, lr0=0.40, lr1=0.04)
        elif mode == 7:
            # random restart with different sharpness (Riesz-like diversity)
            y = _proj(rng.normal(size=(n, d)))
            y = _quench(y, steps=1100, T0=0.04, T1=0.002, lr0=0.40, lr1=0.04)
        elif mode == 8:
            # single-point random kick large angle + quench
            y = best.copy()
            w = int(rng.integers(0, n))
            v = rng.normal(size=d)
            v -= float(v @ y[w]) * y[w]
            v /= (np.linalg.norm(v) + 1e-12)
            ang = 0.3 + rng.random() * 0.8
            y[w] = np.cos(ang) * y[w] + np.sin(ang) * v
            y = _proj(y)
            y = _quench(y, steps=700, T0=0.010, T1=0.0009, lr0=0.25, lr1=0.025)
        else:
            # quench best itself harder (polish without LP)
            y = _quench(best.copy(), steps=600, T0=0.005, T1=0.0005, lr0=0.15, lr1=0.015)
        v = _mc(y)
        if v < bv - 1e-12:
            best, bv = y, v
        # time guard
        if time.time() >= phase1_end:
            break
    # Phase 2: single short LP squeeze on best (and maybe second chance)
    if time.time() < deadline - 2.0:
        t_d = deadline - 1.0
        best = _lp_polish(best, t_d)
        bv = _mc(best)
    best = _proj(best)
    # safety: if somehow worse than loaded best, return loaded best
    try:
        orig = _proj(np.array(np.load("best.npy"), dtype=float))
        if _mc(orig) < _mc(best):
            best = orig
    except Exception:
        pass
    return _proj(best)

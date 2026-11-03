import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench(X, steps, T0, T1, lr):
    n, d = X.shape
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 8e-4:
            T = 8e-4
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        # linear lr decay to 0.3x
        frac = t / steps
        lrt = lr * (1.0 - 0.7 * frac)
        X = X - lrt * step
        X = _norm(X)
    return X

def _push(X, iters=80, eta0=0.02, margin=0.008):
    X = X.copy()
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        thr = M - margin
        W = (G > thr).astype(float)
        # avoid self
        F = W @ X
        dot = (F * X).sum(axis=1, keepdims=True)
        Ft = F - dot * X
        # normalize force scale
        eta = eta0 * (1.0 - 0.8 * it / max(iters, 1))
        X = X - eta * Ft
        X = _norm(X)
    return X

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _hole_greedy(X, rng, C=15000, passes=1):
    n, d = X.shape
    X = X.copy()
    cur = _mc(X)
    for _ in range(passes):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)[:20]
        for vv in order:
            vv = int(vv)
            Xt = np.delete(X, vv, axis=0)
            cand = _norm(rng.normal(size=(C, d)))
            md = (cand @ Xt.T).max(axis=1)
            bi = int(np.argmin(md))
            y = cand[bi]
            # value if replaced
            nv = float((Xt @ y).max())
            if nv < cur:
                X[vv] = y
                cur = _mc(X)
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _mc(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _mc(best)
    # exploration with large kicks + cheap quenches
    EXPL_END = DEAD - 32.0
    trial = 0
    while time.time() < EXPL_END:
        trial += 1
        r = rng.random()
        if r < 0.45:
            sigma = 0.15 + rng.random() * 0.30  # 0.15-0.45 global shake
            Xtry = best + sigma * rng.normal(size=(n, d))
            Xtry = _norm(Xtry)
        elif r < 0.80:
            k = int(rng.integers(8, 36))  # block reinit
            Xtry = best.copy()
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = _norm(rng.normal(size=(k, d)))
        elif r < 0.90:
            # antipodal-structured fresh: 77 + opposites with noise
            h = 77
            H = _norm(rng.normal(size=(h, d)))
            Xtry = np.vstack([H, -H])
            Xtry = _norm(Xtry + 0.05 * rng.normal(size=(n, d)))
        else:
            Xtry = _norm(rng.normal(size=(n, d)))
        T0 = 0.012 + rng.random() * 0.018
        T1 = 0.001 + rng.random() * 0.0025
        lr = 0.013 + rng.random() * 0.012
        s1 = int(rng.integers(140, 240))
        Xtry = _quench(Xtry, steps=s1, T0=T0, T1=max(T1 * 2.0, 0.002), lr=lr)
        # hard push to separate near-contacts
        Xtry = _push(Xtry, iters=int(rng.integers(40, 90)), eta0=0.015 + rng.random() * 0.015, margin=0.006 + rng.random() * 0.008)
        Xtry = _quench(Xtry, steps=int(rng.integers(80, 150)), T0=0.004 + rng.random() * 0.003, T1=0.0009 + rng.random() * 0.0008, lr=0.010 + rng.random() * 0.008)
        v = _mc(Xtry)
        if v < best_val:
            best_val = float(v)
            best = Xtry.copy()
        # early exit if big improvement to spend time polishing
        if best_val < 0.395 and time.time() > t0 + 60:
            break
    # cheap exact-hole greedy on worst points
    try:
        if time.time() < DEAD - 28:
            best = _hole_greedy(best, rng, C=12000, passes=1)
            best_val = _mc(best)
    except Exception:
        pass
    # longer anneal on best
    try:
        if time.time() < DEAD - 26:
            best = _quench(best, steps=500, T0=0.008, T1=0.0008, lr=0.012)
            best = _push(best, iters=60, eta0=0.012, margin=0.005)
            best = _quench(best, steps=300, T0=0.003, T1=0.0008, lr=0.008)
            best_val = _mc(best)
    except Exception:
        pass
    remain = DEAD - time.time() - 1.0
    if remain < 4:
        remain = 4
    if remain > 28:
        remain = 28
    try:
        best = _lp_polish(best, seconds=remain, deadline=DEAD - 0.5)
    except Exception:
        pass
    best = _norm(best)
    return best

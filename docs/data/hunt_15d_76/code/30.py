import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(X, n_iter, T0, T1, lr0, lr1, t_end):
    n, d = X.shape
    for k in range(n_iter):
        if k % 400 == 0 and time.time() > t_end:
            break
        f = k / max(1, n_iter - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # limit step
        X = X - lr * Gt
        X = _norm(X)
    return X

def _refine_hole(v, Y, n_iter, T0, lr0):
    v = v / np.linalg.norm(v)
    for k in range(n_iter):
        f = k / max(1, n_iter - 1)
        Tk = T0 * (0.15 ** f)
        lrk = lr0 * (0.15 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / Tk)
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        g = g - float(g @ v) * v
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv == 0:
            break
        v /= nv
    return v

def _contact_polish(X, steps, thr0, thr1, lr0, lr1, t_end):
    n, d = X.shape
    for k in range(steps):
        if k % 300 == 0 and time.time() > t_end:
            break
        f = k / max(1, steps - 1)
        thr = thr0 * (thr1 / thr0) ** f if thr0 > 0 else thr1
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > m - thr).astype(np.float64)
        # symmetric repulsion
        F = M @ X
        # remove radial part
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        X = X - lr * Ft
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    t_end = t0 + 165.0
    rng0 = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = None
    cands = []
    for c in (Xb, Xr):
        if c is not None and c.shape == (76, 15):
            c = _norm(c.astype(float))
            if np.all(np.isfinite(c)):
                cands.append(c)
    if not cands:
        R = rng0.normal(size=(76, 15))
        return _norm(R)
    best = None
    bv = 1e9
    for c in cands:
        v = _maxcos(c)
        if v < bv:
            bv = v
            best = c.copy()
    Xbest = best.copy()
    n, d = 76, 15
    sigmas = [0.02, 0.06, 0.12, 0.25, 0.04, 0.09, 0.18, 0.35]
    trial = 0
    # precompute nothing heavy
    while time.time() < t_end - 6 and trial < 120:
        if time.time() > t_end - 6:
            break
        rng = np.random.default_rng(7777 + trial * 1009)
        sigma = sigmas[trial % len(sigmas)]
        Xt = Xbest.copy()
        # contact / row-max analysis for targeting
        G = Xt @ Xt.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        rowmax = G.max(axis=1)
        cnt = (G > mcur - 2e-4).sum(axis=1)
        order = np.argsort(-rowmax - 0.001 * cnt)
        # number to reinsert
        if sigma > 0.2:
            k_re = 4
        elif sigma > 0.1:
            k_re = 3
        elif sigma > 0.05:
            k_re = 2
        else:
            k_re = 2
        # pick indices: worst + random
        idxs = []
        idxs.append(int(order[0]))
        if k_re >= 2:
            idxs.append(int(order[1 + (trial % 5)]))
        while len(idxs) < k_re:
            r = int(rng.integers(0, n))
            if r not in idxs:
                idxs.append(r)
        # gaussian shake first
        Xt = Xt + sigma * rng.normal(size=(n, d))
        Xt = _norm(Xt)
        # hole-optimized reinsertion sequential
        for idx in idxs:
            if time.time() > t_end - 6:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = Xt[mask]
            NS = 2500
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ Y.T).max(axis=1)
            bi = int(np.argmin(scores))
            v0 = C[bi]
            v0 = _refine_hole(v0, Y, n_iter=180, T0=0.008, lr0=0.08)
            Xt[idx] = v0
        # anneal params by sigma
        if sigma < 0.05:
            Xt = _anneal(Xt, n_iter=1800, T0=0.010, T1=0.0008, lr0=3.0, lr1=0.12, t_end=t_end - 4)
            Xt = _contact_polish(Xt, steps=700, thr0=3e-3, thr1=2e-4, lr0=0.6, lr1=0.05, t_end=t_end - 4)
        elif sigma < 0.13:
            Xt = _anneal(Xt, n_iter=2200, T0=0.016, T1=0.0008, lr0=4.5, lr1=0.10, t_end=t_end - 4)
            Xt = _contact_polish(Xt, steps=800, thr0=5e-3, thr1=2e-4, lr0=0.7, lr1=0.05, t_end=t_end - 4)
        else:
            Xt = _anneal(Xt, n_iter=2600, T0=0.022, T1=0.0010, lr0=6.0, lr1=0.12, t_end=t_end - 4)
            Xt = _contact_polish(Xt, steps=900, thr0=6e-3, thr1=3e-4, lr0=0.8, lr1=0.06, t_end=t_end - 4)
        v = _maxcos(Xt)
        if v < bv - 1e-14:
            # extra fine polish on improvement
            if time.time() < t_end - 6:
                Xt = _anneal(Xt, n_iter=1200, T0=0.0010, T1=0.0002, lr0=0.6, lr1=0.04, t_end=t_end - 3)
                v2 = _maxcos(Xt)
                if v2 < v:
                    v = v2
            if v < bv:
                bv = v
                Xbest = Xt.copy()
        trial += 1
    Xbest = _norm(Xbest)
    return Xbest

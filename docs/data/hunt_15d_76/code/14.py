import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(X, steps, T0, T1, lr0, lr1, mu, t0, t_lim):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for s in range(steps):
        if (s & 127) == 0 and time.time() - t0 > t_lim:
            break
        f = s / max(1, steps - 1)
        T = np.exp(lT0 + (lT1 - lT0) * f)
        lr = np.exp(llr0 + (llr1 - llr0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # soft-max weights
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        se = E.sum()
        if se <= 0 or not np.isfinite(se):
            continue
        P = E / se
        Gr = P @ X
        Gr *= 2.0
        # tangential projection
        nd = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - nd * X
        rms = float(np.sqrt((Gt * Gt).sum() / n)) + 1e-12
        Gt = Gt / rms
        V = mu * V + (1.0 - mu) * Gt
        # effective step with momentum correction
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _contact_polish(X, steps, tol0, tol1, lr0, lr1, mu, t0, t_lim):
    n, d = X.shape
    V = np.zeros_like(X)
    ltol0 = np.log(tol0); ltol1 = np.log(tol1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for s in range(steps):
        if (s & 127) == 0 and time.time() - t0 > t_lim:
            break
        f = s / max(1, steps - 1)
        tol = np.exp(ltol0 + (ltol1 - ltol0) * f)
        lr = np.exp(llr0 + (llr1 - llr0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = (G >= m - tol).astype(np.float64)
        # W is symmetric, diag 0 already
        WGsum = ((W * G).sum(axis=1, keepdims=True))
        Gr = W @ X - WGsum * X  # already tangential
        rms = float(np.sqrt((Gr * Gr).sum() / n)) + 1e-12
        if rms < 1e-14:
            continue
        Gt = Gr / rms
        V = mu * V + (1.0 - mu) * Gt
        Y = X - lr * V
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # only accept if not catastrophically worse (allow small uphill)
        Gy = Y @ Y.T
        np.fill_diagonal(Gy, -2.0)
        my = float(Gy.max())
        # accept if improves or within tolerance that shrinks
        if my < m + tol * 0.5:
            X = Y
    return X

def _largest_hole(X, rng, ncand=12000):
    n, d = X.shape
    C = rng.standard_normal(size=(ncand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ X.T  # ncand x n
    m = D.max(axis=1)
    return C[np.argmin(m)]

def solve():
    t0 = time.time()
    T_LIM = 150.0  # leave margin for 180s limit
    rng = np.random.default_rng(12345)
    # load starts
    try:
        Xb = np.load("best.npy").astype(np.float64)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(np.float64)
    except Exception:
        Xr = None
    cands = []
    if Xb is not None and Xb.shape == (76, 15):
        cands.append(Xb)
    if Xr is not None and Xr.shape == (76, 15):
        cands.append(Xr)
    if len(cands) == 0:
        X0 = rng.standard_normal(size=(76, 15))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    for c in cands:
        c /= np.linalg.norm(c, axis=1, keepdims=True)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best = float(vals[bi])

    # quick squeeze of starting point with pure contact descent
    Xtmp = _contact_polish(Xbest.copy(), 800, 0.02, 0.001, 0.004, 0.0006, 0.9, t0, T_LIM)
    v = _max_cos(Xtmp)
    if v < best:
        best = v
        Xbest = Xtmp

    sigmas = [0.015, 0.035, 0.06, 0.10, 0.025, 0.05, 0.08, 0.12, 0.02, 0.045, 0.07, 0.11, 0.03, 0.09, 0.15]
    trial = 0
    # hole-reinsertion counts to try
    for sigma in sigmas:
        if time.time() - t0 > T_LIM:
            break
        trial += 1
        X = Xbest + sigma * rng.standard_normal(size=Xbest.shape)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # every 3rd trial teleport 1-2 worst points into largest holes
        if trial % 3 == 0:
            try:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                m0 = G.max()
                # contacts per point with tight tol
                cnt = (G >= m0 - 1e-3).sum(axis=1)
                # pick points with largest row-max (most offending) to move
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                nmove = 1 + (trial % 2)
                for k in range(nmove):
                    if time.time() - t0 > T_LIM:
                        break
                    idx = int(order[k])
                    # hole w.r.t. all other points
                    Xothers = np.delete(X, idx, axis=0)
                    h = _largest_hole(Xothers, rng, ncand=12000)
                    X[idx] = h
            except Exception:
                pass

        # annealed soft-max repulsion
        X = _anneal(X, 1200, 0.02, 0.0025, 0.02, 0.002, 0.9, t0, T_LIM)
        if time.time() - t0 > T_LIM:
            v = _max_cos(X)
            if v < best:
                best = v
                Xbest = X.copy()
            break
        # greedy contact polish
        X = _contact_polish(X, 900, 0.015, 0.0008, 0.005, 0.0005, 0.9, t0, T_LIM)
        v = _max_cos(X)
        if v < best:
            best = v
            Xbest = X.copy()

    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    # final safety: remove any NaN
    Xbest = np.nan_to_num(Xbest)
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _hole(Xo, rng, n_cand=2000, refine=80):
    n, d = Xo.shape
    C = rng.normal(size=(n_cand, d))
    C = _norm(C)
    S = C @ Xo.T
    smax = S.max(axis=1)
    ib = int(np.argmin(smax))
    h = C[ib].copy()
    # refine: soft-max descent on max dot
    beta = 120.0
    for it in range(refine):
        g = Xo @ h
        gm = float(g.max())
        e = np.exp(beta * (g - gm))
        se = float(e.sum())
        if se == 0 or not np.isfinite(se):
            break
        w = e / se
        gr = Xo.T @ w
        grt = gr - float(gr @ h) * h
        lr = 0.08 * (0.15 ** (it / max(1, refine - 1)))
        h = h - lr * grt
        nh = float(np.linalg.norm(h))
        if nh == 0 or not np.isfinite(nh):
            h = C[ib].copy()
            break
        h /= nh
        if it == refine // 2:
            beta = 300.0
    return h

def _whiten(X, alpha=0.30):
    n, d = X.shape
    try:
        C = (X.T @ X) / n
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-6)
        # M = (d*C)^(-1/2)
        s = 1.0 / np.sqrt(d * w)
        M = (V * s) @ V.T
        Ma = (1.0 - alpha) * np.eye(d) + alpha * M
        # rescale to keep avg norm ~1: trace preserving approx
        Y = X @ Ma
        return _norm(Y)
    except Exception:
        return X

def _anneal(X, steps, T0, T1, lr0, lr1, lam0, lam1):
    n, d = X.shape
    eye = np.eye(d)
    tgt = eye / d
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        lam = lam0 + (lam1 - lam0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        Gr = (E @ X) / S
        if lam != 0.0:
            C = (X.T @ X) / n
            D = C - tgt
            Gi = (4.0 / n) * (X @ D)
            Gr = Gr + lam * Gi
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # clip
        gn = np.linalg.norm(Gt)
        if not np.isfinite(gn) or gn == 0:
            break
        # simple tangent SGD
        X = X - lr * Gt
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (76, 15):
        cands.append(_norm(xb))
    if xr is not None and xr.shape == (76, 15):
        cands.append(_norm(xr))
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        return _norm(rng0.normal(size=(76, 15)))
    best = cands[0]
    bv = _max_cos(best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bv:
            best, bv = c.copy(), v
    best = best.copy()
    n, d = 76, 15
    sigmas = [0.01, 0.03, 0.07, 0.15, 0.28, 0.05, 0.12]
    trial = 0
    # hopping budget: leave 12s for final polish
    while time.time() < deadline - 12.0 and trial < 300:
        tl = deadline - 12.0 - time.time()
        if tl <= 0:
            break
        rng = np.random.default_rng(1000 + trial * 7919)
        sigma = float(sigmas[trial % len(sigmas)])
        Y = best + sigma * rng.normal(size=(n, d))
        Y = _norm(Y)
        # occasional whiten jump before teleport
        if trial % 3 == 0:
            Y = _whiten(Y, alpha=0.25 + 0.05 * ((trial // 3) % 3))
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        per_max = G.max(axis=1)
        cur_max = float(per_max.max())
        # contact counts near max
        thr = 1e-4
        cnt = (G > cur_max - thr).sum(axis=1)
        # choose k to teleport
        if sigma > 0.2:
            k = 4
        elif sigma > 0.1:
            k = 3
        else:
            k = 2
        if trial % 4 == 3:
            k = 4
        # half overloaded, half rattlers/random
        order_hi = np.argsort(-per_max)
        order_lo = np.argsort(per_max)
        idxs = []
        nh = min(k // 2 + (k % 2), k)
        for j in range(nh):
            idxs.append(int(order_hi[(trial + j) % n]))
        for j in range(k - nh):
            # prefer rattlers: lowest per_max, cycle
            idxs.append(int(order_lo[(trial * 2 + j) % n]))
        # dedup
        idxs = list(dict.fromkeys(idxs))
        for kk in idxs:
            if time.time() > deadline - 12.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[int(kk)] = False
            Xo = Y[mask]
            h = _hole(Xo, rng, n_cand=1800, refine=60)
            Y[int(kk)] = h
        Y = _norm(Y)
        # mild whiten after teleport for balance
        if trial % 2 == 0:
            Y = _whiten(Y, alpha=0.18)
        if tl < 25:
            steps = 900
        elif tl < 60:
            steps = 1400
        else:
            steps = 1800
        # anneal params by sigma
        if sigma < 0.04:
            T0, T1, lr0, lr1 = 0.010, 0.0009, 3.0, 0.12
        elif sigma < 0.10:
            T0, T1, lr0, lr1 = 0.014, 0.0008, 4.0, 0.10
        else:
            T0, T1, lr0, lr1 = 0.020, 0.0012, 5.0, 0.12
        lam0, lam1 = 0.02, 0.0
        if trial % 3 == 1:
            lam0 = 0.05
        Y = _anneal(Y, steps, T0, T1, lr0, lr1, lam0, lam1)
        v = _max_cos(Y)
        if v < bv and np.all(np.isfinite(Y)):
            best, bv = Y, v
        trial += 1
    # final low-temp polish on best (no LP to avoid timeout)
    try:
        remain = deadline - time.time()
        if remain > 4:
            # two-stage polish
            s1 = 1500 if remain > 10 else 700
            best = _anneal(best, s1, 0.0012, 0.00025, 0.6, 0.04, 0.0, 0.0)
            bv = _max_cos(best)
            remain2 = deadline - time.time()
            if remain2 > 4:
                s2 = 1200 if remain2 > 8 else 500
                best = _anneal(best, s2, 0.0006, 0.00012, 0.3, 0.02, 0.0, 0.0)
    except Exception:
        pass
    best = _norm(best)
    if not np.all(np.isfinite(best)):
        best = _norm(cands[0])
    return best

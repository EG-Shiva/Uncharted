import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos_batch(X):
    # X: (B,n,d)
    G = np.matmul(X, np.transpose(X, (0, 2, 1)))
    B, n, _ = G.shape
    idx = np.arange(n)
    G[:, idx, idx] = -2.0
    m = G.max(axis=(1, 2))
    return m, G

def _maxcos_single(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_single(x, seconds, global_deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    try:
        best = _maxcos_single(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    t_end = min(time.time() + seconds, global_deadline)
    while time.time() < t_end and radius > 1e-14:
        if time.time() > t_end - 0.6:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 1200:
            # keep tightest 1200
            idx = np.argpartition(-c, 1200)[:1200]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.5, t_end - time.time() - 0.1)})
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos_single(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 160.0
    rng = np.random.default_rng(0)
    n, d = 76, 15
    B = 24

    # load fallback record
    def _load(p):
        try:
            a = np.load(p).astype(float)
            if a.shape == (76, 15):
                a = _norm(a)
                return a
        except Exception:
            pass
        return None
    xb = _load("best.npy")
    xr = _load("record.npy")
    if xb is None and xr is None:
        X0 = rng.normal(size=(76, 15))
        return _norm(X0)
    if xb is None:
        xb = xr
    if xr is None:
        xr = xb
    # fallback = better of the two
    vb = _maxcos_single(xb)
    vr = _maxcos_single(xr)
    if vr < vb:
        fallback = xr.copy()
        fval = vr
    else:
        fallback = xb.copy()
        fval = vb

    # ---- build batch (B,76,15) ----
    X = np.empty((B, n, d), dtype=np.float64)
    X[0] = fallback
    X[1] = xr if xr.shape == (76, 15) else fallback
    sigmas = [0.05, 0.12, 0.25, 0.40]
    for i in range(4):
        b = 2 + i
        if b >= B:
            break
        Y = fallback + sigmas[i] * rng.normal(size=(n, d))
        X[b] = _norm(Y)
    # 15-point resampled records
    for k in range(4):
        b = 6 + k
        if b >= B:
            break
        Y = fallback.copy()
        idx = rng.choice(n, size=15, replace=False)
        R = rng.normal(size=(15, d))
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        Y[idx] = R
        # small shake on rest
        Y = Y + 0.02 * rng.normal(size=(n, d))
        X[b] = _norm(Y)
    # remaining: random + Thomson-like (random is Riesz init)
    for b in range(10, B):
        R = rng.normal(size=(n, d))
        X[b] = _norm(R)

    # temperature factors: 8 cold, 8 medium, 8 hot
    factors = np.ones(B, dtype=np.float64)
    factors[8:16] = 0.4
    factors[16:24] = 0.15
    cold_idx = np.where(factors == 1.0)[0]
    hot_idx = np.where(factors < 1.0)[0]

    N = 1100
    beta0, beta1 = 30.0, 500.0
    lr0, lr1 = 0.30, 0.02
    # precompute schedules
    frac = np.linspace(0, 1, N)
    base_beta = beta0 * (beta1 / beta0) ** frac
    base_lr = lr0 * (lr1 / lr0) ** frac
    idxn = np.arange(n)

    for k in range(N):
        if time.time() > HARD - 55.0:
            break
        bb = base_beta[k] * factors  # (B,)
        lr = base_lr[k] / np.sqrt(factors)  # hotter larger step
        # batched Gram
        G = np.matmul(X, np.transpose(X, (0, 2, 1)))
        G[:, idxn, idxn] = -2.0
        m = G.max(axis=(1, 2))  # (B,)
        # log-sum-exp weights
        # E = exp(beta*(G-m))
        Dm = G - m[:, None, None]
        # clip very negative to avoid underflow cost (optional)
        E = np.exp(bb[:, None, None] * Dm)
        E[:, idxn, idxn] = 0.0
        S = E.sum(axis=(1, 2))
        S[~np.isfinite(S) | (S <= 0)] = 1.0
        Gr = np.matmul(E, X) / S[:, None, None]
        dot = (Gr * X).sum(axis=2, keepdims=True)
        Gt = Gr - dot * X
        # optional Langevin noise for hot replicas, decaying
        X = X - lr[:, None, None] * Gt
        # small noise to change topology for non-cold
        if k < N * 0.7:
            nstd = 0.004 * (1.0 - k / N)
            if nstd > 1e-12:
                noise = rng.normal(size=(B, n, d)) * nstd
                # only hot/medium
                noise[cold_idx] = 0.0
                X = X + noise
        X = _norm(X)
        # replica swap every 100 steps
        if (k + 1) % 100 == 0:
            # evaluate true max (m already)
            i_worst_cold = cold_idx[int(np.argmax(m[cold_idx]))]
            i_best_hot = hot_idx[int(np.argmin(m[hot_idx]))]
            if float(m[i_best_hot]) < float(m[i_worst_cold]) - 1e-12:
                tmp = X[i_worst_cold].copy()
                X[i_worst_cold] = X[i_best_hot]
                X[i_best_hot] = tmp

    # evaluate batch
    mfinal, Gfinal = _maxcos_batch(X)
    order = np.argsort(mfinal)
    # distinct contact graphs within 0.005
    survivors = []
    seen_masks = []
    for bi in order:
        if time.time() > HARD - 50.0:
            break
        mb = float(mfinal[bi])
        Gb = X[bi] @ X[bi].T
        mask = Gb > mb - 0.005
        # compare to seen
        dup = False
        for sm in seen_masks:
            # symmetric difference count (upper triangle)
            diff = np.logical_xor(mask, sm).sum()
            # each pair counted twice; require >20 entries diff to be distinct
            if diff < 20:
                dup = True
                break
        if not dup or len(survivors) == 0:
            survivors.append(int(bi))
            seen_masks.append(mask)
        if len(survivors) >= 3:
            break
    if len(survivors) == 0:
        survivors = [int(order[0])]
    # LP polish survivors, at most 15s each
    bestX = fallback.copy()
    bestV = fval
    # also consider unpolished survivors
    for bi in survivors:
        v = float(mfinal[int(bi)])
        if v < bestV:
            bestV = v
            bestX = X[int(bi)].copy()
    for si, bi in enumerate(survivors):
        remain = HARD - time.time()
        if remain < 2.0:
            break
        secs = min(15.0, remain - 1.0)
        if secs <= 0.5:
            break
        Y = _lp_polish_single(X[int(bi)], secs, HARD)
        v = _maxcos_single(Y)
        if v < bestV:
            bestV = v
            bestX = Y.copy()
    # fallback safety
    if bestX.shape != (76, 15) or not np.all(np.isfinite(bestX)):
        return fallback
    bestX = _norm(bestX)
    return bestX

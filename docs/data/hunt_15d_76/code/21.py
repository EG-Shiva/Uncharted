import time
import numpy as np

def _norm1(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _normB(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos1(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_single(x, budget, t_dead):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm1(x.astype(float).copy())
    n, d = x.shape
    try:
        best = _maxcos1(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    t0 = time.time()
    while time.time() < t_dead and (time.time() - t0) < budget and radius > 1e-14:
        if t_dead - time.time() < 0.6:
            break
        G = x @ x.T
        c = G[iu0]
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
        I, J = iu0[0][sel], iu0[1][sel]
        csel = c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(1.0, min(5.0, t_dead - time.time() - 0.2))})
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm1(y)
        try:
            v = _maxcos1(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_HARD = t0 + 160.0
    rng = np.random.default_rng(0)
    n, d, B = 76, 15, 24

    # fallback record
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = None
    if Xb is not None and Xb.shape == (76, 15):
        Xb = _norm1(Xb)
    else:
        Xb = None
    if Xr is not None and Xr.shape == (76, 15):
        Xr = _norm1(Xr)
    else:
        Xr = None
    if Xb is None and Xr is None:
        X = rng.normal(size=(76, 15)); X = _norm1(X)
        return X
    if Xb is None:
        Xb = Xr.copy()
    if Xr is None:
        Xr = Xb.copy()
    fallback = Xb.copy()
    fb_val = _maxcos1(fallback)
    v_r = _maxcos1(Xr)
    if v_r < fb_val:
        fallback = Xr.copy(); fb_val = v_r

    # ---- build batch (B,76,15) ----
    X = np.empty((B, n, d), dtype=np.float64)
    X[0] = Xb
    X[1] = Xr
    sigmas = [0.02, 0.08, 0.15, 0.30]
    for k, s in enumerate(sigmas):
        Y = Xb + s * rng.normal(size=(n, d))
        X[2 + k] = _norm1(Y)
    X[6] = _norm1(Xr + 0.08 * rng.normal(size=(n, d)))
    X[7] = _norm1(Xr + 0.20 * rng.normal(size=(n, d)))
    # random / Thomson-like
    for b in range(8, 11):
        X[b] = _norm1(rng.normal(size=(n, d)))
    # Riesz-ish: random then tiny Coulomb smoothing (few steps inline)
    for b in range(11, 14):
        Y = _norm1(rng.normal(size=(n, d)))
        # 60 steps of simple power repulsion to diversify
        for _ in range(60):
            G = Y @ Y.T
            np.fill_diagonal(G, 0.0)
            # inverse power weight
            with np.errstate(divide='ignore'):
                W = 1.0 / (np.clip(2.0 - 2.0 * G, 1e-6, 4.0) ** 1.5 + 1e-9)
            np.fill_diagonal(W, 0.0)
            F = W @ Y
            dot = np.sum(F * Y, axis=1, keepdims=True)
            Ft = F - dot * Y
            # normalize step
            Y = _norm1(Y - 0.02 * Ft)
        X[b] = Y
    # 15-point resampled from best
    for k, b in enumerate(range(14, 18)):
        Y = Xb.copy()
        idx = rng.choice(n, size=15, replace=False)
        Y[idx] = _norm1(rng.normal(size=(15, d)))
        X[b] = Y
    for k, b in enumerate(range(18, 22)):
        Y = Xr.copy()
        idx = rng.choice(n, size=15, replace=False)
        Y[idx] = _norm1(rng.normal(size=(15, d)))
        X[b] = Y
    X[22] = _norm1(Xb + 0.5 * rng.normal(size=(n, d)))
    X[23] = _norm1(rng.normal(size=(n, d)))
    X = _normB(X)

    # temperature levels for replica exchange
    levels = np.array([1.0]*8 + [0.6]*8 + [0.35]*8, dtype=np.float64)

    total = 2500
    beta0, beta1 = 30.0, 500.0
    lr0, lr1 = 2.5, 0.12
    mu = 0.85
    V = np.zeros_like(X)
    best_ever = fallback.copy()
    best_val = fb_val

    # quick eval helper batched max
    def batch_max(Xb_):
        G = np.matmul(Xb_, np.transpose(Xb_, (0, 2, 1)))
        for b in range(G.shape[0]):
            np.fill_diagonal(G[b], -2.0)
        return G.max(axis=(1, 2)), G

    cur_max, _ = batch_max(X)
    for b in range(B):
        if cur_max[b] < best_val:
            best_val = float(cur_max[b]); best_ever = X[b].copy()

    for k in range(total):
        if time.time() > T_HARD - 45.0:
            break
        f = k / max(1, total - 1)
        beta_base = beta0 * (beta1 / beta0) ** f
        beta_eff = beta_base * levels
        lr = lr0 * (lr1 / lr0) ** f
        noise_sig = 0.004 * (1.0 - f)
        # forward
        G = np.matmul(X, np.transpose(X, (0, 2, 1)))
        # per-replica max for stability
        m = G.max(axis=(1, 2))
        # exp
        # broadcast beta
        be = beta_eff[:, None, None]
        mm = m[:, None, None]
        E = np.exp(be * (G - mm))
        for b in range(B):
            np.fill_diagonal(E[b], 0.0)
        S = E.sum(axis=(1, 2))
        S[~np.isfinite(S)] = 1.0
        S[S <= 0] = 1.0
        W = E / S[:, None, None]
        Gr = 2.0 * np.matmul(W, X)
        dot = np.sum(Gr * X, axis=2, keepdims=True)
        Gt = Gr - dot * X
        # clip
        gn = np.linalg.norm(Gt, axis=2, keepdims=True) + 1e-12
        # clip huge per-point grads to 2.0
        scale = np.minimum(1.0, 2.0 / gn)
        Gt = Gt * scale
        V = mu * V + Gt
        X = X - lr * V
        if noise_sig > 0:
            Nz = rng.normal(size=X.shape) * noise_sig
            proj = np.sum(Nz * X, axis=2, keepdims=True)
            Nz = Nz - proj * X
            X = X + Nz
        X = _normB(X)
        # replica swap every 200 iters
        if (k + 1) % 200 == 0:
            cmax, _ = batch_max(X)
            # update global best
            for b in range(B):
                if cmax[b] < best_val - 1e-12:
                    best_val = float(cmax[b]); best_ever = X[b].copy()
            # swap cold (0:8) with hot (16:24)
            for c in range(8):
                h = int(rng.integers(16, 24))
                if cmax[h] < cmax[c] - 1e-9:
                    tmp = X[c].copy(); X[c] = X[h]; X[h] = tmp
                    tmpv = V[c].copy(); V[c] = V[h]; V[h] = tmpv
                    # also swap recorded max to avoid re-swap confusion (not needed)
            # re-center velocity a bit after swap phase
            V *= 0.9

    # final eval
    fin_max, finG = batch_max(X)
    order = np.argsort(fin_max)
    # build contact sets (pairs within 0.005 of max)
    iu = np.triu_indices(n, 1)
    sets = []
    for b in range(B):
        Gb = finG[b][iu]
        mb = float(fin_max[b])
        sets.append(Gb > mb - 0.005)
    sets = np.array(sets)  # (B,P)
    survivors = []
    for b in order:
        if time.time() > T_HARD - 40:
            break
        keep = True
        for s in survivors:
            # distinct if hamming large
            inter = float(np.logical_and(sets[b], sets[s]).sum())
            union = float(np.logical_or(sets[b], sets[s]).sum()) + 1e-9
            jacc = inter / union
            if jacc > 0.97:
                keep = False
                break
        if keep:
            survivors.append(int(b))
        if len(survivors) >= 3:
            break
    if len(survivors) == 0:
        survivors = [int(order[0])]
    # LP polish survivors, at most ~12s each
    polished = []
    for s in survivors:
        if time.time() > T_HARD - 5:
            break
        cand = X[s].copy()
        v0 = float(fin_max[s])
        # only polish if promising (within 0.01 of best)
        if v0 > best_val + 0.01 and len(polished) > 0:
            continue
        xp = _lp_polish_single(cand, budget=12.0, t_dead=T_HARD)
        vp = _maxcos1(xp)
        polished.append((vp, xp))
        if vp < best_val:
            best_val = vp; best_ever = xp.copy()
    # also consider unpolished best
    for s in survivors:
        v = float(fin_max[s])
        if v < best_val:
            best_val = v; best_ever = X[s].copy()
    if fb_val < best_val:
        best_ever = fallback.copy(); best_val = fb_val
    best_ever = _norm1(best_ever)
    return best_ever

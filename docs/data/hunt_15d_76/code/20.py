import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = X.astype(float, copy=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    X /= n
    return X

def _best_hole(Xo, rng, n_cand=3000, refine=40):
    d = Xo.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xo.T
    smax = S.max(axis=1)
    ib = int(np.argmin(smax))
    h = C[ib].copy()
    # quick tangential descent on max dot (soft-max)
    beta = 150.0
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
        lr = 0.05 * (0.1 ** (it / max(1, refine - 1)))
        h = h - lr * grt
        nh = float(np.linalg.norm(h))
        if nh == 0 or not np.isfinite(nh):
            h = C[ib]
            break
        h /= nh
    return h

def _adam_smooth(Y, steps, beta0, beta1, lr0, lr1):
    n, d = Y.shape
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    lb = np.log(beta1 / beta0)
    ll = np.log(lr1 / lr0)
    for s_idx in range(steps):
        frac = s_idx / (steps - 1) if steps > 1 else 0.0
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            break
        E /= S
        Gr = 2.0 * (E @ Y)
        Gt = Gr - (np.sum(Gr * Y, axis=1, keepdims=True)) * Y
        b1t *= b1
        b2t *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1t)
        vhat = vv / (1.0 - b2t)
        Y -= lr * mhat / (np.sqrt(vhat) + eps)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(x, t_end):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end - 0.4 and radius > 1e-14 and it < 40:
        it += 1
        G = x @ x.T
        c = G[iu]
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 900:
            kth = np.argpartition(-c, 900)[:900]
            mask = np.zeros_like(sel)
            mask[kth] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    import time
    import numpy as np
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
        X = rng0.normal(size=(76, 15))
        return _norm(X)
    best = cands[0]
    bv = _max_cos(best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bv:
            best, bv = c.copy(), v
    best = best.copy()
    n, d = 76, 15
    sigmas = [0.02, 0.05, 0.10, 0.20, 0.35, 0.08, 0.15]
    trial = 0
    # reserve ~22s for final LP
    while time.time() < deadline - 30 and trial < 200:
        time_left = deadline - 30 - time.time()
        if time_left <= 0:
            break
        rng = np.random.default_rng(7777 + trial * 131)
        sigma = float(sigmas[trial % len(sigmas)])
        mode = trial % 4
        Y = best + sigma * rng.normal(size=(n, d))
        Y = _norm(Y)
        # choose points to teleport to holes
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        per_max = G.max(axis=1)
        # rattlers with loose threshold
        cur_max = float(per_max.max())
        # near-contact counts for diagnosis
        if mode == 0:
            # teleport worst k
            k = 2 + (trial % 4)  # 2..5
            idxs = np.argpartition(-per_max, k)[:k]
        elif mode == 1:
            # teleport rattlers (lowest per_max)
            k = 3 + (trial % 3)
            idxs = np.argpartition(per_max, k)[:k]
        elif mode == 2:
            # cluster shake + teleport worst
            u = rng.normal(size=(d,))
            u /= np.linalg.norm(u)
            proj = Y @ u
            # shake one side
            side = proj > np.median(proj) if (trial // 4) % 2 == 0 else proj <= np.median(proj)
            Y[side] += 0.12 * rng.normal(size=(int(side.sum()), d))
            Y = _norm(Y)
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            k = 3
            idxs = np.argpartition(-per_max, k)[:k]
        else:
            # random subset teleport
            k = 4
            idxs = rng.choice(n, size=k, replace=False)
        # teleport selected into holes sequentially
        for kk in idxs:
            if time.time() > deadline - 30:
                break
            mask = np.ones(n, dtype=bool)
            mask[int(kk)] = False
            Xo = Y[mask]
            h = _best_hole(Xo, rng, n_cand=2500, refine=30)
            Y[int(kk)] = h
        Y = _norm(Y)
        # anneal steps adapt to time
        if time_left < 20:
            steps = 250
        elif time_left < 50:
            steps = 400
        else:
            steps = 550
        # vary beta/lr slightly per trial
        beta0, beta1 = 30.0, 400.0
        lr0, lr1 = 0.010, 0.0008
        if mode == 2:
            lr0 = 0.006
        Y = _adam_smooth(Y, steps, beta0, beta1, lr0, lr1)
        val = _max_cos(Y)
        if val < bv:
            best, bv = Y, val
        trial += 1
    # final polish
    remain = deadline - time.time()
    if remain > 5:
        t_end = time.time() + min(remain - 1.0, 22.0)
        try:
            polished = _lp_polish(best, t_end)
            pv = _max_cos(polished)
            if pv < bv:
                best = polished
        except Exception:
            pass
    best = _norm(best)
    if not np.all(np.isfinite(best)):
        best = _norm(cands[0])
    return best

import time
import numpy as np

def _norm(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, slack=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - slack
    deg = mask.sum(axis=1)
    # sum of near-max cosines for ranking
    Gs = np.where(mask, G, 0.0).sum(axis=1)
    permax = G.max(axis=1)
    return deg, Gs, permax, float(m), G

def _adam_hinge(Y, steps, beta0=30.0, beta1=400.0, lr0=0.01, lr1=0.0008,
                hinge_thr=0.164, w_hinge=8.0, w_soft=1.0, fix_mask=None):
    n, d = Y.shape
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    lb = np.log(beta1 / beta0) if beta0 > 0 and beta1 > 0 else 0.0
    ll = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 else 0.0
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
        Gr_soft = 2.0 * (E @ Y)
        # hinge part
        Hp = G - hinge_thr
        np.maximum(Hp, 0.0, out=Hp)
        np.fill_diagonal(Hp, 0.0)
        # W = 2*Hp
        Hp *= 2.0
        Gr_hinge = Hp @ Y
        Gr = w_soft * Gr_soft + w_hinge * Gr_hinge
        Gt = Gr - (np.sum(Gr * Y, axis=1, keepdims=True)) * Y
        if fix_mask is not None:
            Gt[fix_mask] = 0.0
        b1t *= b1
        b2t *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1t)
        vhat = vv / (1.0 - b2t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        if fix_mask is not None:
            step[fix_mask] = 0.0
        Y = Y - step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _refine_holes_batch(V, Y, iters=180, T0=0.008, lr0=0.08):
    # V: (k,d), Y: (m,d)
    V = _norm(V)
    k = V.shape[0]
    for it in range(iters):
        f = it / max(1, iters - 1)
        Tk = T0 * (0.15 ** f)
        lrk = lr0 * (0.15 ** f)
        S = V @ Y.T  # k x m
        smax = S.max(axis=1, keepdims=True)
        W = np.exp((S - smax) / Tk)
        Ws = W.sum(axis=1, keepdims=True)
        Ws[Ws == 0] = 1.0
        W /= Ws
        Gc = W @ Y  # k x d
        dot = np.sum(Gc * V, axis=1, keepdims=True)
        Gt = Gc - dot * V
        # scale guard
        ng = np.linalg.norm(Gt, axis=1, keepdims=True)
        # avoid huge steps
        Gt = Gt / np.maximum(1.0, ng * 5.0)
        V = V - lrk * Gt
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

def _pick_hole(Y, rng, n_cand=50000, n_best=20, refine_iters=180):
    n, d = Y.shape
    m = Y.shape[0]
    best_scores = None
    best_vecs = None
    # chunked sampling to bound memory
    chunk = 10000
    n_chunks = (n_cand + chunk - 1) // chunk
    cand_pool = []
    score_pool = []
    for _ in range(n_chunks):
        cur = chunk if (len(cand_pool) * chunk + cur if False else True) else chunk
        # compute remaining
        done = sum(len(c) if hasattr(c, '__len__') else 0 for c in [])  # placeholder
        # simpler: just generate chunk, last chunk may overshoot slightly (fine)
        C = rng.normal(size=(chunk, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Y.T
        sm = S.max(axis=1)
        cand_pool.append(C)
        score_pool.append(sm)
    C = np.concatenate(cand_pool, axis=0)[:n_cand]
    scores = np.concatenate(score_pool, axis=0)[:n_cand]
    idx = np.argpartition(scores, n_best)[:n_best]
    V0 = C[idx]
    Vr = _refine_holes_batch(V0.copy(), Y, iters=refine_iters)
    # score refined
    Sr = Vr @ Y.T
    smr = Sr.max(axis=1)
    bi = int(np.argmin(smr))
    return Vr[bi]

def _local_polish(Y, idx_new, steps=350):
    n, d = Y.shape
    # 20 nearest neighbours (largest dots to new point)
    g = Y @ Y[idx_new]
    g[idx_new] = -2.0
    nn = np.argpartition(-g, min(20, n - 1))[:min(20, n - 1)]
    subset = np.zeros(n, dtype=bool)
    subset[idx_new] = True
    subset[nn] = True
    fix = ~subset
    # short hinge+soft polish restricted to subset
    Y2 = _adam_hinge(Y.copy(), steps, beta0=80.0, beta1=500.0,
                     lr0=0.005, lr1=0.0005, hinge_thr=0.164,
                     w_hinge=6.0, w_soft=1.0, fix_mask=fix)
    return Y2

def _lp_polish_global(x, t_end):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(x)
    n, d = x.shape
    try:
        best = _maxcos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end - 0.6 and radius > 1e-14 and it < 30:
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
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    base = None
    bv = 1e9
    for arr in (xb, xr):
        if arr is not None and arr.shape == (76, 15):
            a = _norm(arr)
            v = _maxcos(a)
            if v < bv:
                bv = v
                base = a.copy()
    if base is None:
        rng0 = np.random.default_rng(0)
        return _norm(rng0.normal(size=(76, 15)))
    record_thr = 0.168263492524
    best = base.copy()
    bestv = float(bv)
    # rank worst core points at slack 1e-9
    deg, gsum, permax, mcur, Gfull = _contacts(best, slack=1e-9)
    # worst: high degree, then high gsum, then high permax
    order = np.lexsort((-permax, -gsum, -deg))
    # order[0] is worst; lexsort sorts ascending so negate -> check:
    # np.lexsort keys last is primary; we want primary -deg? Actually use structured sort:
    idx_all = np.arange(76)
    # simpler: sort by (deg desc, gsum desc)
    idx_sorted = sorted(idx_all, key=lambda i: (int(deg[i]), float(gsum[i]), float(permax[i])), reverse=True)
    idx_sorted = np.array(idx_sorted, dtype=int)
    del_sizes = [8, 6, 10]
    for di, ds in enumerate(del_sizes):
        if time.time() > deadline - 30:
            break
        del_idx = idx_sorted[:ds].copy()
        # keep mask
        keep = np.ones(76, dtype=bool)
        keep[del_idx] = False
        Y = best[keep].copy()  # (76-ds, 15)
        # re-optimize remaining core with hinge + softmax until below record
        # two-stage anneal
        remain = deadline - time.time()
        if remain < 25:
            break
        # stage 1: coarse
        Y = _adam_hinge(Y, steps=1200, beta0=30.0, beta1=400.0,
                        lr0=0.012, lr1=0.001, hinge_thr=0.164,
                        w_hinge=8.0, w_soft=1.0)
        if time.time() > deadline - 25:
            # evaluate partial? skip insertions if no time
            pass
        Y = _adam_hinge(Y, steps=800, beta0=200.0, beta1=800.0,
                        lr0=0.004, lr1=0.0004, hinge_thr=0.164,
                        w_hinge=10.0, w_soft=1.0)
        # check core max (internal) - should be well below record
        # reinsert one at a time
        rng = np.random.default_rng(1000 + ds * 77 + di * 13)
        # insertion order: worst first (as deleted order)
        for k, orig in enumerate(del_idx):
            if time.time() > deadline - 25:
                break
            h = _pick_hole(Y, rng, n_cand=50000, n_best=20, refine_iters=150)
            Y = np.vstack([Y, h.reshape(1, -1)])
            Y = _norm(Y)
            # short restricted polish on inserted point + 20 NN
            Y = _local_polish(Y, idx_new=Y.shape[0] - 1, steps=250)
            # quick global light smoothing to settle (few steps, cheap)
            Y = _adam_hinge(Y, steps=120, beta0=200.0, beta1=500.0,
                            lr0=0.002, lr1=0.0005, hinge_thr=0.164,
                            w_hinge=6.0, w_soft=1.0)
            # intermediate check: only when full size reached
            if Y.shape[0] == 76:
                v = _maxcos(Y)
                if v < bestv:
                    bestv = float(v)
                    best = Y.copy()
        # after all insertions, Y should be 76
        if Y.shape[0] == 76:
            # a bit more global smoothing
            if time.time() < deadline - 22:
                Y = _adam_hinge(Y, steps=500, beta0=100.0, beta1=600.0,
                                lr0=0.003, lr1=0.0004, hinge_thr=0.164,
                                w_hinge=5.0, w_soft=1.0)
                v = _maxcos(Y)
                if v < bestv:
                    bestv = float(v)
                    best = Y.copy()
    # final global LP polish capped at 20s
    if time.time() < deadline - 2:
        t_end = min(deadline, time.time() + 20.0)
        # ensure at least 2s
        if t_end - time.time() > 2.0:
            try:
                Yp = _lp_polish_global(best, t_end)
                vp = _maxcos(Yp)
                if vp < bestv and np.all(np.isfinite(Yp)):
                    best = Yp
                    bestv = float(vp)
            except Exception:
                pass
    best = _norm(best)
    if not np.all(np.isfinite(best)):
        best = _norm(base)
    return best

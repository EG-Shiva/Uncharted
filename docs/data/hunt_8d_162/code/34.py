import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(X, iters, a0, a1, s0, s1):
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = a0 * (a1 / a0) ** frac if iters > 1 else a0
        step = s0 * (s1 / s0) ** frac if iters > 1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(alpha * (G - gmax))
        W[W < 1e-12] = 0.0
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        nrm = float(np.linalg.norm(Ft, axis=1).mean()) + 1e-12
        X = X - (step / nrm) * Ft
        nr = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
        X = X / nr
    return X

def _find_hole(X, rng, n_sample=35000):
    n, d = X.shape
    best_m = 1e9
    best_C = None
    # batched sampling to bound memory
    chunk = 10000
    cands = []
    # first pass: collect top candidates per chunk
    top_pool = []
    n_chunks = (n_sample + chunk - 1) // chunk
    for _ in range(n_chunks):
        m = min(chunk, n_sample - len(cands) * 0 if False else chunk)
        C = rng.standard_normal((chunk, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        S = C @ X.T
        sc = S.max(axis=1)
        # keep best 10 of chunk
        idx = np.argpartition(sc, 10)[:10]
        for k in idx:
            top_pool.append((float(sc[k]), C[k].copy()))
    top_pool.sort(key=lambda t: t[0])
    pool = [v for _, v in top_pool[:8]]
    best_v = None
    best_s = 1e9
    for v0 in pool:
        v = v0.copy()
        lr = 0.06
        for it in range(220):
            dts = X @ v
            cur = float(dts.max())
            # softmax weights toward closest
            beta = 60.0 + 140.0 * (it / 220)
            w = np.exp(beta * (dts - cur))
            g = w @ X
            gt = g - float(g @ v) * v
            nn = float(np.linalg.norm(gt)) + 1e-12
            gt /= nn
            st = lr * (0.25 ** (it / 220)) + 0.002
            v = v - st * gt
            v /= np.linalg.norm(v) + 1e-30
        s = float((X @ v).max())
        if s < best_s:
            best_s = s
            best_v = v
    return best_v

def _contact_hist(X, tol=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tol, axis=1)
    return m, cnt

def _lp_polish(X, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1.5:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        if m < best:
            best = m
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 4000:
            thr = np.sort(c[sel])[-4000]
            sel = c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 175.0
    # load incumbent for fallback
    inc = None
    inc_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            A = np.load(f).astype(float)
            A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-30
            if A.shape == (162, 8):
                v = _max_cos(A)
                if v < inc_val:
                    inc_val = v
                    inc = A.copy()
        except Exception:
            pass

    def grow(seed, n_start, n_sample, it_init, it_ins, rng_seed_offset=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_start, 8))
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
        # initial batched softmax descent with increasing sharpness
        X = _anneal(X, it_init, 20.0, 220.0, 0.025, 0.002)
        for ncur in range(n_start, 162):
            if time.time() - t0 > BUDGET - 45.0:
                break
            hv = _find_hole(X, rng, n_sample=n_sample)
            X = np.vstack([X, hv[None, :]])
            # re-anneal all points, sharpness increasing with size
            # later insertions use sharper schedule
            frac = (ncur - n_start + 1) / max(1, 162 - n_start)
            a1 = 250.0 + 200.0 * frac
            X = _anneal(X, it_ins, 25.0, a1, 0.02, 0.0015)
        # ensure exactly 162 (if time cut early, pad by holes without anneal)
        while X.shape[0] < 162:
            rng2 = np.random.default_rng(seed + X.shape[0] * 1009)
            hv = _find_hole(X, rng2, n_sample=20000)
            X = np.vstack([X, hv[None, :]])
            X = _anneal(X, 200, 30.0, 300.0, 0.012, 0.001)
        return X

    configs = [
        (101, 145, 35000, 1100, 380),
        (202, 141, 35000, 1100, 380),
        (303, 149, 30000, 900, 320),
    ]
    results = []
    grow_deadline = t0 + 120.0
    for (seed, ns, nsamp, it0, iti) in configs:
        if time.time() > grow_deadline:
            break
        try:
            X = grow(seed, ns, nsamp, it0, iti)
        except Exception:
            continue
        # final short anneal to settle basin
        try:
            if time.time() < grow_deadline:
                X = _anneal(X, 500, 30.0, 450.0, 0.01, 0.0008)
        except Exception:
            pass
        try:
            v = _max_cos(X)
        except Exception:
            continue
        _, cnt = _contact_hist(X)
        results.append((v, X, cnt, seed))
        if time.time() > grow_deadline:
            break

    # rank by score, keep distinct basins via contact histogram + score gap
    results.sort(key=lambda t: t[0])
    picked = []
    for v, X, cnt, seed in results:
        keep = True
        for _, X2, cnt2, _ in picked:
            # same basin if scores within 2e-4 and contact histograms similar
            # quick proxy: mean contacts close and score close
            if abs(v - _max_cos(X2)) < 2e-4 and abs(float(cnt.mean()) - float(cnt2.mean())) < 1.0:
                # Procrustes rotation check: align via SVD on cross-cov (ignores permutation,
                # so only used as coarse similarity; require also score proximity)
                try:
                    M = X.T @ X2
                    U, _, Vt = np.linalg.svd(M)
                    R = U @ Vt
                    # centered? just rotation error of Gram
                    diff = float(np.linalg.norm(X @ R.T - X2) / 162.0)
                    if diff < 0.15:
                        keep = False
                        break
                except Exception:
                    pass
        if keep:
            picked.append((v, X, cnt, seed))
        if len(picked) >= 2:
            break
    if len(picked) == 0 and len(results) > 0:
        picked = [(results[0][0], results[0][1], results[0][2], results[0][3])]
    # if nothing grew, fall back to incumbent
    if len(picked) == 0:
        if inc is not None:
            return inc / (np.linalg.norm(inc, axis=1, keepdims=True) + 1e-30)
        Xf = np.random.default_rng(0).standard_normal((162, 8))
        return Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-30)

    # LP-polish best one or two distinct results with remaining time
    remain = BUDGET - (time.time() - t0)
    # reserve 3s for return
    lp_budget = max(5.0, remain - 3.0)
    # split: 60% to best, 40% to second if exists
    polished = []
    for i, (v, X, cnt, seed) in enumerate(picked[:2]):
        if lp_budget <= 3:
            polished.append((v, X))
            continue
        if len(picked[:2]) == 1:
            sec = min(30.0, lp_budget)
        else:
            sec = min(30.0, lp_budget * (0.6 if i == 0 else 0.4))
        if sec < 3:
            polished.append((v, X))
            continue
        try:
            Y = _lp_polish(X.copy(), sec)
            vy = _max_cos(Y)
            polished.append((vy, Y))
        except Exception:
            polished.append((v, X))
        lp_budget -= (time.time() - t0) * 0  # time accounted via clock
        remain = BUDGET - (time.time() - t0)
        if remain < 6:
            break
        lp_budget = max(0.0, remain - 3.0)

    polished.sort(key=lambda t: t[0])
    best_v, best_X = polished[0]
    if inc is not None and inc_val <= best_v:
        return inc / (np.linalg.norm(inc, axis=1, keepdims=True) + 1e-30)
    best_X = best_X / (np.linalg.norm(best_X, axis=1, keepdims=True) + 1e-30)
    # final safety: return better vs incumbent
    try:
        bv = _max_cos(best_X)
        if inc is not None and inc_val < bv:
            return inc
    except Exception:
        if inc is not None:
            return inc
    return best_X

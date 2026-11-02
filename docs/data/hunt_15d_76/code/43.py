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

def _anneal_adam(X, steps, beta0, beta1, lr0, lr1, seed, t_dead=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute schedule
    for k in range(steps):
        f = k / max(1, steps - 1)
        beta = beta0 * ((beta1 / beta0) ** f)
        lr = lr0 * ((lr1 / lr0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        # E = exp(beta*(G-gm))
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # Adam
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m1 / bc1
        vh = m2 / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        # clip step norm to avoid explosion
        X = X - step
        X = _norm(X)
        if (k & 31) == 31 and t_dead is not None:
            if time.time() > t_dead:
                break
    return X

def _lp_polish_limit(X, time_limit, t_dead):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _norm(X)
    n, d = X.shape
    best = _maxcos(X)
    radius = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() < t_dead and (time.time() - t0) < time_limit and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 1300:
            # keep tightest 1300
            idx = np.argpartition(-c, 1300)[:1300]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 165.0
    LP_TOTAL = 20.0
    lp_used = 0.0

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
    for a in (Xb, Xr):
        if a is not None and a.shape == (76, 15):
            a = _norm(a.astype(float))
            if np.all(np.isfinite(a)):
                cands.append(a)
    if not cands:
        R = rng0.normal(size=(76, 15))
        return _norm(R)
    bestX = min(cands, key=_maxcos).copy()
    bestV = _maxcos(bestX)
    baseX = bestX.copy()
    n, d = 76, 15
    N = n * d

    # --- active set slack < 1e-7 ---
    G0 = baseX @ baseX.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    iu0 = np.triu_indices(n, 1)
    c0 = (baseX @ baseX.T)[iu0]
    slack = m0 - c0
    order = np.argsort(slack)  # tightest first
    # keep pairs with slack < 1e-7, cap 1500
    tight = order[slack[order] < 1e-7]
    if len(tight) == 0:
        tight = order[:500]
    if len(tight) > 1500:
        tight = tight[:1500]
    Iact = iu0[0][tight]
    Jact = iu0[1][tight]
    p = len(tight)
    # contact counts for rattler detection
    cnt = np.zeros(n, dtype=int)
    # use threshold 1e-7 for contacts
    for ii, jj in zip(Iact, Jact):
        cnt[ii] += 1
        cnt[jj] += 1
    rattlers = np.where(cnt == 0)[0]

    # --- build tangent-space Jacobian: row (i,j) has blocks xj-c*xi at i, xi-c*xj at j ---
    J = np.zeros((p, N), dtype=np.float64)
    Gall = baseX @ baseX.T
    for k in range(p):
        i = int(Iact[k]); j = int(Jact[k])
        c = float(Gall[i, j])
        J[k, i * d:(i + 1) * d] = baseX[j] - c * baseX[i]
        J[k, j * d:(j + 1) * d] = baseX[i] - c * baseX[j]

    # --- smallest singular vectors with svds ---
    modes = None
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds
        Js = csr_matrix(J)
        kk = int(min(15, min(J.shape) - 2))
        if kk >= 3:
            _, _, Vt_s = svds(Js, k=kk, which='SM', tol=1e-4, maxiter=800)
            # Vt_s shape kk x N
            modes = Vt_s.copy()
    except Exception:
        modes = None
    if modes is None:
        # dense fallback: thin SVD, take smallest rows
        try:
            _, Sthin, Vtthin = np.linalg.svd(J, full_matrices=False)
            kk = min(15, Vtthin.shape[0])
            modes = Vtthin[-kk:].copy()
        except Exception:
            modes = rng0.normal(size=(15, N))

    # add random tangent vectors to reach ~20 modes + ensure full rank
    Kneed = 20
    Khave = modes.shape[0]
    if Khave < Kneed:
        extra = []
        rngE = np.random.default_rng(7)
        for _ in range(Kneed - Khave):
            r = rngE.normal(size=(n, d))
            # project out radial
            dot = np.sum(r * baseX, axis=1, keepdims=True)
            r = r - dot * baseX
            extra.append(r.ravel() / (np.linalg.norm(r.ravel()) + 1e-12))
        modes = np.concatenate([modes, np.stack(extra, axis=0)], axis=0)
    # normalize each mode and make tangent (project radial)
    for k in range(modes.shape[0]):
        M = modes[k].reshape(n, d)
        dot = np.sum(M * baseX, axis=1, keepdims=True)
        M = M - dot * baseX
        nn = np.linalg.norm(M)
        if nn > 0 and np.isfinite(nn):
            M /= nn
        modes[k] = M.ravel()
    K = modes.shape[0]

    amps = [0.03, 0.06, 0.12, 0.24]
    # trial loop
    trial = 0
    # total trials bounded by time
    while time.time() < DEAD - 8.0:
        if time.time() > DEAD - 6.0:
            break
        # choose Kuse 5..20 cycling
        Kuse_list = [5, 8, 12, 16, 20]
        Kuse = Kuse_list[trial % len(Kuse_list)]
        amp = amps[(trial // 1) % len(amps)]
        rng = np.random.default_rng(12345 + trial * 977)
        # pick subset of modes (lowest): use first Kuse (svds order arbitrary, use random subset)
        idxm = rng.choice(K, size=min(Kuse, K), replace=False)
        coeff = rng.normal(size=len(idxm))
        nn = np.linalg.norm(coeff)
        if nn == 0:
            trial += 1
            continue
        coeff /= nn
        disp = (coeff @ modes[idxm]).reshape(n, d)  # unit norm total
        Y = baseX + amp * disp
        # larger motion to rattlers
        if len(rattlers) > 0:
            Rr = rng.normal(size=(len(rattlers), d))
            # make tangent-ish
            for ti, ri in enumerate(rattlers):
                # project out radial of base
                v = Rr[ti]
                v = v - np.dot(v, baseX[ri]) * baseX[ri]
                nv = np.linalg.norm(v)
                if nv > 0:
                    v /= nv
                Rr[ti] = v
            boost = amp * 1.5 + 0.05
            Y[rattlers] += boost * Rr
        Y = _norm(Y)
        # fast vectorized log-sum-exp Adam anneal beta 40->400 ~1500 steps
        steps = 1500
        # shorten if low time
        if time.time() > DEAD - 30:
            steps = 800
        Ya = _anneal_adam(Y, steps=steps, beta0=40.0, beta1=400.0,
                          lr0=0.02, lr1=0.003, seed=trial, t_dead=DEAD - 6.0)
        va = _maxcos(Ya)
        # only polish if promising (within 0.004 of best or better) and LP budget left
        did_polish = False
        if va < bestV + 0.004 and (LP_TOTAL - lp_used) > 1.0 and time.time() < DEAD - 8.0:
            allow = min(4.0, LP_TOTAL - lp_used, max(1.0, DEAD - 6.0 - time.time()))
            if allow > 0.5:
                lp_s = time.time()
                Yp = _lp_polish_limit(Ya, time_limit=allow, t_dead=DEAD - 5.0)
                lp_used += time.time() - lp_s
                vp = _maxcos(Yp)
                did_polish = True
                if vp < bestV - 1e-12:
                    bestV = vp
                    bestX = Yp.copy()
                    baseX = Yp.copy()  # move basin (keep modes? recompute counts only)
                    # update rattlers based on new best
                    Gn = bestX @ bestX.T
                    np.fill_diagonal(Gn, -2.0)
                    mn = float(Gn.max())
                    # quick rattler update with 1e-7
                    Gfull = bestX @ bestX.T
                    mask_c = (Gfull > mn - 1e-7)
                    np.fill_diagonal(mask_c, False)
                    cnt2 = mask_c.sum(axis=1)
                    rattlers = np.where(cnt2 == 0)[0]
        else:
            # no polish: direct accept if annealed alone beats best (rare)
            if va < bestV - 1e-12:
                bestV = va
                bestX = Ya.copy()
                baseX = Ya.copy()
        trial += 1
        # time-check every 32 iterations is done inside anneal; outer also checks
        if trial > 60:
            break

    # final safety: ensure finite, normalized
    bestX = _norm(bestX)
    return bestX

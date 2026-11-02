import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    A = A.astype(float, copy=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _hole_refine(h, Xo, iters=80):
    # minimize max_j <h,Xo_j> by soft-max descent
    beta = 200.0
    for it in range(iters):
        g = Xo @ h
        gm = float(g.max())
        e = np.exp(beta * (g - gm))
        s = float(e.sum())
        if s == 0 or not np.isfinite(s):
            break
        w = e / s
        gr = Xo.T @ w
        grt = gr - float(gr @ h) * h
        lr = 0.05 * (0.1 ** (it / max(1, iters - 1)))
        h = h - lr * grt
        n = np.linalg.norm(h)
        if n == 0 or not np.isfinite(n):
            break
        h /= n
    return h

def _find_hole(Xo, rng, n_cand=3000):
    d = Xo.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xo.T
    smax = S.max(axis=1)
    ib = int(np.argmin(smax))
    h = C[ib].copy()
    h = _hole_refine(h, Xo, iters=80)
    return h

def _annealed_descent(X, rng, steps, beta0, beta1, lr0, lr1, deadline):
    n, d = X.shape
    X = X.copy()
    best_local = X.copy()
    best_val = _max_cos(X)
    # momentum buffer
    vel = np.zeros_like(X)
    mu = 0.7
    for it in range(steps):
        if time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        # current true max (off diag)
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # track
        if m < best_val:
            best_val = m
            best_local = X.copy()
        # soft-max weights; clip exponent for stability
        # G currently has -2 on diag, exp will be ~0 there
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()) * 0.5
        if S == 0 or not np.isfinite(S):
            break
        Gr = (E @ X) / (2.0 * S + 1e-300)  # (n,d)
        # tangent projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # normalize per-row step to keep stable
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # Adam-like: momentum + normalized direction
        vel = mu * vel + (1 - mu) * Gt
        # step: combine vel direction
        step = lr * (Gt / nr + 0.3 * vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12))
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # occasional exact hill-climb check is done via best tracking
    return best_local, best_val

def _lp_polish(X, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _normed(X)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.6 and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) > 1200:
            idx = np.argpartition(-c, 1200)[:1200]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            best = v
            bestX = Y.copy()
            X = Y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # also keep X as bestX if diverged? keep X=bestX when stuck
        if v > best + 1e-12:
            # revert to best to stay in good basin
            X = bestX.copy()
    return bestX

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    polish_deadline = deadline
    # load starts
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
        cands.append(_normed(xb))
    if xr is not None and xr.shape == (76, 15):
        cands.append(_normed(xr))
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    best = cands[0].copy()
    bv = _max_cos(best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bv:
            best, bv = c.copy(), v
    n, d = 76, 15
    # contact analysis for targeted moves
    def contacts(X, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2)
        m = G.max()
        C = (G > m - tol)
        per = C.sum(axis=1)
        return m, per
    _, per = contacts(best)
    # order indices: rattlers first
    # basin hopping: reserve ~22s for LP polish
    hop_end = deadline - 22.0
    rng_master = np.random.default_rng(20261101)
    seeds = rng_master.integers(0, 1 << 30, size=40)
    sigmas = [0.02, 0.05, 0.09, 0.14, 0.035, 0.07, 0.11]
    trial = 0
    # quick hill-climb: try small shakes without descent first (cheap)
    while trial < 28 and time.time() < hop_end:
        rng = np.random.default_rng(int(seeds[trial % len(seeds)]) + trial * 7919)
        sigma = float(sigmas[trial % len(sigmas)])
        Y = best + sigma * rng.normal(size=(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # every other trial: hole reinsertion of 1-2 worst points
        if trial % 2 == 1:
            # find points involved in max
            G = Y @ Y.T
            np.fill_diagonal(G, -2)
            mcur = float(G.max())
            # score per point: max dot
            pmax = G.max(axis=1)
            k = int(np.argmax(pmax))
            # also occasionally second point
            ks = [k]
            if trial % 4 == 3:
                # second worst distinct
                tmp = pmax.copy(); tmp[k] = -2
                ks.append(int(np.argmax(tmp)))
            for kk in ks:
                if time.time() > hop_end:
                    break
                mask = np.ones(n, dtype=bool); mask[kk] = False
                Xo = Y[mask]
                h = _find_hole(Xo, rng, n_cand=2500)
                Y[kk] = h
        # annealed descent, short
        steps = 500 if trial < 8 else 700
        # beta schedule grows with trial to diversify
        beta0 = 80.0
        beta1 = 600.0 + (trial % 4) * 200.0
        lr0 = 0.06
        lr1 = 0.004
        Yb, vb = _annealed_descent(Y, rng, steps, beta0, beta1, lr0, lr1, hop_end)
        # exact check also on Y (end point)
        ve = _max_cos(Yb)
        if ve < bv - 1e-12:
            best, bv = Yb.copy(), ve
            _, per = contacts(best)
        trial += 1
        # early exit if big win? continue to push
    # final LP polish with remaining time
    best = _lp_polish(best, deadline)
    best = _normed(best)
    return best

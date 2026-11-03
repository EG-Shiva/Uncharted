import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _sym_score(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())

def _unfold(Y):
    Y = _norm(Y)
    return _norm(np.vstack([Y, -Y]))

def _greedy_Y(X):
    X = _norm(np.array(X, float))
    n = X.shape[0]
    m = n // 2
    G = X @ X.T
    rem = np.ones(n, dtype=bool)
    Y = []
    for _ in range(m):
        idx = np.where(rem)[0]
        if len(idx) == 0:
            break
        sub = G[np.ix_(idx, idx)]
        # most negative = most antipodal
        # argmin off-diagonal
        np.fill_diagonal(sub, 1e9)
        k = int(np.argmin(sub))
        ii = k // len(idx); jj = k % len(idx)
        i = int(idx[ii]); j = int(idx[jj])
        v = X[i] - X[j]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = X[i].copy()
            nv = np.linalg.norm(v)
        Y.append(v / nv)
        rem[i] = False; rem[j] = False
        if len(Y) >= m:
            break
    Y = np.array(Y)
    # if odd leftover (should not happen for 162)
    while Y.shape[0] < m:
        v = np.random.randn(X.shape[1]); v /= np.linalg.norm(v)
        Y = np.vstack([Y, v])
    return _norm(Y)

def _sym_obj_grad(Zflat, m, d, a, iuI, iuJ):
    Z = Zflat.reshape(m, d)
    r = np.linalg.norm(Z, axis=1, keepdims=True)
    r = np.maximum(r, 1e-12)
    Y = Z / r
    G = Y @ Y.T
    # clip for stability
    # M = max |G| off-diagonal
    vals_p = G[iuI, iuJ]
    vals_m = -vals_p
    M = float(np.maximum(vals_p.max(), vals_m.max()))
    Ep = np.exp(a * (vals_p - M))
    Em = np.exp(a * (vals_m - M))
    S = float(Ep.sum() + Em.sum()) + 1e-300
    obj = M + np.log(S) / a
    # C_ij = (Ep-Em)/S
    Cvals = (Ep - Em) / S
    # build symmetric C matrix (m x m), diag 0
    C = np.zeros((m, m), dtype=float)
    C[iuI, iuJ] = Cvals
    C[iuJ, iuI] = Cvals
    gradY = C @ Y
    # project to tangent and scale by r
    dot = (gradY * Y).sum(axis=1, keepdims=True)
    gradZ = (gradY - dot * Y) / r
    return obj, gradZ.ravel().astype(float)

def _sym_optimize(Y0, a_list, time_limit, maxiter=400):
    m, d = Y0.shape
    iu = np.triu_indices(m, 1)
    iuI, iuJ = iu[0], iu[1]
    Z = _norm(np.array(Y0, float)).ravel().copy()
    # keep Z away from zero: norms are 1
    for a in a_list:
        if time.time() > time_limit:
            break
        def fun(z):
            return _sym_obj_grad(z, m, d, float(a), iuI, iuJ)
        try:
            res = minimize(fun, Z, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': 5000})
            Z = res.x
        except Exception:
            break
        # renormalize Z to unit rows to keep scaling sane
        Zm = Z.reshape(m, d)
        Zm = _norm(Zm)
        Z = Zm.ravel().copy()
    Yopt = _norm(Z.reshape(m, d))
    return Yopt

def _smooth_opt(X, iters, a0, a1, lr0, lr1, rng, noise=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    mu = 0.8
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a1 != a0 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(a * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        V = mu * V + Grad
        X = X - lr * V
        if noise > 0:
            nm = noise * (1.0 - f)
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iuI, iuJ]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            idx = np.argpartition(-cfull, 3500)[:3500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            cnt = 3500
        I = iuI[sel]; J = iuJ[sel]
        c = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)

    m, d = 81, 8
    Y_inits = []
    # greedy pairings from incumbent and record
    try:
        Y_inits.append(_greedy_Y(X_best))
    except Exception:
        pass
    try:
        Xr = np.load("record.npy").astype(float)
        if Xr.shape == (162, 8):
            Y_inits.append(_greedy_Y(_norm(Xr)))
    except Exception:
        pass
    # greedy from slightly perturbed best (different pairing basins)
    for k in range(2):
        try:
            Xp = _norm(X_best + rng.randn(162, 8) * 0.01)
            Y_inits.append(_greedy_Y(Xp))
        except Exception:
            pass
    # random starts
    for k in range(4):
        Y_inits.append(_norm(rng.randn(m, d)))

    a_lists = [
        [40, 120, 350, 900],
        [60, 180, 500, 1200],
        [40, 120, 350, 900],
        [50, 150, 450, 1000],
        [40, 120, 350, 900],
        [60, 180, 500, 1200],
        [40, 120, 350, 900],
        [50, 150, 450, 1000],
    ]
    sym_cands = []
    sym_time_limit = t0 + 128.0
    # always keep unfolded incumbent as candidate too
    for gi, Y0 in enumerate(Y_inits):
        if time.time() > sym_time_limit - 5:
            break
        if time.time() > deadline - 45:
            break
        al = a_lists[gi % len(a_lists)]
        # tighter per-init limit
        Yopt = _sym_optimize(Y0, al, min(sym_time_limit, deadline - 42), maxiter=350)
        sc = _sym_score(Yopt)
        Xsym = _unfold(Yopt)
        # true score equals sym score by construction
        s = _max_cos(Xsym)
        sym_cands.append((s, Yopt.copy(), Xsym.copy()))
    sym_cands.sort(key=lambda z: z[0])
    # pick best symmetric unfolded; add small symmetry-breaking noise
    if len(sym_cands) > 0:
        best_sym_X = sym_cands[0][2]
    else:
        best_sym_X = X_best.copy()

    # build pool for final unconstrained phase: best symmetric + incumbent
    pool = []
    for s, Y, Xsym in sym_cands[:4]:
        pool.append(Xsym)
    pool.append(X_best.copy())
    # dedup pool to 5 max
    pool = pool[:5]

    # short unconstrained soft-max to break exact antipodality on symmetric cands
    broken = []
    for Xc in pool:
        if time.time() > deadline - 40:
            break
        Xb2 = np.array(Xc, copy=True)
        # if nearly exactly antipodal, add tiny noise to allow symmetry breaking
        # detect antipodality: check if set is closed under negation
        # just add tiny noise to symmetric candidates (not to incumbent)
        is_sym = False
        # cheap check: for first 5 points, look for near-negation
        try:
            G = Xb2 @ Xb2.T
            # count entries < -0.999
            if (G < -0.999).sum() >= 100:
                is_sym = True
        except Exception:
            pass
        if is_sym:
            Xb2 = _norm(Xb2 + rng.randn(*Xb2.shape) * 3e-4)
            # quick smooth to let symmetry break slightly
            Xb2 = _smooth_opt(Xb2, 120, 80, 700, 0.005, 0.001, rng, noise=0.0)
        broken.append(Xb2)
    if len(broken) == 0:
        broken = [X_best.copy()]

    # choose current best among broken
    for Xc in broken:
        try:
            s = _max_cos(_norm(Xc))
            if s < s_best:
                s_best = s; X_best = _norm(Xc)
        except Exception:
            pass

    # final 25-35 s unconstrained LP polishing, symmetry allowed to break
    # spend remaining time (at least 25s if available) on LP
    remain = deadline - time.time()
    # ensure we have ~28s for final polish if possible
    final_budget = min(remain - 1.5, 34.0)
    if final_budget < 5:
        final_budget = max(remain - 1.0, 1.0)
    # polish candidates in order, best first
    # sort broken by score
    scored = []
    for Xc in broken:
        try:
            scored.append((_max_cos(Xc), Xc))
        except Exception:
            pass
    scored.sort(key=lambda z: z[0])
    # first polish the most promising
    t_final_deadline = deadline - 1.0
    for si, (s0, Xc) in enumerate(scored):
        if time.time() > t_final_deadline - 2:
            break
        if si == 0:
            budget = final_budget * 0.6
        else:
            budget = final_budget * 0.4 / max(1, len(scored) - 1)
        budget = max(budget, 2.0)
        # don't exceed deadline
        budget = min(budget, t_final_deadline - time.time())
        if budget < 1.5:
            break
        Xp = _lp_polish(Xc, budget, t_deadline=t_final_deadline)
        try:
            sp = _max_cos(Xp)
        except Exception:
            continue
        if sp < s_best:
            s_best = sp; X_best = Xp.copy()
    # last full polish on global best if time remains
    if time.time() < t_final_deadline - 2:
        budget = t_final_deadline - time.time()
        if budget > 2:
            Xp = _lp_polish(X_best, min(budget, 12.0), t_deadline=t_final_deadline)
            try:
                sp = _max_cos(Xp)
                if sp < s_best:
                    s_best = sp; X_best = Xp.copy()
            except Exception:
                pass
    X_best = _norm(X_best)
    return X_best

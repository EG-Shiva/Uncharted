import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lse_relax(X0, alpha, maxiter):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    x0 = X0.ravel().copy()
    def fun(xf):
        Y = xf.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        G = X @ X.T
        dots = G[iu0, iu1]
        M = float(dots.max())
        e = np.exp(alpha * (dots - M))
        S = float(e.sum())
        loss = M + np.log(S) / alpha
        p = e / S
        W = np.zeros((n, n))
        W[iu0, iu1] = p
        W[iu1, iu0] = p
        GX = W @ X
        dg = (GX * X).sum(axis=1, keepdims=True)
        Gt = GX - dg * X
        grad = (Gt / rn).ravel()
        return float(loss), grad.astype(float)
    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 8})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = X0.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _short_anneal(X0):
    X = X0.copy()
    for alpha, mi in ((200.0, 20), (800.0, 20), (2000.0, 20)):
        X = _lse_relax(X, alpha=alpha, maxiter=mi)
    return X

def _lp_polish(X, best, seconds, t_end, radius0=5e-4, cap_pairs=2000):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    radius = radius0
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap_pairs:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap_pairs]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def _lp_duals_and_resultants(X, radius=1e-3, slack=3e-6):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Cfull = X @ X.T
    c = Cfull[iu0, iu1]
    m = float(c.max())
    sel = c > m - max(20.0 * radius, slack)
    if np.count_nonzero(sel) > 2000:
        idx = np.where(sel)[0]
        order = np.argsort(-c[idx])[:2000]
        mask = np.zeros_like(sel, dtype=bool)
        mask[idx[order]] = True
        sel = mask
    I, J = iu0[sel], iu1[sel]
    p = len(I)
    csel = c[sel]
    lam = None
    try:
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        if res.status == 0:
            cand = None
            try:
                if hasattr(res, "ineqlin") and res.ineqlin is not None:
                    try:
                        cand = np.asarray(res.ineqlin.marginals).ravel()
                    except Exception:
                        cand = None
                if cand is None and isinstance(res, dict):
                    pass
            except Exception:
                cand = None
            # try dict-style access
            if cand is None:
                try:
                    v = res.get("ineqlin", None)
                    if v is not None:
                        cand = np.asarray(v.marginals if hasattr(v, "marginals") else v).ravel()
                except Exception:
                    cand = None
            if cand is not None and cand.shape[0] == p:
                lam = np.abs(cand)
    except Exception:
        lam = None
    if lam is None or lam.shape[0] != p or (np.all(lam == 0)):
        # fallback: weight by closeness to max
        sl = m - csel
        lam = np.exp(-sl / 5e-7) + 1e-3
    # compute resultants
    R = np.zeros((n, d))
    dots = np.sum(X[I] * X[J], axis=1)
    for k in range(p):
        l = float(lam[k])
        if l == 0 or not np.isfinite(l):
            continue
        i = int(I[k]); j = int(J[k])
        dd = float(dots[k])
        R[i] += l * (X[j] - dd * X[i])
        R[j] += l * (X[i] - dd * X[j])
    # project to tangent
    proj = (R * X).sum(axis=1, keepdims=True)
    R -= proj * X
    norms = np.linalg.norm(R, axis=1)
    return I, J, lam, R, norms, m

def solve():
    t0 = time.time()
    SEARCH_END = t0 + 120.0
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 73, 13
    Xbest = None
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if Xbest is None:
                    Xbest = v.copy()
                else:
                    if _max_cos(v) < _max_cos(Xbest):
                        Xbest = v.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = rng.normal(size=(n, d))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best_val = _max_cos(Xbest)

    thetas = [1e-4, 3e-4, 1e-3, 3e-3]
    set_sizes = [3, 5, 8]
    modes = ["plus", "minus", "orth"]

    # outer loop: recompute stresses after each improvement
    n_outer = 0
    while time.time() < SEARCH_END and n_outer < 6:
        n_outer += 1
        try:
            I, J, lam, R, norms, mcur = _lp_duals_and_resultants(Xbest, radius=1e-3, slack=3e-6)
        except Exception:
            R = np.zeros((n, d)); norms = np.zeros(n)
            mcur = best_val
        order = np.argsort(-norms)
        # force include 4 and 22
        forced = [4, 22]
        improved_this_round = False
        for ksz in set_sizes:
            if time.time() > SEARCH_END or time.time() > DEADLINE - 15.0:
                break
            top = list(order[:ksz])
            for f in forced:
                if f not in top:
                    top.append(f)
            # trim to ksz+2 max (3-8 plus forced -> up to 10); keep as is but cap 10
            if len(top) > 10:
                top = top[:10]
            # precompute unit R dirs
            Rdir = {}
            for idx in top:
                v = R[idx].copy()
                # ensure tangent
                v = v - float(v @ Xbest[idx]) * Xbest[idx]
                nv = float(np.linalg.norm(v))
                if nv < 1e-15 or not np.isfinite(nv):
                    # random tangent fallback
                    g = rng.normal(size=(d,))
                    g = g - float(g @ Xbest[idx]) * Xbest[idx]
                    ng = float(np.linalg.norm(g))
                    if ng < 1e-12:
                        g = np.zeros(d); g[0] = 1.0
                        g = g - float(g @ Xbest[idx]) * Xbest[idx]
                        ng = float(np.linalg.norm(g))
                    v = g / max(ng, 1e-12)
                else:
                    v = v / nv
                Rdir[idx] = v
            for th in thetas:
                if time.time() > SEARCH_END or time.time() > DEADLINE - 15.0:
                    break
                for mode in modes:
                    if time.time() > SEARCH_END or time.time() > DEADLINE - 12.0:
                        break
                    Xtry = Xbest.copy()
                    for idx in top:
                        if mode == "plus":
                            dv = Rdir[idx]
                        elif mode == "minus":
                            dv = -Rdir[idx]
                        else:
                            g = rng.normal(size=(d,))
                            g = g - float(g @ Xbest[idx]) * Xbest[idx]
                            # orthogonalize against Rdir
                            g = g - float(g @ Rdir[idx]) * Rdir[idx]
                            # re-tangent (still tangent)
                            g = g - float(g @ Xbest[idx]) * Xbest[idx]
                            ng = float(np.linalg.norm(g))
                            if ng < 1e-12 or not np.isfinite(ng):
                                dv = Rdir[idx]
                            else:
                                dv = g / ng
                        # geodesic kick
                        ct = float(np.cos(th)); st = float(np.sin(th))
                        nv = ct * Xbest[idx] + st * dv
                        nv /= max(float(np.linalg.norm(nv)), 1e-12)
                        Xtry[idx] = nv
                    # short LSE anneal
                    try:
                        Xtry2 = _short_anneal(Xtry)
                    except Exception:
                        Xtry2 = Xtry
                    v0 = _max_cos(Xtry2)
                    # LP repolish short (1.2s cap each)
                    t_rem = min(SEARCH_END, DEADLINE - 8.0) - time.time()
                    if t_rem > 0.4:
                        try:
                            Xtry2, v0 = _lp_polish(Xtry2, v0, seconds=min(1.2, t_rem - 0.05), t_end=min(SEARCH_END, DEADLINE - 5.0))
                        except Exception:
                            pass
                    if v0 < best_val - 1e-12:
                        Xbest = Xtry2.copy()
                        best_val = float(v0)
                        improved_this_round = True
                        # recompute stresses immediately (break to outer)
                        break
                if improved_this_round:
                    break
            if improved_this_round:
                # continue outer to recompute R with new topology
                continue
        if not improved_this_round:
            break
    # final squeeze if time remains
    if time.time() < DEADLINE - 4.0:
        try:
            Xbest, best_val = _lp_polish(Xbest, best_val, seconds=max(0.5, min(8.0, (DEADLINE - 3.0) - time.time())), t_end=DEADLINE - 2.0, radius0=4e-4)
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

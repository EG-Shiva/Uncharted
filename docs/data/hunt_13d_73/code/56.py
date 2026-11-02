import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, t_end, r_init=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X, 1e9
    radius = r_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds or now >= t_end or radius < 1e-14:
            break
        Gfull = (X @ X.T)[iu0]
        m = float(np.max(Gfull))
        thresh = max(20.0 * radius, 1e-9)
        sel = Gfull > m - thresh
        cnt = int(np.sum(sel))
        if cnt > 900:
            thr = np.partition(Gfull, -900)[-900]
            sel = Gfull >= thr
            idx = np.where(sel)[0]
            if len(idx) > 1100:
                order = np.argsort(-Gfull[idx])[:1100]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        csub = Gfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def _lse_anneal(X, steps, a0, a1, s0, s1, rng):
    n, d = X.shape
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        # linear in log for alpha 200->2000
        la0, la1 = np.log(a0), np.log(a1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        step = s0 + (s1 - s0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # avoid overflow: (G-m)<=0
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        # sparsify extremely far pairs for stability
        # keep all (n small)
        grad = W @ X
        proj = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - proj * X
        gn = float(np.max(np.sum(grad * grad, axis=1)))
        gn = np.sqrt(gn)
        if not np.isfinite(gn) or gn < 1e-14:
            continue
        grad *= (step / gn)
        X = X - grad
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _dual_resultants(X, slack=3e-6, radius=8e-4):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Gfull = (X @ X.T)[iu0, iu1]
    m = float(np.max(Gfull))
    sel = Gfull > m - slack
    I = iu0[sel]; J = iu1[sel]
    csub = Gfull[sel]
    p = len(I)
    lam = np.ones(p, dtype=float)
    # solve trust-region LP once to get duals
    try:
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        got = False
        marg = None
        try:
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                # res.ineqlin can be OptimizeResult with marginals
                if hasattr(res.ineqlin, 'marginals'):
                    marg = np.asarray(res.ineqlin.marginals, dtype=float)
                    got = True
                elif isinstance(res.ineqlin, dict) and 'marginals' in res.ineqlin:
                    marg = np.asarray(res.ineqlin['marginals'], dtype=float)
                    got = True
            if not got and hasattr(res, 'marginals'):
                marg = np.asarray(res.marginals, dtype=float)[-p:] if len(np.asarray(res.marginals)) >= p else None
                got = marg is not None
        except Exception:
            got = False
        if got and marg is not None and marg.shape[0] == p:
            # duals for A_ub x <= b, minimization => marginals <=0; lambda = -marginals
            lam2 = -marg
            # fallback to abs if sign convention differs
            if np.max(lam2) < 1e-12:
                lam2 = np.abs(marg)
            lam2 = np.maximum(lam2, 0.0)
            if np.all(np.isfinite(lam2)) and np.max(lam2) > 0:
                lam = lam2
    except Exception:
        pass
    # compute R_i = sum_j lam_ij (X_j - c_ij X_i)
    R = np.zeros((n, d), dtype=float)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        c = float(csub[k]); l = float(lam[k])
        if not np.isfinite(l) or l < 0:
            l = 0.0
        R[i] += l * (X[j] - c * X[i])
        R[j] += l * (X[i] - c * X[j])
    # project to tangent (remove radial part numerically)
    rad = np.sum(R * X, axis=1, keepdims=True)
    R = R - rad * X
    norms = np.linalg.norm(R, axis=1)
    return R, norms, lam, I, J, csub, m

def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    SEARCH_END = t0 + 120.0
    rng = np.random.RandomState(0)
    n, d = 73, 13
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape != (n, d):
                continue
            nr = np.linalg.norm(v, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            v = v / nr
            if not np.all(np.isfinite(v)):
                continue
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        best = rng.randn(n, d)
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    Xbest = best.copy()

    # compute dual stresses at incumbent
    try:
        R, norms, lam, I, J, csub, m = _dual_resultants(Xbest, slack=3e-6, radius=8e-4)
    except Exception:
        R = np.zeros_like(Xbest); norms = np.zeros(n)
    # order by resultant norm
    order = np.argsort(-norms)
    # candidate pool: top 12 + points 4,22 explicitly
    pool = list(order[:12])
    for q in [4, 22]:
        if q not in pool:
            pool.append(q)
    # precompute Rhat
    Rhat = np.zeros_like(R)
    for i in range(n):
        nn = float(np.linalg.norm(R[i]))
        if nn > 1e-15 and np.all(np.isfinite(R[i])):
            Rhat[i] = R[i] / nn
        else:
            # random tangent fallback
            v = rng.randn(d)
            v = v - float(v @ Xbest[i]) * Xbest[i]
            vn = float(np.linalg.norm(v))
            if vn < 1e-12:
                v = np.ones(d); v = v - float(v @ Xbest[i]) * Xbest[i]
                vn = float(np.linalg.norm(v))
            Rhat[i] = v / max(vn, 1e-12)

    angles = [1e-4, 3e-4, 1e-3, 3e-3]
    trial = 0
    # short LSE params: alpha 200->2000
    while time.time() < SEARCH_END and time.time() < DEADLINE - 25:
        trial += 1
        # choose k in 3..8 cycling + random
        k = 3 + (trial % 6)  # 3..8
        # top-k by norms, ensure 4,22 included
        topk = list(order[:k])
        for q in [4, 22]:
            if q not in topk:
                # replace smallest
                topk[-1] = q
        # randomize: occasionally swap one with pool random
        if rng.rand() < 0.5:
            cand = pool[rng.randint(0, len(pool))]
            if cand not in topk:
                topk[rng.randint(0, k)] = int(cand)
        S = [int(x) for x in topk]
        ang = angles[trial % len(angles)] if rng.rand() < 0.7 else angles[rng.randint(0, len(angles))]
        # jitter angle slightly
        ang = float(ang * (0.8 + 0.4 * rng.rand()))
        X = Xbest.copy()
        # build kick directions
        for idx_pos, i in enumerate(S):
            mode = rng.rand()
            xi = Xbest[i].copy()
            rh = Rhat[i].copy()
            # ensure tangent
            rh = rh - float(rh @ xi) * xi
            rn = float(np.linalg.norm(rh))
            if rn > 1e-12:
                rh = rh / rn
            else:
                v = rng.randn(d); v = v - float(v @ xi) * xi
                rh = v / max(float(np.linalg.norm(v)), 1e-12)
            if mode < 0.3:
                u = rh  # +R
            elif mode < 0.6:
                u = -rh  # -R
            else:
                # random tangent orthogonal to R
                v = rng.randn(d)
                v = v - float(v @ xi) * xi - float(v @ rh) * rh
                # re-orthogonalize vs xi
                v = v - float(v @ xi) * xi
                vn = float(np.linalg.norm(v))
                if vn < 1e-9 or not np.isfinite(vn):
                    v = rng.randn(d)
                    v = v - float(v @ xi) * xi - float(v @ rh) * rh
                    v = v - float(v @ xi) * xi
                    vn = float(np.linalg.norm(v))
                    if vn < 1e-9:
                        u = rh
                    else:
                        u = v / vn
                else:
                    u = v / vn
            # occasional sign flip
            if rng.rand() < 0.2:
                u = -u
            th = float(ang * (0.7 + 0.6 * rng.rand()))
            X[i] = np.cos(th) * xi + np.sin(th) * u
        # renormalize (should already be unit)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        # short LSE anneal alpha 200->2000
        try:
            X = _lse_anneal(X, steps=160, a0=200.0, a1=2000.0, s0=0.02, s1=0.002, rng=rng)
        except Exception:
            continue
        try:
            v0 = _max_cos(X)
        except Exception:
            continue
        if not np.isfinite(v0):
            continue
        # accept only if exact max improves after repolish; screen first
        # if far worse, skip LP to save time
        if v0 > best_val + 0.004:
            continue
        # short LP repolish
        tl = min(1.2, SEARCH_END - (time.time() - t0), DEADLINE - 25 - (time.time() - t0))
        if tl < 0.25:
            # no time for LP; check direct improvement
            if v0 < best_val:
                Xbest = X.copy(); best_val = v0
                # refresh resultants
                try:
                    R, norms, lam, I, J, csub, m = _dual_resultants(Xbest, slack=3e-6, radius=8e-4)
                    order = np.argsort(-norms)
                    pool = list(order[:12])
                    for q in [4, 22]:
                        if q not in pool:
                            pool.append(q)
                    for i in range(n):
                        nn = float(np.linalg.norm(R[i]))
                        if nn > 1e-15:
                            Rhat[i] = R[i] / nn
                except Exception:
                    pass
            continue
        try:
            Y, vy = _lp_polish(X, seconds=tl, t_end=min(SEARCH_END, DEADLINE - 20), r_init=8e-4)
        except Exception:
            continue
        if np.all(np.isfinite(Y)) and vy < best_val:
            Xbest = Y.copy(); best_val = float(vy)
            # refresh duals after improvement
            try:
                R, norms, lam, I, J, csub, m = _dual_resultants(Xbest, slack=3e-6, radius=8e-4)
                order = np.argsort(-norms)
                pool = list(order[:12])
                for q in [4, 22]:
                    if q not in pool:
                        pool.append(q)
                Rhat = np.zeros_like(R)
                for i in range(n):
                    nn = float(np.linalg.norm(R[i]))
                    if nn > 1e-15 and np.all(np.isfinite(R[i])):
                        rh = R[i] / nn
                        rh = rh - float(rh @ Xbest[i]) * Xbest[i]
                        rnn = float(np.linalg.norm(rh))
                        if rnn > 1e-12:
                            Rhat[i] = rh / rnn
                        else:
                            Rhat[i] = R[i] / nn
                    else:
                        v = rng.randn(d); v = v - float(v @ Xbest[i]) * Xbest[i]
                        Rhat[i] = v / max(float(np.linalg.norm(v)), 1e-12)
            except Exception:
                pass
        # early exit if time
        if time.time() >= SEARCH_END:
            break
    # final repolish with remaining time
    remain = DEADLINE - time.time() - 1.0
    if remain < 2.0:
        remain = 2.0
    remain = min(remain, 30.0)
    try:
        Xbest, best_val = _lp_polish(Xbest, seconds=remain, t_end=DEADLINE - 0.5, r_init=8e-4)
    except Exception:
        pass
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xbest = Xbest / nr
    return Xbest

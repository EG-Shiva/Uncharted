import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return X / n

def _build_B(X):
    n, d = X.shape
    k = d - 1
    B = np.empty((n, d, k), dtype=np.float64)
    for i in range(n):
        xi = X[i]
        # null space of xi via SVD of 1x6
        _, _, Vh = np.linalg.svd(xi.reshape(1, -1), full_matrices=True)
        B[i] = Vh[1:, :].T
    # ensure tangent: remove any normal component numerically
    # B already orthogonal by construction
    return B

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    n, d = 250, 6
    k5 = d - 1
    N = n * k5

    best0 = None
    best0_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm(z)
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best0_v:
                        best0_v = v
                        best0 = z.copy()
        except Exception:
            continue
    if best0 is None:
        rng0 = np.random.default_rng(0)
        best0 = _norm(rng0.standard_normal((n, d)))
        best0_v = _max_cos(best0)

    Xcur = best0.copy()
    Mcur = float(best0_v)
    Mbest = float(Mcur)
    Xbest = Xcur.copy()

    # initial active set: all pairs within 1e-5 of max
    try:
        G0 = Xcur @ Xcur.T
        iu0_all = np.triu_indices(n, 1)
        gv0 = G0[iu0_all]
        M0 = float(np.max(gv0))
        thr0 = M0 - 1e-5
        sel0 = np.where(gv0 >= thr0)[0]
        active_keys = set()
        for s in sel0.tolist():
            i = int(iu0_all[0][s]); j = int(iu0_all[1][s])
            if i > j:
                i, j = j, i
            active_keys.add(i * n + j)
        if len(active_keys) == 0:
            # fallback: top 1600
            idx = np.argsort(-gv0)[:1600]
            for s in idx.tolist():
                i = int(iu0_all[0][s]); j = int(iu0_all[1][s])
                if i > j:
                    i, j = j, i
                active_keys.add(i * n + j)
    except Exception:
        active_keys = set()

    streak = {kk: 0 for kk in active_keys}
    rho = 1e-3
    RHO_MIN_CERT = 1e-5
    RHO_MAX = 1e-2
    RHO_MIN = 1e-5

    obj = np.zeros(N + 1, dtype=np.float64)
    obj[-1] = 1.0
    ar5 = np.arange(k5)

    it = 0
    # main SQP loop
    while time.time() < t_end - 3.0 and it < 300:
        it += 1
        m = float(Mcur)
        # build tangent basis at current point
        try:
            B = _build_B(Xcur)
        except Exception:
            break
        if len(active_keys) == 0:
            # rebuild from current max
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            gv = G[iu0_all]
            Mtmp = float(np.max(gv))
            sel = np.where(gv >= Mtmp - 1e-5)[0]
            for s in sel.tolist():
                i = int(iu0_all[0][s]); j = int(iu0_all[1][s])
                if i > j:
                    i, j = j, i
                kk = i * n + j
                if kk not in streak:
                    streak[kk] = 0
                active_keys.add(kk)
            if len(active_keys) == 0:
                break
            continue

        keys_list = sorted(active_keys)
        p = len(keys_list)
        I = np.empty(p, dtype=np.int64)
        J = np.empty(p, dtype=np.int64)
        for tii, kk in enumerate(keys_list):
            I[tii] = kk // n
            J[tii] = kk % n

        XI = Xcur[I]
        XJ = Xcur[J]
        c = np.sum(XI * XJ, axis=1)

        BI = B[I]
        BJ = B[J]
        vI = np.einsum('pd,pdk->pk', XJ, BI)
        vJ = np.einsum('pd,pdk->pk', XI, BJ)

        # build sparse A_ub: a^T u - t <= -c
        try:
            col_mat = np.empty((p, 2 * k5 + 1), dtype=np.int64)
            col_mat[:, :k5] = I[:, None] * k5 + ar5[None, :]
            col_mat[:, k5:2 * k5] = J[:, None] * k5 + ar5[None, :]
            col_mat[:, 2 * k5] = N
            data_mat = np.empty((p, 2 * k5 + 1), dtype=np.float64)
            data_mat[:, :k5] = vI
            data_mat[:, k5:2 * k5] = vJ
            data_mat[:, 2 * k5] = -1.0
            rows = np.repeat(np.arange(p), 2 * k5 + 1)
            cols = col_mat.ravel()
            vals = data_mat.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, N + 1)).tocsr()
            b_ub = -c
            bounds = [(-rho, rho)] * N + [(None, float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception:
            rho = max(rho * 0.5, RHO_MIN * 0.5)
            if rho < 1e-12:
                break
            continue

        if res.status != 0 or res.x is None:
            rho *= 0.5
            if rho < RHO_MIN and time.time() > t_end - 10.0:
                break
            if rho < 1e-12:
                break
            # if infeasible at tiny rho, treat as certified
            if rho <= RHO_MIN + 1e-18:
                # check prediction anyway
                pass
            continue

        xopt = np.asarray(res.x, dtype=np.float64)
        uvec = xopt[:-1]
        t_pred = float(xopt[-1])
        pred_gain = m - t_pred

        # dual multipliers / activity
        inactive = np.zeros(p, dtype=bool)
        try:
            marg = None
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                # OptimizeResult with marginals
                try:
                    marg = np.asarray(res.ineqlin.marginals, dtype=np.float64)
                except Exception:
                    try:
                        marg = np.asarray(res.ineqlin["marginals"], dtype=np.float64)
                    except Exception:
                        marg = None
            if marg is not None and marg.shape[0] == p:
                inactive = np.abs(marg) < 1e-9
            else:
                raise ValueError("no marginals")
        except Exception:
            # fallback: linear slack
            try:
                U = uvec.reshape(n, k5)
                UI = U[I]
                UJ = U[J]
                lin_val = c + np.sum(vI * UI, axis=1) + np.sum(vJ * UJ, axis=1)
                slack = t_pred - lin_val
                inactive = slack > 1e-12
            except Exception:
                inactive = np.zeros(p, dtype=bool)

        # update streaks
        for tii, kk in enumerate(keys_list):
            if bool(inactive[tii]):
                streak[kk] = int(streak.get(kk, 0)) + 1
            else:
                streak[kk] = 0

        # certified optimum check at small radius
        if pred_gain <= 1e-12 and rho <= RHO_MIN_CERT + 1e-18:
            # LP predicts t >= m-1e-12 : certified local optimum
            break

        if pred_gain <= 1e-14 and rho <= RHO_MIN:
            # no descent even at cert radius but not quite threshold -> shrink & recheck
            # force one more solve at RHO_MIN_CERT (already here) then stop
            break

        # line search over alpha
        try:
            U = uvec.reshape(n, k5)
            Dm = np.einsum('ndk,nk->nd', B, U)
        except Exception:
            rho *= 0.5
            continue

        best_alpha = 0.0
        best_Xt = None
        best_mt = m
        for alpha in (0.25, 0.5, 1.0, 2.0):
            if time.time() > t_end - 2.0:
                break
            Y = Xcur + alpha * Dm
            Yn = np.linalg.norm(Y, axis=1, keepdims=True)
            np.maximum(Yn, 1e-12, out=Yn)
            Y = Y / Yn
            v = _max_cos(Y)
            if v < best_mt - 1e-15:
                best_mt = v
                best_alpha = alpha
                best_Xt = Y

        if best_Xt is not None and best_mt < m - 1e-14:
            Xcur = best_Xt
            Mcur = float(best_mt)
            if Mcur < Mbest:
                Mbest = float(Mcur)
                Xbest = Xcur.copy()
            # adapt rho up
            rho = min(rho * 1.5, RHO_MAX)
            # add every pair within 1e-6 of new max
            try:
                Gn = Xcur @ Xcur.T
                np.fill_diagonal(Gn, -2.0)
                mnew = float(np.max(Gn))
                # find pairs >= mnew-1e-6
                # use triu indices
                gv = Gn[iu0_all]
                sel = np.where(gv >= mnew - 1e-6)[0]
                for s in sel.tolist():
                    i = int(iu0_all[0][s]); j = int(iu0_all[1][s])
                    if i > j:
                        i, j = j, i
                    kk = i * n + j
                    if kk not in active_keys:
                        active_keys.add(kk)
                        streak[kk] = 0
            except Exception:
                pass
            # drop pairs with zero dual for 3 consecutive iterations
            todrop = [kk for kk in list(active_keys) if streak.get(kk, 0) >= 3]
            # keep at least ~200 pairs to avoid collapse
            if len(active_keys) - len(todrop) >= 200:
                for kk in todrop:
                    active_keys.discard(kk)
                    streak.pop(kk, None)
            else:
                # reset streaks to avoid mass drop
                for kk in todrop:
                    streak[kk] = 0
        else:
            # rejected: shrink trust region, drop persistently inactive
            rho *= 0.5
            if rho < RHO_MIN:
                rho = RHO_MIN
                # if prediction also shows no progress, certified
                if pred_gain <= 1e-12:
                    break
                # else drop and continue; avoid infinite loop at min rho
                # if many consecutive rejects, break
                # count rejects via it window: just break if tiny rho and no gain for a while
                # we let loop continue a few more times but limit
                pass
            todrop = [kk for kk in list(active_keys) if streak.get(kk, 0) >= 3]
            if len(active_keys) - len(todrop) >= 200:
                for kk in todrop:
                    active_keys.discard(kk)
                    streak.pop(kk, None)
            if rho <= RHO_MIN + 1e-18 and pred_gain <= 5e-12:
                # no usable descent
                # try to avoid stall: if rho at min and still rejected, break after a few
                # check time and iteration: break if pred tiny
                if pred_gain <= 1e-11:
                    break
            if rho < 1e-12:
                break

    out = _norm(np.asarray(Xbest if Mbest <= best0_v else best0, dtype=np.float64))
    if not np.all(np.isfinite(out)):
        out = _norm(best0)
    # strict validity: no zero rows
    nr = np.linalg.norm(out, axis=1)
    if np.any(nr < 1e-9) or np.any(~np.isfinite(nr)):
        out = _norm(best0)
    return out

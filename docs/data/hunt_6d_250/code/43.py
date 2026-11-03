import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_Q(X):
    n, d = X.shape
    k = d - 1
    Q = np.empty((n, d, k), dtype=np.float64)
    I6 = np.eye(d)
    cols_cache = {}
    for i in range(n):
        xi = X[i]
        am = int(np.argmax(np.abs(xi)))
        if am not in cols_cache:
            cols = [j for j in range(d) if j != am]
            cols_cache[am] = cols
        else:
            cols = cols_cache[am]
        E = I6[:, cols]  # 6x5
        proj = xi @ E  # 5
        E = E - np.outer(xi, proj)
        Qq, _ = np.linalg.qr(E)
        # ensure tangent (remove any residual normal component)
        # Qq already spans E which is orthogonal to xi
        Q[i] = Qq
    return Q

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    n, d = 250, 6
    k5 = 5
    N = n * k5
    NV = N + 1
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                zn = np.linalg.norm(z, axis=1, keepdims=True)
                zn = np.maximum(zn, 1e-12)
                z = z / zn
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best_val:
                        best_val = v
                        best = z.copy()
        except Exception:
            continue
    if best is None:
        rng = np.random.default_rng(0)
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = float(best_val)
    # initial active set: all pairs within 1e-5 of max
    iu0 = np.triu_indices(n, 1)
    G0 = cur @ cur.T
    gv0 = G0[iu0]
    M0 = float(np.max(gv0))
    mask0 = gv0 >= M0 - 1e-5
    I = iu0[0][mask0].astype(np.int64).copy()
    J = iu0[1][mask0].astype(np.int64).copy()
    # safety cap at 12000 tightest
    if I.size > 12000:
        gsel = gv0[mask0]
        ord_ = np.argsort(-gsel)[:12000]
        I = I[ord_]; J = J[ord_]
    streak = np.zeros(I.size, dtype=np.int64)
    rho = 8e-4
    RHO_MIN = 1e-5
    RHO_MAX = 1e-2
    alphas = (1.0, 2.0, 0.5, 0.25)
    # map for quick membership when adding
    max_iter = 80
    it = 0
    # precompute arange5
    ar5 = np.arange(k5)
    while it < max_iter and remaining() > 12.0:
        it += 1
        M = float(cur_val)
        p = I.size
        if p == 0:
            # rebuild from current
            G = cur @ cur.T
            gv = G[iu0]
            M = float(np.max(gv))
            cur_val = M
            msk = gv >= M - 1e-5
            I = iu0[0][msk].astype(np.int64).copy()
            J = iu0[1][msk].astype(np.int64).copy()
            streak = np.zeros(I.size, dtype=np.int64)
            p = I.size
            if p == 0:
                break
        # build tangent bases at cur
        try:
            Q = _build_Q(cur)
        except Exception:
            break
        XI = cur[I]
        XJ = cur[J]
        cij = np.sum(XI * XJ, axis=1)
        QI = Q[I]  # p,6,5
        QJ = Q[J]
        # vi = QI^T XJ , vj = QJ^T XI
        vi = np.einsum('pdk,pd->pk', QI, XJ)
        vj = np.einsum('pdk,pd->pk', QJ, XI)
        # build sparse A_ub
        # rows/cols/vals
        try:
            # blocks
            rI = np.repeat(np.arange(p), k5)
            cI = (I[:, None] * k5 + ar5[None, :]).ravel()
            vI = vi.ravel()
            rJ = np.repeat(np.arange(p), k5)
            cJ = (J[:, None] * k5 + ar5[None, :]).ravel()
            vJ = vj.ravel()
            rT = np.arange(p)
            cT = np.full(p, N, dtype=np.int64)
            vT = np.full(p, -1.0, dtype=np.float64)
            rows = np.concatenate([rI, rJ, rT])
            cols = np.concatenate([cI, cJ, cT]).astype(np.int64)
            vals = np.concatenate([vI, vJ, vT])
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, NV)).tocsr()
            b_ub = -cij
            obj = np.zeros(NV, dtype=np.float64)
            obj[-1] = 1.0
            bounds = [(-rho, rho)] * N + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception:
            rho = max(rho * 0.5, RHO_MIN)
            if rho <= RHO_MIN + 1e-12:
                # try continue
                pass
            continue
        if res.status != 0:
            rho = max(rho * 0.5, RHO_MIN)
            # if already at min and infeasible, break
            if res.status != 0 and rho <= RHO_MIN + 1e-15 and remaining() < 30:
                break
            continue
        xopt = np.asarray(res.x, dtype=np.float64).ravel()
        uopt = xopt[:-1]
        t_pred = float(xopt[-1])
        pred_gain = M - t_pred
        # dual / slack based streak update
        # compute predicted values
        # uI = uopt[I*5:(I*5+5)]
        try:
            U = uopt.reshape(n, k5)
            UI = U[I]  # p,5
            UJ = U[J]
            pred_vals = cij + np.sum(vi * UI, axis=1) + np.sum(vj * UJ, axis=1)
            slack = t_pred - pred_vals  # >=0 ideally
        except Exception:
            slack = np.zeros(p)
        # try to get true duals if available
        use_dual = False
        dual = None
        try:
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                mg = getattr(res.ineqlin, "marginals", None)
                if mg is not None:
                    dual = np.asarray(mg, dtype=np.float64).ravel()
                    if dual.size == p:
                        use_dual = True
        except Exception:
            use_dual = False
        if use_dual:
            is_zero = np.abs(dual) < 1e-9
        else:
            is_zero = slack > 1e-12
        # update streak (will apply after accept/drop decision; but update now)
        # we keep streak array aligned with I,J
        streak = np.where(is_zero, streak + 1, 0)
        if pred_gain < 1e-12:
            # no predicted improvement at this rho
            if rho <= RHO_MIN + 1e-15:
                # certified local optimum
                break
            else:
                rho = max(rho * 0.5, RHO_MIN)
                # drop long-zero pairs before retry
                drop_m = streak >= 3
                if np.any(drop_m):
                    keep = ~drop_m
                    I = I[keep]; J = J[keep]; streak = streak[keep]
                    # reset? keep streak for kept (already updated)
                    # cap: keep streak as is
                    pass
                continue
        # predicted improvement: line search on retraction
        D = np.einsum('ndk,nk->nd', Q, U.reshape(n, k5))
        best_a_val = M
        best_a_X = None
        for a in alphas:
            Y = cur + a * D
            nn = np.linalg.norm(Y, axis=1, keepdims=True)
            np.maximum(nn, 1e-12, out=nn)
            Yn = Y / nn
            v = _max_cos(Yn)
            if v < best_a_val:
                best_a_val = v
                best_a_X = Yn
        if best_a_X is not None and best_a_val < M - 1e-13:
            # accept
            cur = best_a_X
            cur_val = float(best_a_val)
            if cur_val < best_val:
                best_val = float(cur_val)
                best = cur.copy()
            # adapt rho up
            rho = min(rho * 1.5, RHO_MAX)
            # active set maintenance: drop zero-streak>=3, add new within 1e-6
            drop_m = streak >= 3
            if np.any(drop_m):
                keep = ~drop_m
                I = I[keep]; J = J[keep]; streak = streak[keep]
            # find new pairs within 1e-6 of new max
            Gnew = cur @ cur.T
            gvnew = Gnew[iu0]
            add_mask = gvnew >= cur_val - 1e-6
            # existing set lookup
            # build set of existing (use python set for p~few k, iu size 31125 -> ok)
            exist = set(zip(I.tolist(), J.tolist()))
            ni0 = iu0[0][add_mask]
            nj0 = iu0[1][add_mask]
            newI = []
            newJ = []
            for a_, b_ in zip(ni0.tolist(), nj0.tolist()):
                if (a_, b_) not in exist:
                    newI.append(a_); newJ.append(b_)
            if newI:
                newI = np.asarray(newI, dtype=np.int64)
                newJ = np.asarray(newJ, dtype=np.int64)
                I = np.concatenate([I, newI])
                J = np.concatenate([J, newJ])
                streak = np.concatenate([streak, np.zeros(len(newI), dtype=np.int64)])
                # optional cap: if too large, keep tightest? evaluate tightness on new cur
                if I.size > 12000:
                    c_all = np.sum(cur[I] * cur[J], axis=1)
                    ord2 = np.argsort(-c_all)[:12000]
                    I = I[ord2]; J = J[ord2]; streak = streak[ord2]
        else:
            # reject, shrink
            rho = max(rho * 0.5, RHO_MIN)
            # still drop dead constraints to improve model
            drop_m = streak >= 3
            if np.any(drop_m):
                keep = ~drop_m
                I = I[keep]; J = J[keep]; streak = streak[keep]
            if rho <= RHO_MIN + 1e-15:
                # check certification next loop; if still no gain will stop
                # avoid infinite loop: if we rejected at min rho, break if no progress for a while
                # continue to let pred_gain<1e-12 trigger break
                pass
        # loop
    out = best.copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    am = np.argmax(np.abs(X), axis=1)
    eye = np.eye(dm)
    for i in range(n):
        xi = X[i]
        a = int(am[i])
        mask = np.arange(d) != a
        xj = xi[mask]  # (12,)
        M = -xi[:, None] * xj[None, :]
        M[mask, :] += eye
        Q, _ = np.linalg.qr(M, mode='reduced')
        # ensure tangent (fix sign drift)
        B[i] = Q
    return B

def solve():
    t0 = time.time()
    SQP_BUDGET = 115.0
    # load start
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.shape(xb) == (99, 13):
        cands.append(_norm_rows(np.asarray(xb, float)))
    if xr is not None and np.shape(xr) == (99, 13):
        cands.append(_norm_rows(np.asarray(xr, float)))
    if len(cands) == 0:
        rng = np.random.default_rng(0)
        cands.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    X = cands[bi].copy()
    m_cur = float(vals[bi])
    best_X = X.copy()
    best_m = m_cur
    n, d = 99, 13
    NVAR = n * (d - 1)  # 1188
    NCOL = NVAR + 1
    iu0, iu1 = np.triu_indices(n, 1)
    ar12 = np.arange(d - 1)
    r = 3e-4
    r_min = 1e-8
    # main homotopy
    it = 0
    while time.time() - t0 < SQP_BUDGET:
        it += 1
        if r < r_min:
            break
        # epsilon tied to trust radius: 0.01*r, floored
        eps = max(0.01 * r, 5e-9)
        # cap eps to plateau scale initially
        if it == 1:
            eps = 3e-6
        else:
            # allow slight growth if starved (handled below), else geometric shrink via r
            pass
        G = X @ X.T
        c_all = G[iu0, iu1]
        m_cur = float(np.max(c_all))
        slack = m_cur - c_all
        sel = slack < eps
        cnt = int(np.sum(sel))
        # ensure enough constraints for balance, ensure at most 1200
        if cnt < 350:
            # widen to at least 600 tightest
            order = np.argsort(c_all)  # ascending
            need = min(600, c_all.size)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[-need:]] = True
            sel = sel | sel2
            cnt = int(np.sum(sel))
        if cnt > 1200:
            # keep tightest 1200
            idx_sel = np.where(sel)[0]
            c_sel_tmp = c_all[idx_sel]
            keep = np.argsort(-c_sel_tmp)[:1200]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx_sel[keep]] = True
            sel = sel2
            cnt = 1200
        if cnt == 0:
            break
        I = iu0[sel].astype(np.int64)
        J = iu1[sel].astype(np.int64)
        c_sel = c_all[sel]
        p = cnt
        # tangent bases at current X
        B = _tangent_bases(X)  # (99,13,12)
        # coefficients aI = xJ^T B_I, aJ = xI^T B_J
        XI = X[I]  # (p,13)
        XJ = X[J]
        BI = B[I]  # (p,13,12)
        BJ = B[J]
        aI = np.einsum('pi,pij->pj', XJ, BI)
        aJ = np.einsum('pi,pij->pj', XI, BJ)
        # build sparse A_ub: p x (1188+1), 25 nnz per row
        cols_mat = np.empty((p, 25), dtype=np.int64)
        cols_mat[:, :12] = I[:, None] * 12 + ar12
        cols_mat[:, 12:24] = J[:, None] * 12 + ar12
        cols_mat[:, 24] = NVAR
        vals_mat = np.empty((p, 25), dtype=float)
        vals_mat[:, :12] = aI
        vals_mat[:, 12:24] = aJ
        vals_mat[:, 24] = -1.0
        rows_mat = np.repeat(np.arange(p), 25)
        cols_flat = cols_mat.ravel()
        vals_flat = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals_flat, (rows_mat, cols_flat)), shape=(p, NCOL)).tocsr()
        except Exception:
            r *= 0.5
            continue
        b_ub = -c_sel
        obj = np.zeros(NCOL)
        obj[-1] = 1.0
        bounds = [(-r, r)] * NVAR + [(None, m_cur)]
        # per-LP time limit to respect internal clock
        remain = SQP_BUDGET - (time.time() - t0)
        tl = max(1.0, min(8.0, remain / 3.0))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                          options={"presolve": True, "time_limit": tl})
        except Exception:
            r *= 0.5
            continue
        if res.status != 0 or res.x is None:
            r *= 0.5
            continue
        t_pred = float(res.x[-1])
        # no predicted improvement -> tighten trust
        if not np.isfinite(t_pred) or t_pred >= m_cur - 1e-12:
            r *= 0.5
            continue
        dvec = np.asarray(res.x[:-1], dtype=float)
        if not np.all(np.isfinite(dvec)):
            r *= 0.5
            continue
        D = dvec.reshape(n, d - 1)
        # actual move along tangent then renormalise
        DX = np.einsum('nij,nj->ni', B, D)
        # backtracking along the same direction (stay inside trust)
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Y = X + alpha * DX
            Y = _norm_rows(Y)
            if not np.all(np.isfinite(Y)):
                continue
            m_new = _max_cos(Y)
            if m_new < m_cur - 1e-12:
                X = Y
                m_cur = m_new
                if m_new < best_m:
                    best_m = m_new
                    best_X = Y.copy()
                # geometric shrink of trust (homotopy to 1e-8)
                r = max(r * 0.85, r_min * 0.5)
                # if big gain keep radius a bit larger
                if m_cur < best_m + 1e-12 and alpha < 1.0:
                    pass
                accepted = True
                break
        if not accepted:
            r *= 0.5
            continue
        # early exit if time nearly up
        if time.time() - t0 > SQP_BUDGET - 2.0:
            break
    return _norm_rows(np.asarray(best_X, dtype=float))

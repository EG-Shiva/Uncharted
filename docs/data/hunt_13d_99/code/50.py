import time
import numpy as np
from scipy.optimize import linprog, minimize
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
        xj = xi[mask]
        M = -xi[:, None] * xj[None, :]
        M[mask, :] += eye
        Q, _ = np.linalg.qr(M, mode='reduced')
        B[i] = Q
    return B

def _lse_polish(X0, alpha, maxiter):
    N, d = X0.shape
    U0 = X0.ravel().astype(float).copy()
    def fun(Uf):
        U = Uf.reshape(N, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = U / nr
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5) + 1e-300
        L = m + np.log(S) / alpha
        GX = (E @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GU = (GX - dot * X) / nr
        return float(L), GU.ravel().astype(float)
    try:
        res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                       options={'maxiter': int(maxiter), 'ftol': 1e-20,
                                'gtol': 1e-9, 'maxfun': int(maxiter)*5, 'maxcor': 8})
        U = res.x.reshape(N, d)
        return _norm_rows(U)
    except Exception:
        return X0.copy()

def solve():
    t0 = time.time()
    LP_BUDGET = 105.0
    HARD = 172.0
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
    NVAR = n * (d - 1)
    NCOL = NVAR + 1
    iu0, iu1 = np.triu_indices(n, 1)
    ar12 = np.arange(d - 1)
    r = 4e-4
    r_min = 1e-8
    eps = 4e-6
    it = 0
    # homotopy loop
    while time.time() - t0 < LP_BUDGET:
        it += 1
        if r < r_min:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m_cur = float(np.max(c_all))
        # adapt eps: tie to r but also to observed spread
        # spread of top 800
        if it == 1:
            eps = 4e-6
        else:
            # geometric target tied to trust radius, floored
            target = max(0.01 * r, 5e-9)
            # keep eps non-increasing mostly, allow slow decay
            eps = min(eps, max(target, eps * 0.7))
            # if active set too small, widen next selection (handled below)
        sel = (m_cur - c_all) < eps
        cnt = int(np.sum(sel))
        if cnt < 400:
            need = min(700, c_all.size)
            order = np.argpartition(c_all, -need)[-need:]
            # argpartition not sorted, fine for widening
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order] = True
            sel = sel | sel2
            cnt = int(np.sum(sel))
        if cnt > 1200:
            idx_sel = np.where(sel)[0]
            c_tmp = c_all[idx_sel]
            keep = np.argpartition(c_tmp, -1200)[-1200:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx_sel[keep]] = True
            sel = sel2
            cnt = 1200
        if cnt == 0:
            r *= 0.5
            continue
        I = iu0[sel].astype(np.int64)
        J = iu1[sel].astype(np.int64)
        c_sel = c_all[sel]
        p = cnt
        B = _tangent_bases(X)
        XI = X[I]
        XJ = X[J]
        BI = B[I]
        BJ = B[J]
        aI = np.einsum('pi,pij->pj', XJ, BI)
        aJ = np.einsum('pi,pij->pj', XI, BJ)
        cols_mat = np.empty((p, 25), dtype=np.int64)
        cols_mat[:, :12] = I[:, None] * 12 + ar12
        cols_mat[:, 12:24] = J[:, None] * 12 + ar12
        cols_mat[:, 24] = NVAR
        vals_mat = np.empty((p, 25), dtype=float)
        vals_mat[:, :12] = aI
        vals_mat[:, 12:24] = aJ
        vals_mat[:, 24] = -1.0
        rows_mat = np.repeat(np.arange(p), 25)
        try:
            A_ub = coo_matrix((vals_mat.ravel(), (rows_mat, cols_mat.ravel())), shape=(p, NCOL)).tocsr()
        except Exception:
            r *= 0.5
            continue
        b_ub = -c_sel
        obj = np.zeros(NCOL)
        obj[-1] = 1.0
        bounds = [(-r, r)] * NVAR + [(None, m_cur)]
        remain = LP_BUDGET - (time.time() - t0)
        tl = max(1.0, min(6.0, remain / 3.0))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                          options={"presolve": True, "time_limit": tl})
        except Exception:
            r *= 0.5
            continue
        if res.status != 0 or res.x is None:
            r *= 0.5
            # shrink eps a bit to force better conditioning
            eps = max(eps * 0.8, 5e-9)
            continue
        t_pred = float(res.x[-1])
        if not np.isfinite(t_pred) or t_pred >= m_cur - 1e-12:
            r *= 0.5
            eps = max(eps * 0.85, 5e-9)
            continue
        dvec = np.asarray(res.x[:-1], dtype=float)
        if not np.all(np.isfinite(dvec)):
            r *= 0.5
            continue
        D = dvec.reshape(n, d - 1)
        DX = np.einsum('nij,nj->ni', B, D)
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Y = _norm_rows(X + alpha * DX)
            if not np.all(np.isfinite(Y)):
                continue
            m_new = _max_cos(Y)
            if m_new < m_cur - 1e-12:
                X = Y
                m_cur = m_new
                if m_new < best_m:
                    best_m = m_new
                    best_X = Y.copy()
                # slow geometric shrink, less aggressive than 0.85
                r = max(r * 0.92, r_min * 0.5)
                if alpha < 1.0:
                    r = max(r * 0.8, r_min * 0.5)
                # collapse eps toward spread: force equioscillation
                # estimate new spread of plateau
                G2 = X @ X.T
                c2 = G2[iu0, iu1]
                m2 = float(np.max(c2))
                # width of top-800
                if c2.size > 800:
                    part = np.argpartition(c2, -800)[-800:]
                    w = m2 - float(np.min(c2[part]))
                else:
                    w = m2 - float(np.min(c2))
                eps = max(min(eps * 0.9, max(10.0 * w, 0.01 * r, 5e-9)), 5e-9)
                accepted = True
                break
        if not accepted:
            r *= 0.5
            continue
        if time.time() - t0 > LP_BUDGET - 1.5:
            break
    # final LogSumExp polish to squeeze plateau width into real gain
    # use remaining time up to HARD
    cur = best_X.copy()
    cur_m = best_m
    sched = [(800.0, 120), (2000.0, 120), (5000.0, 100)]
    for alpha, mit in sched:
        if time.time() - t0 > HARD - 6.0:
            break
        Y = _lse_polish(cur, alpha, mit)
        mY = _max_cos(Y)
        if mY < cur_m:
            cur_m = mY
            cur = Y.copy()
        if mY < best_m:
            best_m = mY
            best_X = Y.copy()
        # also try from best_X
        if time.time() - t0 > HARD - 6.0:
            break
    return _norm_rows(np.asarray(best_X, dtype=float))

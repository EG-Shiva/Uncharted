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
                                'gtol': 1e-9, 'maxfun': int(maxiter) * 5, 'maxcor': 8})
        U = res.x.reshape(N, d)
        return _norm_rows(U)
    except Exception:
        return X0.copy()

def solve():
    t0 = time.time()
    HARD = 150.0
    LP_BUDGET = 125.0
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
    dm = d - 1
    NVAR = n * dm
    NCOL = NVAR + 1
    iu0, iu1 = np.triu_indices(n, 1)
    ar12 = np.arange(dm)
    r = 3e-4
    r_min = 1e-7
    EPS = 2e-6
    it = 0
    sqNV = float(np.sqrt(NVAR))
    while True:
        if time.time() - t0 > LP_BUDGET:
            break
        if r < r_min:
            break
        it += 1
        if it > 90:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m_cur = float(np.max(c_all))
        sel = (m_cur - c_all) < EPS
        cnt = int(np.sum(sel))
        if cnt < 300:
            need = min(700, c_all.size)
            order = np.argpartition(c_all, -need)[-need:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order] = True
            sel = sel | sel2
            cnt = int(np.sum(sel))
        if cnt > 1300:
            idx_sel = np.where(sel)[0]
            c_tmp = c_all[idx_sel]
            keep = np.argpartition(c_tmp, -1300)[-1300:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx_sel[keep]] = True
            sel = sel2
            cnt = 1300
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
        lp_ok = False
        dvec = None
        t_pred = np.inf
        if res.status == 0 and res.x is not None:
            t_pred = float(res.x[-1])
            if np.isfinite(t_pred) and t_pred < m_cur - 1e-12:
                dvec = np.asarray(res.x[:-1], dtype=float)
                if np.all(np.isfinite(dvec)):
                    lp_ok = True
        accepted = False
        if lp_ok:
            D = dvec.reshape(n, dm)
            DX = np.einsum('nij,nj->ni', B, D)
            for alpha in (1.0, 0.5, 0.25, 0.125):
                if time.time() - t0 > HARD:
                    break
                Y = _norm_rows(X + alpha * DX)
                m_new = _max_cos(Y)
                if m_new < m_cur - 1e-12:
                    X = Y
                    m_cur = m_new
                    if m_new < best_m:
                        best_m = m_new
                        best_X = Y.copy()
                    accepted = True
                    break
            if accepted:
                r = max(r * 0.85, r_min * 0.5)
                if time.time() - t0 < HARD - 12.0:
                    try:
                        Yp = _lse_polish(X, 3500.0, 30)
                        mp = _max_cos(Yp)
                        if mp < m_cur - 1e-12:
                            X = Yp
                            m_cur = mp
                            if mp < best_m:
                                best_m = mp
                                best_X = Yp.copy()
                    except Exception:
                        pass
                continue
        # LP failed or no gain -> SVD near-null exploration
        if time.time() - t0 > LP_BUDGET:
            break
        try:
            Jd = np.zeros((p, NVAR), dtype=float)
            rr = np.arange(p)
            for k in range(dm):
                Jd[rr, I * 12 + k] = aI[:, k]
                Jd[rr, J * 12 + k] = aJ[:, k]
            # economy SVD
            _, Ss, Vt = np.linalg.svd(Jd, full_matrices=False)
            ndir = min(5, Vt.shape[0])
            Vsel = Vt[-ndir:, :].copy()
        except Exception:
            r *= 0.5
            continue
        steps = [3.5e-3, 1e-3, 3.5e-4, 1e-4, 3e-5, 1e-5]
        best_Y = None
        best_m_try = m_cur
        for vi in range(ndir):
            if time.time() - t0 > HARD:
                break
            v = Vsel[vi]
            nv = float(np.linalg.norm(v))
            if not np.isfinite(nv) or nv < 1e-15:
                continue
            v = v / nv
            for sgn in (1.0, -1.0):
                Dv = (sgn * v).reshape(n, dm)
                DXv_base = np.einsum('nij,nj->ni', B, Dv)
                for s in steps:
                    Y = _norm_rows(X + s * DXv_base)
                    m_new = _max_cos(Y)
                    if m_new < best_m_try - 1e-12:
                        best_m_try = m_new
                        best_Y = Y.copy()
                    # also scaled-by-sqrt variant is already covered by larger steps
                # explicit small-step set required by spec (unit-norm scaling)
                # already included as 1e-4,3e-5,1e-5
        if best_Y is not None and best_m_try < m_cur - 1e-12:
            X = best_Y
            m_cur = best_m_try
            if best_m_try < best_m:
                best_m = best_m_try
                best_X = best_Y.copy()
            r = max(r * 0.85, r_min * 0.5)
            if time.time() - t0 < HARD - 12.0:
                try:
                    Yp = _lse_polish(X, 4000.0, 30)
                    mp = _max_cos(Yp)
                    if mp < m_cur - 1e-12:
                        X = Yp
                        m_cur = mp
                        if mp < best_m:
                            best_m = mp
                            best_X = Yp.copy()
                except Exception:
                    pass
            continue
        else:
            r *= 0.5
            continue
    # final short high-alpha polish if time permits
    if time.time() - t0 < 165.0:
        try:
            remain = 172.0 - (time.time() - t0)
            if remain > 8.0:
                Yp = _lse_polish(best_X, 5000.0, 40)
                mp = _max_cos(Yp)
                if mp < best_m:
                    best_m = mp
                    best_X = Yp.copy()
        except Exception:
            pass
    return _norm_rows(np.asarray(best_X, dtype=float))

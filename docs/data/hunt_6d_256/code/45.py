import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_ties(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu[0], iu[1]]
    m = float(c.max())
    ties = int(np.count_nonzero(c >= m - 1e-8))
    return m, ties, c, iu

def _tangent_bases(x):
    n, d = x.shape
    m5 = d - 1
    U = np.empty((n, d, m5))
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        cols = []
        for j in range(d):
            if j == k:
                continue
            ej = np.zeros(d)
            ej[j] = 1.0
            v = ej - xi[j] * xi
            cols.append(v)
        M = np.stack(cols, axis=1)  # 6x5
        Q, _ = np.linalg.qr(M)
        U[i] = Q
    return U

def solve():
    t0 = time.time()
    DEADLINE = t0 + 178.0
    RESERVE = 20.0
    STOP_EQ = DEADLINE - RESERVE
    n, d = 256, 6
    m5 = 5
    NV = n * m5

    base = None
    base_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    G = a @ a.T
                    np.fill_diagonal(G, -2.0)
                    vv = float(G.max())
                    if vv < base_v:
                        base_v = vv
                        base = a.copy()
        except Exception:
            pass
    if base is None:
        rs = np.random.default_rng(0)
        base = rs.normal(size=(n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_v = float((lambda G: (np.fill_diagonal(G, -2.0), float(G.max()))[1])(base @ base.T))

    cur_x = base.copy()
    cur_x /= np.linalg.norm(cur_x, axis=1, keepdims=True)
    cur_val, cur_ties, _, _ = _max_cos_ties(cur_x)
    best_x = cur_x.copy()
    best_val = cur_val
    best_ties = cur_ties

    radii = [3e-4, 1e-4, 3e-5, 1e-5]
    it = 0
    ar5 = np.arange(m5)

    while time.time() < STOP_EQ:
        r = radii[it % len(radii)]
        it += 1
        # active set: slack below 3e-4
        try:
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            iuI, iuJ = np.triu_indices(n, 1)
            c_all = G[iuI, iuJ]
            mcur = float(c_all.max())
        except Exception:
            break
        thr = mcur - 3e-4
        sel = c_all > thr
        I = iuI[sel]
        J = iuJ[sel]
        c_act = c_all[sel]
        p = len(I)
        if p < 2 or p > 8000:
            # if too many, keep tightest 2500
            if p > 8000:
                part = np.argpartition(-c_act, 8000)[:8000]
                I = I[part]; J = J[part]; c_act = c_act[part]
                p = 8000
            if p < 2:
                continue
        # tangent bases at cur
        try:
            U = _tangent_bases(cur_x)
        except Exception:
            continue
        # vectorized coefficients
        try:
            U_I = U[I]  # p,6,5
            U_J = U[J]
            x_J = cur_x[J]  # p,6
            x_I = cur_x[I]
            a_I = np.einsum('pgk,pg->pk', U_I, x_J)  # p,5
            a_J = np.einsum('pgk,pg->pk', U_J, x_I)
        except Exception:
            continue
        # Build LP1: min t s.t. a^T y - t <= -c
        try:
            rows = np.repeat(np.arange(p), 2 * m5 + 1)
            cols_I = (I[:, None] * m5 + ar5).ravel()
            cols_J = (J[:, None] * m5 + ar5).ravel()
            cols = np.empty(p * (2 * m5 + 1), dtype=int)
            cols[0::(2 * m5 + 1)] = cols_I[0::m5]  # placeholder, rebuild properly below
            # build properly with reshape
            cols_mat = np.empty((p, 2 * m5 + 1), dtype=int)
            cols_mat[:, :m5] = I[:, None] * m5 + ar5
            cols_mat[:, m5:2 * m5] = J[:, None] * m5 + ar5
            cols_mat[:, 2 * m5] = NV
            cols = cols_mat.ravel()
            vals_mat = np.empty((p, 2 * m5 + 1))
            vals_mat[:, :m5] = a_I
            vals_mat[:, m5:2 * m5] = a_J
            vals_mat[:, 2 * m5] = -1.0
            vals = vals_mat.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, NV + 1)).tocsr()
            b_ub = -c_act
            obj = np.zeros(NV + 1)
            obj[-1] = 1.0
            bounds = [(-r, r)] * NV + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception:
            continue
        if res.status != 0:
            continue
        try:
            y1 = np.asarray(res.x[:-1], dtype=float).ravel()
            t_star = float(res.x[-1])
        except Exception:
            continue
        if not np.all(np.isfinite(y1)) or not np.isfinite(t_star):
            continue
        # second lexicographic LP: fix t near-optimal, minimize sum of top1500 linearized cos
        t_fix = t_star + 2e-7
        # top 1500 by current cosine
        try:
            T2 = min(1500, c_all.size)
            idx_top = np.argpartition(-c_all, T2 - 1)[:T2]
            It = iuI[idx_top]
            Jt = iuJ[idx_top]
            # coefficients for top set (recompute using U)
            U_It = U[It]
            U_Jt = U[Jt]
            x_Jt = cur_x[Jt]
            x_It = cur_x[It]
            a_It = np.einsum('pgk,pg->pk', U_It, x_Jt)
            a_Jt = np.einsum('pgk,pg->pk', U_Jt, x_It)
            s = np.zeros(NV)
            # accumulate
            # use add.at for speed
            np.add.at(s, (It[:, None] * m5 + ar5).ravel(), a_It.ravel())
            np.add.at(s, (Jt[:, None] * m5 + ar5).ravel(), a_Jt.ravel())
        except Exception:
            continue
        # LP2: min s^T y s.t. A2 y <= t_fix - c_act
        y2 = None
        try:
            b2 = t_fix - c_act  # p vector, small
            # A2 same as A_ub without last column
            rows2 = np.repeat(np.arange(p), 2 * m5)
            cols_mat2 = np.empty((p, 2 * m5), dtype=int)
            cols_mat2[:, :m5] = I[:, None] * m5 + ar5
            cols_mat2[:, m5:] = J[:, None] * m5 + ar5
            cols2 = cols_mat2.ravel()
            vals_mat2 = np.empty((p, 2 * m5))
            vals_mat2[:, :m5] = a_I
            vals_mat2[:, m5:] = a_J
            vals2 = vals_mat2.ravel()
            A2 = coo_matrix((vals2, (rows2, cols2)), shape=(p, NV)).tocsr()
            bounds2 = [(-r, r)] * NV
            res2 = linprog(s, A_ub=A2, b_ub=b2, bounds=bounds2, method="highs")
            if res2.status == 0:
                y2 = np.asarray(res2.x, dtype=float).ravel()
        except Exception:
            y2 = None
        # candidates to try: y2 first, then y1
        cands = []
        if y2 is not None and np.all(np.isfinite(y2)):
            cands.append(y2)
        cands.append(y1)
        accepted = False
        for yc in cands:
            if time.time() > STOP_EQ:
                break
            try:
                Ym = yc.reshape(n, m5)
                dx = np.einsum('ngk,nk->ng', U, Ym)
                xn = cur_x + dx
                nr = np.linalg.norm(xn, axis=1, keepdims=True)
                if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
                    continue
                xn /= nr
                Gn = xn @ xn.T
                np.fill_diagonal(Gn, -2.0)
                cn = Gn[iuI, iuJ]
                mnew = float(cn.max())
                ties_new = int(np.count_nonzero(cn >= mnew - 1e-8))
                # track global best
                improved_best = False
                if mnew < best_val - 1e-12:
                    best_x = xn.copy()
                    best_val = mnew
                    best_ties = ties_new
                    improved_best = True
                elif mnew <= best_val + 1e-12 and ties_new > best_ties:
                    best_x = xn.copy()
                    best_ties = ties_new
                    improved_best = True
                # acceptance for cur move
                if mnew < cur_val - 1e-12:
                    cur_x = xn.copy()
                    cur_val = mnew
                    cur_ties = ties_new
                    accepted = True
                    break
                elif mnew <= cur_val + 1e-12 and ties_new > cur_ties:
                    cur_x = xn.copy()
                    cur_ties = ties_new
                    # cur_val stays (equal within tol) but update to mnew
                    cur_val = mnew
                    accepted = True
                    break
                elif improved_best and mnew <= cur_val + 5e-7:
                    # allow sideways drift if it improved global best slightly above cur due to numerics
                    pass
            except Exception:
                continue
        # if neither accepted, continue cycling (topology change via different radius)
        continue

    # no-regression fallback + final normalize
    try:
        b = np.load("best.npy")
        b = np.asarray(b, dtype=float)
        if b.shape == (n, d):
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            Gb = b @ b.T
            np.fill_diagonal(Gb, -2.0)
            bv = float(Gb.max())
            if bv < best_val:
                best_x = b.copy()
                best_val = bv
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

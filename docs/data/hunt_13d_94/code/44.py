import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import null_space

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _tangent_bases(x):
    n, d = x.shape
    Bs = np.empty((n, d, d - 1))
    for i in range(n):
        Ns = null_space(x[i:i+1, :])
        # null_space returns (d, d-1) orthonormal
        if Ns.shape != (d, d - 1):
            # fallback via QR
            q, _ = np.linalg.qr(np.random.randn(d, d))
            # orthogonalize against x[i]
            q = q - np.outer(x[i], x[i] @ q)
            q, _ = np.linalg.qr(q[:, :d-1])
            Ns = q
        Bs[i] = Ns
    return Bs

def solve():
    T0 = time.time()
    POLISH_BUDGET = 60.0
    POLISH_END = T0 + POLISH_BUDGET
    TOTAL_END = T0 + 168.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xt = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xt):
            xb = xr.copy()
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    dm = d - 1
    Nvar = n * dm
    x = xb.copy()
    best = _max_cos(x)
    init_best = best
    iu0 = np.triu_indices(n, 1)

    radius = 1e-3
    fail_count = 0

    # preallocate objective
    obj = np.zeros(Nvar + 1)
    obj[-1] = 1.0

    it = 0
    while time.time() < POLISH_END and time.time() < TOTAL_END - 5.0:
        it += 1
        if radius < 1e-9:
            break
        G = x @ x.T
        # max over off-diagonal
        iu = iu0
        c_all = G[iu]
        m = float(c_all.max())
        # active set: slack below 1e-6, plus retain slack < 10*trial step
        retain = max(1e-6, 10.0 * radius)
        sel = c_all > m - retain
        # cap size for speed: keep tightest 4000
        if int(sel.sum()) > 4000:
            idx = np.argpartition(c_all, -4000)[-4000:]
            sel2 = np.zeros_like(sel)
            sel2[idx] = True
            sel = sel2
        I = iu[0][sel].astype(int)
        J = iu[1][sel].astype(int)
        c_act = c_all[sel]
        p = len(I)
        if p == 0:
            break
        # tangent bases at current x
        try:
            Bs = _tangent_bases(x)
        except Exception:
            break
        # Build linearized rows: A*u - t <= -c
        # A row k: N_i^T x_j at block i, N_j^T x_i at block j
        rows = []
        cols = []
        vals = []
        # dense At for rank test if p moderate (store for null move)
        At_dense = None
        need_dense = (p <= 1500)
        if need_dense:
            At_dense = np.zeros((p, Nvar))
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            ai = Bs[i].T @ x[j]  # (dm,)
            aj = Bs[j].T @ x[i]
            # sparse entries
            base_i = i * dm
            base_j = j * dm
            for t_ in range(dm):
                rows.append(k); cols.append(base_i + t_); vals.append(float(ai[t_]))
                if need_dense:
                    At_dense[k, base_i + t_] = float(ai[t_])
            for t_ in range(dm):
                rows.append(k); cols.append(base_j + t_); vals.append(float(aj[t_]))
                if need_dense:
                    At_dense[k, base_j + t_] = float(aj[t_])
            rows.append(k); cols.append(Nvar); vals.append(-1.0)
        try:
            A_ub = coo_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(p, Nvar + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        b_ub = -c_act
        bounds = [(-radius, radius)] * Nvar + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            # rank-deficiency test -> null move attempt before shrinking
            fail_count += 1
            if fail_count >= 2 and time.time() < POLISH_END - 2:
                # null-space equalization attempt
                try:
                    # tightest subset
                    order = np.argsort(-c_all)
                    pt = min(400, p)
                    tight_idx = order[:pt]
                    It = iu[0][tight_idx].astype(int)
                    Jt = iu[1][tight_idx].astype(int)
                    Att = np.zeros((pt, Nvar))
                    for k2 in range(pt):
                        ii = int(It[k2]); jj = int(Jt[k2])
                        ai2 = Bs[ii].T @ x[jj]
                        aj2 = Bs[jj].T @ x[ii]
                        Att[k2, ii*dm:(ii+1)*dm] = ai2
                        Att[k2, jj*dm:(jj+1)*dm] = aj2
                    # rank test
                    try:
                        sv = np.linalg.svd(Att, compute_uv=False)
                        rnk = int((sv > 1e-10).sum())
                        rank_def = rnk < min(Att.shape)
                    except Exception:
                        rank_def = True
                    if rank_def or True:
                        rvec = rng.standard_normal(Nvar)
                        # project onto null(Att): w = r - At^T z, (At At^T)z = At r
                        Atr = Att @ rvec
                        M = Att @ Att.T + 1e-12 * np.eye(pt)
                        try:
                            z = np.linalg.solve(M, Atr)
                        except Exception:
                            z, *_ = np.linalg.lstsq(M, Atr, rcond=None)
                        vnull = rvec - Att.T @ z
                        # scale to trust radius
                        Vb = vnull.reshape(n, dm)
                        # displacement norm per point in tangent coords ~ euclidean
                        norms = np.linalg.norm(Vb, axis=1).max()
                        if norms > 0 and np.isfinite(norms):
                            vnull = vnull / norms * radius
                            improved_null = False
                            for sgn in (1.0, -1.0):
                                for alpha in (1.0, 0.5, 0.25):
                                    uu = sgn * alpha * vnull
                                    Dd = np.empty_like(x)
                                    for ii2 in range(n):
                                        Dd[ii2] = Bs[ii2] @ uu[ii2*dm:(ii2+1)*dm]
                                    y = x + Dd
                                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                                    v = _max_cos(y)
                                    if v < best - 1e-15:
                                        x, best = y, v
                                        improved_null = True
                                        break
                                if improved_null:
                                    break
                            if improved_null:
                                fail_count = 0
                                radius = min(radius * 1.2, 5e-3)
                                continue
                        fail_count = 0
                    except Exception:
                        pass
            radius *= 0.5
            continue
        u = res.x[:-1]
        # t_pred = res.x[-1]
        # build displacement
        D = np.empty_like(x)
        for ii in range(n):
            D[ii] = Bs[ii] @ u[ii*dm:(ii+1)*dm]
        # exact renormalization + second-order correction (renorm is exact 2nd order)
        # backtracking line search on true maximum
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            y = x + alpha * D
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            y = y / nr
            v = _max_cos(y)
            if v < best - 1e-15 and np.all(np.isfinite(y)):
                x, best = y, v
                improved = True
                break
        if improved:
            fail_count = 0
            radius = min(radius * 1.5, 5e-3)
        else:
            fail_count += 1
            # explicit rank-deficiency check + null-space equalization move
            if fail_count >= 2:
                try:
                    if At_dense is not None and At_dense.shape[0] >= 10:
                        try:
                            sv2 = np.linalg.svd(At_dense, compute_uv=False)
                            rnk2 = int((sv2 > 1e-9).sum())
                        except Exception:
                            rnk2 = -1
                        # attempt null move using tightest 300 of this active set
                        order2 = np.argsort(c_act)  # smallest first (loosest)
                        # tightest = largest c
                        pt2 = min(300, p)
                        tight_rows = np.argsort(-c_act)[:pt2]
                        Att2 = At_dense[tight_rows, :]
                        rvec2 = rng.standard_normal(Nvar)
                        Atr2 = Att2 @ rvec2
                        M2 = Att2 @ Att2.T + 1e-12 * np.eye(pt2)
                        try:
                            z2 = np.linalg.solve(M2, Atr2)
                        except Exception:
                            z2, *_ = np.linalg.lstsq(M2, Atr2, rcond=None)
                        vnull2 = rvec2 - Att2.T @ z2
                        norms2 = np.linalg.norm(vnull2.reshape(n, dm), axis=1).max()
                        if norms2 > 0 and np.isfinite(norms2):
                            vnull2 = vnull2 / norms2 * radius
                            got = False
                            for sgn in (1.0, -1.0):
                                for alpha in (1.0, 0.5, 0.25, 0.125):
                                    uu2 = sgn * alpha * vnull2
                                    Dd2 = np.empty_like(x)
                                    for ii3 in range(n):
                                        Dd2[ii3] = Bs[ii3] @ uu2[ii3*dm:(ii3+1)*dm]
                                    y2 = x + Dd2
                                    y2 /= np.linalg.norm(y2, axis=1, keepdims=True)
                                    vv2 = _max_cos(y2)
                                    if vv2 < best - 1e-15:
                                        x, best = y2, vv2
                                        got = True
                                        break
                                if got:
                                    break
                            if got:
                                fail_count = 0
                                radius = min(radius * 1.2, 5e-3)
                                continue
                except Exception:
                    pass
                fail_count = 0
            radius *= 0.5
            # follow schedule roughly: snap toward 1e-3,3e-4,1e-4...
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    # keep only if beats best.npy
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        v0 = _max_cos(x0)
        if best > v0 - 1e-15:
            # also compare to init_best; if not better, return incumbent
            if best > init_best - 1e-15:
                return x0
            # if we tied/beats incumbent but not file (race), still return x
            if v0 <= init_best:
                return x0
    except Exception:
        pass
    # also never return worse than input
    if best > init_best + 1e-12:
        try:
            return np.load("best.npy").astype(float)
        except Exception:
            pass
    return x

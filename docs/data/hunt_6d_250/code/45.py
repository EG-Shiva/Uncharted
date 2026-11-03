import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_Q(X):
    # Householder basis: Q_i (6x5) orthonormal, Q_i^T x_i = 0, vectorized
    n, d = X.shape
    assert d == 6
    s = np.sign(X[:, 0])
    s[s == 0] = 1.0
    V = X.copy()
    V[:, 0] += s
    nr = np.linalg.norm(V, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-300)
    V = V / nr
    vc = V[:, 1:]  # n,5
    # Q[n,j,k] = (1 if j==k+1 else 0) - 2*V[n,j]*vc[n,k]
    Q = np.empty((n, 6, 5), dtype=np.float64)
    # E part
    Q[:] = 0.0
    for k in range(5):
        Q[:, k + 1, k] = 1.0
    Q -= 2.0 * V[:, :, None] * vc[:, None, :]
    return Q

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    n, d = 250, 6
    k5 = 5
    # load best
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
    Xcur = best.copy()
    Mcur = float(best_val)
    Xbest = best.copy()
    Mbest = float(best_val)

    # initial active set: all pairs within 1e-5 of max
    try:
        G0 = Xcur @ Xcur.T
        iu0 = np.triu_indices(n, 1)
        gv0 = G0[iu0]
        M0 = float(np.max(gv0))
        thr0 = M0 - 1e-5
        m0 = gv0 >= thr0
        I0 = iu0[0][m0].astype(np.int64)
        J0 = iu0[1][m0].astype(np.int64)
        if I0.size == 0:
            idx = np.argsort(-gv0)[:2000]
            I0 = iu0[0][idx].astype(np.int64)
            J0 = iu0[1][idx].astype(np.int64)
    except Exception:
        I0 = np.array([0], dtype=np.int64)
        J0 = np.array([1], dtype=np.int64)

    active = {}
    for a, b in zip(I0.tolist(), J0.tolist()):
        if a > b:
            a, b = b, a
        active[(int(a), int(b))] = 0

    rho = 1e-3
    rho_min = 1e-5
    rho_max = 1e-2
    Nvar = n * k5
    consec_fail = 0

    it = 0
    iu_all0, iu_all1 = np.triu_indices(n, 1)
    while remaining() > 8.0 and it < 250:
        it += 1
        if len(active) == 0:
            # rebuild from current max
            G = Xcur @ Xcur.T
            gv = G[iu_all0, iu_all1]
            Mcur = float(np.max(gv))
            sel = gv >= Mcur - 1e-5
            I0 = iu_all0[sel]; J0 = iu_all1[sel]
            for a, b in zip(I0.tolist(), J0.tolist()):
                active[(int(a), int(b))] = 0
            if len(active) == 0:
                break
        keys = sorted(active.keys())
        p = len(keys)
        I = np.array([k[0] for k in keys], dtype=np.int64)
        J = np.array([k[1] for k in keys], dtype=np.int64)

        # tangent bases at current X
        try:
            Q = _build_Q(Xcur)
        except Exception:
            break
        # coefficients
        XI = Xcur[I]  # p,6
        XJ = Xcur[J]
        cvec = np.sum(XI * XJ, axis=1)  # p
        QI = Q[I]  # p,6,5
        QJ = Q[J]
        try:
            AI = np.einsum('pkm,pk->pm', QI, XJ)
            AJ = np.einsum('pkm,pk->pm', QJ, XI)
        except Exception:
            break

        # build sparse LP: min t s.t. c + AI*UI + AJ*UJ - t <= 0? -> A u -t <= -c
        # variables [u (Nvar), t]
        try:
            cols_I = I[:, None] * k5 + np.arange(k5)[None, :]
            cols_J = J[:, None] * k5 + np.arange(k5)[None, :]
            cols_mid = np.concatenate([cols_I, cols_J, np.full((p, 1), Nvar, dtype=np.int64)], axis=1)
            rows_mid = np.repeat(np.arange(p, dtype=np.int64), 11)
            cols_flat = cols_mid.ravel()
            data_mid = np.concatenate([AI, AJ, -np.ones((p, 1), dtype=np.float64)], axis=1).ravel()
            A_ub = coo_matrix((data_mid, (rows_mid, cols_flat)), shape=(p, Nvar + 1)).tocsr()
            b_ub = -cvec
            c_obj = np.zeros(Nvar + 1, dtype=np.float64)
            c_obj[-1] = 1.0
            bounds = [(-float(rho), float(rho))] * Nvar + [(None, float(Mcur))]
        except Exception:
            rho = max(rho_min, rho * 0.5)
            continue

        if remaining() < 5.0:
            break
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            rho = max(rho_min, rho * 0.5)
            consec_fail += 1
            if consec_fail > 5:
                break
            continue
        if res.status != 0 or res.x is None or (not np.all(np.isfinite(res.x))):
            rho = max(rho_min, rho * 0.5)
            consec_fail += 1
            if consec_fail > 8:
                # try smaller rho once
                if rho > rho_min:
                    continue
                break
            continue

        xopt = res.x
        t_pred = float(xopt[-1])
        Umat = xopt[:-1].reshape(n, k5)
        # linearized values for dual/stress estimate
        UI = Umat[I]
        UJ = Umat[J]
        lin_vals = cvec + np.sum(AI * UI, axis=1) + np.sum(AJ * UJ, axis=1)
        pred = Mcur - t_pred
        # update zero-counts (dual zero ~ loose constraint)
        tol_tight = 1e-9
        tight = lin_vals >= t_pred - tol_tight
        for idx_k, kk in enumerate(keys):
            if tight[idx_k]:
                active[kk] = 0
            else:
                active[kk] = active.get(kk, 0) + 1
        # drop those zero for 3 consecutive iterations
        todel = [kk for kk, vv in active.items() if vv >= 3]
        for kk in todel:
            del active[kk]

        if pred < 1e-12:
            # no predicted progress
            if rho <= rho_min * 1.01:
                # certified local optimum
                break
            else:
                rho = rho_min
                consec_fail += 1
                continue

        # retracted line search over alpha in {0.25,0.5,1,2}
        try:
            Dmat = np.einsum('nkm,nm->nk', Q, Umat)
        except Exception:
            rho = max(rho_min, rho * 0.5)
            continue
        # guard huge direction (should be bounded by rho)
        if not np.all(np.isfinite(Dmat)):
            rho = max(rho_min, rho * 0.5)
            continue
        best_alpha_M = Mcur
        best_alpha_X = None
        for alpha in (1.0, 0.5, 0.25, 2.0):
            Y = Xcur + float(alpha) * Dmat
            nrY = np.linalg.norm(Y, axis=1, keepdims=True)
            nrY = np.maximum(nrY, 1e-12)
            Y = Y / nrY
            if not np.all(np.isfinite(Y)):
                continue
            v = _max_cos(Y)
            if v < best_alpha_M - 1e-15:
                best_alpha_M = v
                best_alpha_X = Y
        if best_alpha_X is not None:
            ared = Mcur - best_alpha_M
            # ratio to predicted
            ratio = ared / pred if pred > 1e-14 else 1.0
            # adapt rho
            if ratio > 0.7:
                rho = min(rho_max, rho * 1.5)
            elif ratio < 0.2:
                rho = max(rho_min, rho * 0.5)
            # accept
            Xcur = best_alpha_X
            Mcur = float(best_alpha_M)
            consec_fail = 0
            if Mcur < Mbest - 1e-15:
                Mbest = float(Mcur)
                Xbest = Xcur.copy()
            # add every pair within 1e-6 of new max
            try:
                G = Xcur @ Xcur.T
                gv = G[iu_all0, iu_all1]
                Mcur_check = float(np.max(gv))
                Mcur = Mcur_check
                sel = gv >= Mcur - 1e-6
                Ia = iu_all0[sel]; Ja = iu_all1[sel]
                for a, b in zip(Ia.tolist(), Ja.tolist()):
                    kk = (int(a), int(b)) if a < b else (int(b), int(a))
                    if kk not in active:
                        active[kk] = 0
            except Exception:
                pass
            # safety cap: if active huge, keep tightest 6000 by true cosine
            if len(active) > 6000:
                try:
                    kkall = sorted(active.keys())
                    Ia2 = np.array([k[0] for k in kkall], dtype=np.int64)
                    Ja2 = np.array([k[1] for k in kkall], dtype=np.int64)
                    vv = np.sum(Xcur[Ia2] * Xcur[Ja2], axis=1)
                    odr = np.argsort(-vv)[:6000]
                    keep = set([kkall[o] for o in odr.tolist()])
                    for kk in list(active.keys()):
                        if kk not in keep:
                            del active[kk]
                except Exception:
                    pass
        else:
            # no improvement: shrink trust region
            rho = max(rho_min, rho * 0.5)
            consec_fail += 1
            if consec_fail > 12 and rho <= rho_min * 1.01:
                # check certification: if pred tiny we already handled; else dead
                if pred < 1e-9:
                    break
                # keep trying a bit but avoid infinite loop
                if consec_fail > 20:
                    break
            continue

    out = Xbest.copy() if Mbest <= best_val + 1e-15 else best.copy()
    # normalize strictly
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    out = out / nr
    # tiny shrink not needed; ensure finite
    out[~np.isfinite(out)] = 0.0
    # renormalize after fix
    nr2 = np.linalg.norm(out, axis=1, keepdims=True)
    nr2[nr2 == 0] = 1.0
    out = out / nr2
    # if degenerate row, set to e1
    for i in range(n):
        if not np.all(np.isfinite(out[i])) or float(np.linalg.norm(out[i])) < 0.5:
            out[i, :] = 0.0
            out[i, 0] = 1.0
    nr3 = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(nr3, 1e-12)
    return out

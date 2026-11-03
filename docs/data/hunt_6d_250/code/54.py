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
    s = np.sign(X[:, 0])
    s[s == 0] = 1.0
    V = X.copy()
    V[:, 0] += s
    nr = np.linalg.norm(V, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-300)
    V = V / nr
    vc = V[:, 1:]
    Q = np.empty((n, 6, 5), dtype=np.float64)
    Q[:] = 0.0
    for k in range(5):
        Q[:, k + 1, k] = 1.0
    Q -= 2.0 * V[:, :, None] * vc[:, None, :]
    return Q

def _extract_duals(res, p):
    try:
        il = getattr(res, "ineqlin", None)
        if il is not None:
            if isinstance(il, dict):
                m = il.get("marginals", None)
            else:
                try:
                    m = il.marginals
                except Exception:
                    try:
                        m = il["marginals"]
                    except Exception:
                        m = None
            if m is not None:
                m = np.asarray(m, dtype=float).ravel()
                if m.size == p:
                    return m
    except Exception:
        pass
    return None

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    n, d = 250, 6
    k5 = 5
    Nvar = n * k5
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
        best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
        best_val = _max_cos(best)
    Xcur = best.copy()
    Mcur = float(best_val)
    Xbest = best.copy()
    Mbest = float(best_val)

    iu0, iu1 = np.triu_indices(n, 1)
    # initial active set: all pairs within 1e-5 of max
    try:
        G0 = Xcur @ Xcur.T
        gv0 = G0[iu0, iu1]
        M0 = float(np.max(gv0))
        thr0 = M0 - 1e-5
        sel0 = gv0 >= thr0
        I0 = iu0[sel0].astype(np.int64)
        J0 = iu1[sel0].astype(np.int64)
        if I0.size == 0:
            idx = np.argsort(-gv0)[:2000]
            I0 = iu0[idx].astype(np.int64)
            J0 = iu1[idx].astype(np.int64)
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
    consec_fail = 0
    it = 0
    alphas = (1.0, 0.5, 0.25, 2.0, 0.125, 0.0625)

    while remaining() > 8.0 and it < 300:
        it += 1
        if len(active) == 0:
            G = Xcur @ Xcur.T
            gv = G[iu0, iu1]
            Mcur = float(np.max(gv))
            sel = gv >= Mcur - 1e-5
            Ia = iu0[sel]; Ja = iu1[sel]
            for a, b in zip(Ia.tolist(), Ja.tolist()):
                kk = (int(a), int(b)) if a < b else (int(b), int(a))
                active[kk] = 0
            if len(active) == 0:
                break
        keys = sorted(active.keys())
        p = len(keys)
        I = np.array([k[0] for k in keys], dtype=np.int64)
        J = np.array([k[1] for k in keys], dtype=np.int64)
        try:
            Q = _build_Q(Xcur)
        except Exception:
            break
        XI = Xcur[I]
        XJ = Xcur[J]
        cvec = np.sum(XI * XJ, axis=1)
        QI = Q[I]
        QJ = Q[J]
        try:
            AI = np.einsum('pkm,pk->pm', QI, XJ)
            AJ = np.einsum('pkm,pk->pm', QJ, XI)
        except Exception:
            rho = max(rho_min, rho * 0.5)
            continue
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
            if consec_fail > 8:
                break
            continue
        if res.status != 0 or res.x is None or (not np.all(np.isfinite(res.x))):
            rho = max(rho_min, rho * 0.5)
            consec_fail += 1
            if consec_fail > 10:
                if rho <= rho_min * 1.01:
                    break
                else:
                    continue
            continue
        xopt = res.x
        t_pred = float(xopt[-1])
        Umat = xopt[:-1].reshape(n, k5)
        pred = Mcur - t_pred
        # dual-based zero detection
        duals = _extract_duals(res, p)
        if duals is not None:
            is_active = np.abs(duals) > 1e-9
        else:
            try:
                UI_t = Umat[I]
                UJ_t = Umat[J]
                lin_vals = cvec + np.sum(AI * UI_t, axis=1) + np.sum(AJ * UJ_t, axis=1)
                is_active = lin_vals >= t_pred - 1e-11
            except Exception:
                is_active = np.ones(p, dtype=bool)
        for idx_k, kk in enumerate(keys):
            if bool(is_active[idx_k]):
                active[kk] = 0
            else:
                active[kk] = active.get(kk, 0) + 1
        todel = [kk for kk, vv in active.items() if vv >= 3]
        for kk in todel:
            del active[kk]

        if pred < 1e-12:
            if rho <= rho_min * 1.01:
                break
            else:
                rho = rho_min
                consec_fail += 1
                if consec_fail > 15:
                    break
                continue
        try:
            Dmat = np.einsum('nkm,nm->nk', Q, Umat)
        except Exception:
            rho = max(rho_min, rho * 0.5)
            continue
        if not np.all(np.isfinite(Dmat)):
            rho = max(rho_min, rho * 0.5)
            continue
        best_alpha_M = Mcur
        best_alpha_X = None
        for alpha in alphas:
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
                # keep scanning smaller alphas? take smallest best
        if best_alpha_X is not None:
            ared = Mcur - best_alpha_M
            ratio = ared / pred if pred > 1e-14 else 1.0
            if ratio > 0.7:
                rho = min(rho_max, rho * 1.5)
            elif ratio < 0.2:
                rho = max(rho_min, rho * 0.5)
            Xcur = best_alpha_X
            Mcur = float(best_alpha_M)
            consec_fail = 0
            if Mcur < Mbest - 1e-15:
                Mbest = float(Mcur)
                Xbest = Xcur.copy()
            try:
                G = Xcur @ Xcur.T
                gv = G[iu0, iu1]
                Mc = float(np.max(gv))
                Mcur = Mc
                sel = gv >= Mcur - 1e-6
                Ia = iu0[sel]; Ja = iu1[sel]
                for a, b in zip(Ia.tolist(), Ja.tolist()):
                    kk = (int(a), int(b)) if a < b else (int(b), int(a))
                    if kk not in active:
                        active[kk] = 0
            except Exception:
                pass
            if len(active) > 8000:
                try:
                    kkall = sorted(active.keys())
                    Ia2 = np.array([k[0] for k in kkall], dtype=np.int64)
                    Ja2 = np.array([k[1] for k in kkall], dtype=np.int64)
                    vv = np.sum(Xcur[Ia2] * Xcur[Ja2], axis=1)
                    odr = np.argsort(-vv)[:7000]
                    keep = set([kkall[o] for o in odr.tolist()])
                    for kk in list(active.keys()):
                        if kk not in keep:
                            del active[kk]
                except Exception:
                    pass
        else:
            rho = max(rho_min, rho * 0.5)
            consec_fail += 1
            if consec_fail > 20:
                if pred < 1e-9 and rho <= rho_min * 1.01:
                    break
                if consec_fail > 30:
                    break
            continue

    out = Xbest.copy() if Mbest <= best_val + 1e-15 else best.copy()
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    out = out / nr
    out[~np.isfinite(out)] = 0.0
    nr2 = np.linalg.norm(out, axis=1, keepdims=True)
    nr2[nr2 == 0] = 1.0
    out = out / nr2
    for i in range(n):
        if not np.all(np.isfinite(out[i])) or float(np.linalg.norm(out[i])) < 0.5:
            out[i, :] = 0.0
            out[i, 0] = 1.0
    nr3 = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(nr3, 1e-12)
    return out

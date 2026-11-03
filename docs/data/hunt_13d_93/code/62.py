import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 150.0
    n, d = 93, 13
    dm = d - 1
    Nvar = n * dm

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    iu0_all, iu1_all = np.triu_indices(n, 1)
    P = len(iu0_all)

    base = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base = a.copy()
        except Exception:
            pass
    if base is None:
        rng0 = np.random.default_rng(0)
        base = _norm(rng0.standard_normal((n, d)))
        base_val = _maxcos(base)

    X = base.copy()
    cur = float(base_val)
    bestX = X.copy()
    bestV = cur

    # Householder tangent bases: Barr (n,d,dm)
    D0 = np.zeros((d, dm), dtype=np.float64)
    D0[1:, :] = np.eye(dm)

    def build_bases(Xm):
        Ed = np.empty_like(Xm)
        Ed[:, 0] = 1.0 - Xm[:, 0]
        Ed[:, 1:] = -Xm[:, 1:]
        n2 = np.einsum('ij,ij->i', Ed, Ed)
        Barr = np.empty((n, d, dm), dtype=np.float64)
        Barr[:] = D0[None, :, :]
        mask = n2 > 1e-14
        if np.any(mask):
            nr = np.sqrt(n2[mask])
            U = Ed[mask] / nr[:, None]  # (m,d)
            U1 = U[:, 1:]  # (m,dm)
            # Barr = D0 - 2 u outer u1
            Barr[mask] = D0[None, :, :] - 2.0 * U[:, :, None] * U1[:, None, :]
        return Barr

    try:
        from scipy.optimize import linprog, nnls
        from scipy.sparse import coo_matrix
        has_opt = True
    except Exception:
        has_opt = False

    if not has_opt:
        return _norm(bestX)

    ar = np.arange(dm)
    neg_count = np.zeros(P, dtype=np.int32)

    # map from (i,j) to linear index k in c_all ordering: since iu0_all,iu1_all
    # order is lexicographic, we can build a lookup matrix idx_of_pair
    idx_of = np.full((n, n), -1, dtype=np.int64)
    idx_of[iu0_all, iu1_all] = np.arange(P)
    idx_of[iu1_all, iu0_all] = np.arange(P)

    rho = 1e-3
    it = 0
    c_obj = np.zeros(Nvar + 1)
    c_obj[-1] = 1.0

    while True:
        it += 1
        if time.time() - t0 > BUDGET:
            break
        if rho < 1e-9:
            break
        if it > 500:
            break
        G = X @ X.T
        c_all = G[iu0_all, iu1_all]
        cmax = float(np.max(c_all))
        slack = cmax - c_all
        # select 800-1500 pairs or slack<2e-6
        cnt2e6 = int(np.sum(slack < 2e-6))
        if cnt2e6 < 800:
            K = 800
            sel_idx = np.argpartition(-c_all, K - 1)[:K]
        elif cnt2e6 > 1500:
            K = 1500
            sel_idx = np.argpartition(-c_all, K - 1)[:K]
        else:
            sel_idx = np.where(slack < 2e-6)[0]
        # drop persistently zero-multiplier constraints (not very tight ones)
        if np.any(neg_count[sel_idx] >= 5):
            keep = np.ones(len(sel_idx), dtype=bool)
            # drop if counter>=5 and slack>=5e-7
            drop = (neg_count[sel_idx] >= 5) & (slack[sel_idx] >= 5e-7)
            # keep at least 400 tightest
            if np.sum(~drop) >= 400:
                keep = ~drop
                sel_idx = sel_idx[keep]
            else:
                # keep top 400 by c_all regardless
                order = np.argsort(-c_all[sel_idx])
                keep2 = np.zeros(len(sel_idx), dtype=bool)
                keep2[order[:400]] = True
                # also keep non-dropped
                keep2 = keep2 | (~drop)
                sel_idx = sel_idx[keep2]
        p_ = len(sel_idx)
        if p_ == 0:
            break
        I = iu0_all[sel_idx]
        J = iu1_all[sel_idx]
        csub = c_all[sel_idx]
        Barr = build_bases(X)
        BI = Barr[I]  # (p,d,dm)
        BJ = Barr[J]
        XI = X[I]
        XJ = X[J]
        CI = np.einsum('pab,pa->pb', BI, XJ)
        CJ = np.einsum('pab,pa->pb', BJ, XI)

        # Build sparse A_ub: CI y_I + CJ y_J - t <= -csub
        try:
            cols = np.empty((p_, 2 * dm + 1), dtype=np.int64)
            cols[:, :dm] = I[:, None] * dm + ar[None, :]
            cols[:, dm:2 * dm] = J[:, None] * dm + ar[None, :]
            cols[:, 2 * dm] = Nvar
            vals = np.empty((p_, 2 * dm + 1), dtype=np.float64)
            vals[:, :dm] = CI
            vals[:, dm:2 * dm] = CJ
            vals[:, 2 * dm] = -1.0
            rows = np.repeat(np.arange(p_, dtype=np.int64), 2 * dm + 1)
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p_, Nvar + 1)).tocsr()
            b_ub = -csub
            bounds = [(-rho, rho)] * Nvar + [(-2.0, cmax)]
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'presolve': True})
        except Exception:
            rho *= 0.5
            continue
        if not res.success:
            rho *= 0.5
            continue
        y = res.x[:Nvar].reshape(n, dm)
        t_pred = float(res.x[-1])
        dX = np.einsum('ndm,nm->nd', Barr, y)
        Xn = _norm(X + dX)
        vn = _maxcos(Xn)
        if vn < cur - 1e-14:
            X = Xn
            cur = vn
            if vn < bestV:
                bestV = vn
                bestX = Xn.copy()
            # adjust rho: grow if predicted improvement realized
            if t_pred < cmax - 1e-12:
                rho = min(rho * 1.4, 5e-3)
            else:
                rho = rho * 0.85
                if rho < 1e-9:
                    rho = 1e-9
            # KKT NNLS on tightest ~400
            try:
                if time.time() - t0 > BUDGET - 3:
                    continue
                G2 = Xn @ Xn.T
                c2 = G2[iu0_all, iu1_all]
                cmax2 = float(np.max(c2))
                Kk = min(400, P)
                kidx = np.argpartition(-c2, Kk - 1)[:Kk]
                # build Gmat (Nvar x Kk)
                Ik = iu0_all[kidx]
                Jk = iu1_all[kidx]
                Barr2 = build_bases(Xn)
                CIk = np.einsum('pab,pa->pb', Barr2[Ik], Xn[Jk])
                CJk = np.einsum('pab,pa->pb', Barr2[Jk], Xn[Ik])
                Gm = np.zeros((Nvar, Kk), dtype=np.float64)
                for qq in range(Kk):
                    Gm[Ik[qq] * dm:(Ik[qq] + 1) * dm, qq] = CIk[qq]
                    Gm[Jk[qq] * dm:(Jk[qq] + 1) * dm, qq] += CJk[qq]
                w = 10.0
                Aaug = np.vstack([Gm, w * np.ones((1, Kk))])
                baug = np.zeros(Nvar + 1)
                baug[-1] = w
                lam, rnorm = nnls(Aaug, baug)
                lmax = float(np.max(lam)) if Kk > 0 else 0.0
                thr = 1e-10 if lmax == 0 else lmax * 1e-7
                for qq in range(Kk):
                    g = int(kidx[qq])
                    if lam[qq] <= thr:
                        neg_count[g] += 1
                    else:
                        neg_count[g] = 0
                # decay others slightly? keep as is
            except Exception:
                pass
        else:
            rho *= 0.5

    return _norm(bestX)

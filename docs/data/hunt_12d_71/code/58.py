import time
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import qr as scipy_qr

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n, d = X.shape
    Bs = np.empty((n, d, d - 1), dtype=np.float64)
    for i in range(n):
        xi = X[i]
        # null space of 1x12 row via full SVD
        try:
            _, _, Vh = np.linalg.svd(xi.reshape(1, d), full_matrices=True)
            B = Vh[1:, :].T  # 12x11
        except Exception:
            # fallback Gram-Schmidt
            Q, _ = np.linalg.qr(np.concatenate([xi.reshape(d, 1), np.eye(d)], axis=1))
            B = Q[:, 1:d]
        # ensure orthogonality to xi
        # re-orthogonalize once
        proj = B.T @ xi
        B = B - np.outer(xi, proj)
        # normalize columns
        cn = np.linalg.norm(B, axis=0, keepdims=True) + 1e-30
        B = B / cn
        # re-project to be safe (one step)
        # use QR to orthonormalize B in subspace orthogonal to xi
        try:
            Qb, _ = np.linalg.qr(B, mode='reduced')
            # Qb may have component along xi; remove
            c = Qb.T @ xi
            Qb = Qb - np.outer(xi, c)
            Qb, _ = np.linalg.qr(Qb, mode='reduced')
            B = Qb[:, :d-1]
        except Exception:
            pass
        Bs[i] = B
    return Bs

def _build_J0(X, Bs, I, J):
    P = len(I)
    n, d = X.shape
    tdim = d - 1
    N = n * tdim
    A = np.zeros((P, N), dtype=np.float64)
    for k in range(P):
        i = int(I[k]); j = int(J[k])
        bi = Bs[i].T @ X[j]  # 11
        bj = Bs[j].T @ X[i]
        A[k, i*tdim:(i+1)*tdim] = bi
        A[k, j*tdim:(j+1)*tdim] = bj
    return A

def _qr_select(A, k_target=700):
    # A: P x N, select well-conditioned rows via pivoted QR on A.T
    P, N = A.shape
    if P <= k_target:
        return np.arange(P, dtype=int)
    try:
        # A.T is N x P
        Q, R, perm = scipy_qr(A.T, pivoting=True)
        # R shape N x P, diag length min(N,P)
        m = min(N, P)
        dg = np.abs(np.diag(R[:m, :m]))
        # cutoff ill-conditioned
        if dg[0] > 0:
            ratio = dg / dg[0]
            # find first where ratio < 1e-9
            bad = np.where(ratio < 1e-9)[0]
            if len(bad) > 0 and bad[0] < k_target:
                k_target = max(300, int(bad[0]))
        k_target = min(k_target, P)
        sel = np.asarray(perm[:k_target], dtype=int)
        return sel
    except Exception:
        return np.argsort(np.linalg.norm(A, axis=1))[::-1][:k_target].astype(int) if False else np.arange(k_target, dtype=int)

def solve():
    t_start = time.time()
    DEADLINE = t_start + 170.0
    n, d = 71, 12
    tdim = d - 1
    N = n * tdim  # 781
    SCALE = 1e6
    BETA = 1.0

    # load best
    try:
        X0 = np.load("best.npy")
        X0 = np.asarray(X0, dtype=np.float64)
    except Exception:
        X0 = None
    try:
        Xr = np.load("record.npy")
        Xr = np.asarray(Xr, dtype=np.float64)
    except Exception:
        Xr = None
    cands = []
    if X0 is not None and X0.shape == (71, 12):
        cands.append(_norm_rows(X0))
    if Xr is not None and Xr.shape == (71, 12):
        cands.append(_norm_rows(Xr))
    if len(cands) == 0:
        rng = np.random.default_rng(0)
        cands.append(_norm_rows(rng.standard_normal((71, 12))))
    # pick best among cands
    bestX = cands[0]
    bestM = _max_cos(bestX)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bestM:
            bestM = v
            bestX = c.copy()
    Xcur = bestX.copy()
    curM = bestM

    # outer reselection 3-5 times
    deltas = [1.0e-5, 8.0e-6, 6.0e-6, 4.0e-6, 3.0e-6]
    n_outer = 5
    for outer in range(n_outer):
        if time.time() > DEADLINE - 28.0:
            break
        # time per LS cap ~25s: control via max_nfev
        remaining = DEADLINE - time.time()
        if remaining < 12.0:
            break
        m = _max_cos(Xcur)
        # build pair list within 1e-4
        G = Xcur @ Xcur.T
        iu0 = np.triu_indices(n, 1)
        vals = G[iu0]
        thresh = 1e-4
        # adapt thresh to get P in [400,1200]
        # first try 1e-4
        mask = vals > m - thresh
        P = int(np.count_nonzero(mask))
        if P < 350:
            thresh = 3e-4
            mask = vals > m - thresh
            P = int(np.count_nonzero(mask))
        if P > 1400:
            # keep tightest 1400
            order = np.argsort(-vals)
            keep = order[:1400]
            mask = np.zeros_like(mask)
            mask[keep] = True
            P = 1400
        if P < 50:
            # nothing to equalize
            break
        Iall = iu0[0][mask]
        Jall = iu0[1][mask]
        Vall = vals[mask]
        # sort tightest first for stability (QR will reorder anyway)
        od = np.argsort(-Vall)
        Iall = Iall[od]; Jall = Jall[od]; Vall = Vall[od]
        # tangent bases at Xcur
        Bs = _tangent_bases(Xcur)
        A = _build_J0(Xcur, Bs, Iall, Jall)
        sel = _qr_select(A, k_target=700)
        sel = np.asarray(sel, dtype=int)
        # map to pairs
        Is = Iall[sel]; Js = Jall[sel]
        K = len(Is)
        if K == 0:
            break
        # target
        delta = deltas[outer] if outer < len(deltas) else 3e-6
        # if previous outer failed, delta already small; keep
        t_target = m - delta
        X0b = Xcur.copy()

        # closures
        # precompute for speed: Bs, X0b, Is, Js
        def apply_y(y):
            # y: (N,)
            Y = y.reshape(n, tdim)
            # D = Bi @ yi
            # vectorized: (n,12,11) * (n,11) -> (n,12)
            D = np.einsum('nik,nk->ni', Bs, Y)
            Xn = X0b + D
            nn = np.linalg.norm(Xn, axis=1, keepdims=True)
            nn[~np.isfinite(nn)] = 1.0
            nn[nn == 0] = 1.0
            Xn = Xn / nn
            return Xn, nn[:, 0]

        def fun(y):
            Xn, nn = apply_y(y)
            # dots for selected
            dots = np.sum(Xn[Is] * Xn[Js], axis=1)
            r1 = (dots - t_target) * SCALE
            r2 = y * BETA
            return np.concatenate([r1, r2])

        def jac(y):
            Xn, nn = apply_y(y)
            # J_dot K x N
            Jd = np.zeros((K, N), dtype=np.float64)
            # loop
            for k in range(K):
                i = int(Is[k]); j = int(Js[k])
                xi = Xn[i]; xj = Xn[j]
                dot = float(xi @ xj)
                ni = float(nn[i]); nj = float(nn[j])
                fi = (xj - dot * xi) / (ni + 1e-30)  # 12
                fj = (xi - dot * xj) / (nj + 1e-30)
                Jd[k, i*tdim:(i+1)*tdim] = Bs[i].T @ fi
                Jd[k, j*tdim:(j+1)*tdim] = Bs[j].T @ fj
            Jd *= SCALE
            # stack reg
            Jr = np.eye(N, dtype=np.float64) * BETA
            return np.vstack([Jd, Jr])

        y0 = np.zeros(N, dtype=np.float64)
        try:
            # cap iterations to fit ~25s
            res = least_squares(fun, y0, jac=jac, method='lm',
                                ftol=1e-14, xtol=1e-14, gtol=1e-14,
                                max_nfev=60, verbose=0)
        except Exception:
            continue
        if time.time() > DEADLINE - 5.0:
            # still evaluate
            pass
        try:
            ysol = np.asarray(res.x, dtype=np.float64)
            if not np.all(np.isfinite(ysol)):
                continue
            Xnew, _ = apply_y(ysol)
            Xnew = _norm_rows(Xnew)
            if not np.all(np.isfinite(Xnew)):
                continue
            m_new = _max_cos(Xnew)
            # accept only true-max improvement
            if m_new < curM - 1e-12 and m_new < m - 1e-12:
                Xcur = Xnew
                curM = m_new
                if curM < bestM:
                    bestM = curM
                    bestX = Xcur.copy()
                # keep same delta ambition next round (success)
            else:
                # try smaller target once with same active set if time permits
                # attempt half delta
                if time.time() < DEADLINE - 30.0:
                    t_target2 = m - delta * 0.4
                    # quick second solve with warm start ysol*0.2?
                    def fun2(y):
                        Xn, nn = apply_y(y)
                        dots = np.sum(Xn[Is] * Xn[Js], axis=1)
                        r1 = (dots - t_target2) * SCALE
                        r2 = y * BETA
                        return np.concatenate([r1, r2])
                    def jac2(y):
                        Xn, nn = apply_y(y)
                        Jd = np.zeros((K, N), dtype=np.float64)
                        for k in range(K):
                            i = int(Is[k]); j = int(Js[k])
                            xi = Xn[i]; xj = Xn[j]
                            dot = float(xi @ xj)
                            ni = float(nn[i]); nj = float(nn[j])
                            fi = (xj - dot * xi) / (ni + 1e-30)
                            fj = (xi - dot * xj) / (nj + 1e-30)
                            Jd[k, i*tdim:(i+1)*tdim] = Bs[i].T @ fi
                            Jd[k, j*tdim:(j+1)*tdim] = Bs[j].T @ fj
                        Jd *= SCALE
                        Jr = np.eye(N, dtype=np.float64) * BETA
                        return np.vstack([Jd, Jr])
                    try:
                        res2 = least_squares(fun2, np.zeros(N), jac=jac2, method='lm',
                                             ftol=1e-14, xtol=1e-14, gtol=1e-14,
                                             max_nfev=40, verbose=0)
                        y2 = np.asarray(res2.x)
                        X2, _ = apply_y(y2)
                        # note apply_y still uses t_target closure? need recompute with same Bs/X0b
                        # apply_y is same, fine
                        X2 = _norm_rows(X2)
                        m2 = _max_cos(X2)
                        if m2 < curM - 1e-12:
                            Xcur = X2; curM = m2
                            if curM < bestM:
                                bestM = curM; bestX = Xcur.copy()
                    except Exception:
                        pass
        except Exception:
            continue

    bestX = _norm_rows(np.asarray(bestX, dtype=np.float64))
    return bestX

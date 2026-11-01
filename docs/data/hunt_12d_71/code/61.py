import time
import numpy as np

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
    # per-point SVD bases: null space of 1x12 row vector
    n, d = X.shape
    B = np.empty((n, d, d - 1), dtype=float)
    for i in range(n):
        x = X[i]
        # SVD of 1x12 matrix
        try:
            _, _, Vt = np.linalg.svd(x.reshape(1, d), full_matrices=True)
            Bi = Vt[1:, :].T  # 12x11
        except Exception:
            # fallback Householder via QR
            A = np.eye(d) - np.outer(x, x)
            Q, _ = np.linalg.qr(A + 1e-12 * np.eye(d))
            # pick columns most orthogonal to x
            dots = np.abs(Q.T @ x)
            idx = np.argsort(dots)[:d - 1]
            Bi = Q[:, idx]
        # re-orthogonalize against x
        Bi = Bi - x[:, None] * (x @ Bi)[None, :]
        # orthonormalize columns
        try:
            Qb, _ = np.linalg.qr(Bi)
            Bi = Qb
        except Exception:
            pass
        B[i] = Bi
    return B

def solve():
    import time as _tm
    t0 = _tm.time()
    HARD_END = t0 + 168.0
    try:
        from scipy.linalg import qr as _qr
        from scipy.optimize import least_squares as _ls
        has_scipy = True
    except Exception:
        has_scipy = False

    n, d = 71, 12
    # load start
    bestX = None
    best_m = 1e100
    for fn in ("best.npy", "record.npy"):
        try:
            Xc = _norm_rows(np.asarray(np.load(fn), dtype=float))
            if Xc.shape == (71, 12) and np.all(np.isfinite(Xc)):
                m = _max_cos(Xc)
                if m < best_m:
                    best_m = m
                    bestX = Xc.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = _norm_rows(rng0.standard_normal((n, d)))
        best_m = _max_cos(bestX)
    Xcur = bestX.copy()
    mcur = float(best_m)
    Ntan = n * (d - 1)  # 781

    if not has_scipy:
        return _norm_rows(Xcur)

    S = 1e6  # residual scaling
    # schedule over 5 solves
    K_target_list = [700, 700, 680, 700, 650]
    reg_list = [1e3, 3e3, 1e3, 8e2, 3e3]
    BAND = 1e-4

    n_solves = 0
    max_solves = 5
    for it in range(max_solves):
        if _tm.time() > HARD_END - 30.0:
            break
        # --- build active set within BAND of current max ---
        G = Xcur @ Xcur.T
        iu0 = np.triu_indices(n, 1)
        cuv = G[iu0]
        m0 = float(cuv.max())
        # sync mcur with Xcur
        # select pairs within BAND
        sel_mask = cuv > (m0 - BAND)
        Iall = iu0[0][sel_mask]
        Jall = iu0[1][sel_mask]
        Call = cuv[sel_mask]
        P = len(Iall)
        if P == 0:
            break
        # order by tightness for fallback
        order_tight = np.argsort(-Call)
        # --- tangent bases at Xcur ---
        Bstack = _tangent_bases(Xcur)  # 71x12x11
        # --- linearised Jacobian rows at y=0: grad = x_b^T B_a ---
        # compute quickly vectorized
        # XB[a,b]? need per pair
        # J0 shape (P, Ntan)
        # To avoid huge alloc if P large (>3000), cap P for QR by tightest 1500
        cap_qr = 1500
        if P > cap_qr:
            keep = order_tight[:cap_qr]
            Iqr = Iall[keep]; Jqr = Jall[keep]; Cqr = Call[keep]
        else:
            Iqr, Jqr, Cqr = Iall, Jall, Call
        Pqr = len(Iqr)
        # build dense J0 for QR (Pqr x 781)
        # J0[k, a*11:(a+1)*11] = Xcur[J]^T B_a ; similarly for b
        J0 = np.zeros((Pqr, Ntan), dtype=float)
        # vectorized per point? loop over pairs is ~1500*22 ops fine
        Xc = Xcur
        for k in range(Pqr):
            a = int(Iqr[k]); b = int(Jqr[k])
            # 11-vectors
            J0[k, a * 11:(a + 1) * 11] = Xc[b] @ Bstack[a]
            J0[k, b * 11:(b + 1) * 11] = Xc[a] @ Bstack[b]
        # --- pivoted QR on J0.T to select well-conditioned rows ---
        Kt = K_target_list[it] if it < len(K_target_list) else 700
        K = int(min(Pqr, Kt))
        # ensure max pair included
        # find index of global max within Iqr/Jqr
        try:
            # global max pair
            flat_max = int(np.argmax(cuv))
            amax = int(iu0[0][flat_max]); bmax = int(iu0[1][flat_max])
            # locate in Iqr/Jqr
            has_max = False
            max_pos = -1
            for k in range(Pqr):
                if (int(Iqr[k]) == amax and int(Jqr[k]) == bmax):
                    max_pos = k
                    has_max = True
                    break
        except Exception:
            has_max = False
            max_pos = -1
        try:
            Qm, Rm, piv = _qr(J0.T, pivoting=True)
            sel_idx = np.asarray(piv[:K], dtype=int)
            if has_max and (max_pos not in set(sel_idx.tolist())):
                # force include max: replace last
                sel_idx[-1] = max_pos
        except Exception:
            # fallback: tightest K
            o = np.argsort(-Cqr)[:K]
            sel_idx = o
        Isel = Iqr[sel_idx]
        Jsel = Jqr[sel_idx]
        # --- nonlinear least squares equalisation ---
        reg_alpha = float(reg_list[it] if it < len(reg_list) else 1e3)
        # closures
        X0loc = Xcur.copy()
        Bloc = Bstack.copy()
        m0loc = float(m0)
        Ksel = len(Isel)
        # preallocate? define fun/jac
        def fun(p):
            y = p[:-1]
            dt = p[-1]
            Ym = y.reshape(n, d - 1)
            # V = X0 + B*y
            # einsum: (n,12,11),(n,11)->(n,12)
            V = X0loc + np.einsum('ijk,ik->ij', Bloc, Ym)
            nr = np.linalg.norm(V, axis=1)
            nr[nr == 0] = 1.0
            Xn = V / nr[:, None]
            dots = np.sum(Xn[Isel] * Xn[Jsel], axis=1)
            r_dot = (dots - m0loc - dt) * S
            r_reg = reg_alpha * y
            return np.concatenate([r_dot, r_reg])

        def jac(p):
            y = p[:-1]
            # dt = p[-1] unused for geometry
            Ym = y.reshape(n, d - 1)
            V = X0loc + np.einsum('ijk,ik->ij', Bloc, Ym)
            nr = np.linalg.norm(V, axis=1)
            nr[nr == 0] = 1.0
            Xn = V / nr[:, None]
            # C_i = (B_i - Xn_i*(Xn_i^T B_i))/n_i
            # XB = Xn_i @ B_i : (n,11)
            XB = np.einsum('ij,ijk->ik', Xn, Bloc)
            # Ci stack: (n,12,11)
            Ci = (Bloc - Xn[:, :, None] * XB[:, None, :]) / nr[:, None, None]
            M = Ksel + Ntan
            Nv = Ntan + 1
            J = np.zeros((M, Nv), dtype=float)
            # fill dot block
            # for each k, two blocks
            # vectorize using loops (700 iters fine)
            for k in range(Ksel):
                a = int(Isel[k]); b = int(Jsel[k])
                # d dot / d Ya = Xn_b @ C_a  (11,)
                Ja = Xn[b] @ Ci[a]  # (11,)
                Jb = Xn[a] @ Ci[b]
                J[k, a * 11:(a + 1) * 11] = Ja * S
                J[k, b * 11:(b + 1) * 11] = Jb * S
                J[k, Ntan] = -S
            # reg block: diagonal
            # rows Ksel : Ksel+Ntan, cols 0:Ntan
            # set diagonal efficiently
            J[Ksel:, :Ntan] = np.eye(Ntan) * reg_alpha
            # last column already 0
            return J

        p0 = np.zeros(Ntan + 1, dtype=float)
        # cap time: estimate remaining, allow up to 25s per call but leave 25s margin at end
        # least_squares has no timeout; limit max_nfev
        try:
            res = _ls(fun, p0, jac=jac, method='lm',
                      xtol=1e-14, ftol=1e-14, gtol=1e-14,
                      max_nfev=80, x_scale='jac', verbose=0)
        except Exception:
            continue
        n_solves += 1
        try:
            pstar = res.x
            ystar = pstar[:-1]
            Ym = ystar.reshape(n, d - 1)
            V = X0loc + np.einsum('ijk,ik->ij', Bloc, Ym)
            Xnew = _norm_rows(V)
            mnew = _max_cos(Xnew)
        except Exception:
            continue
        if np.isfinite(mnew) and mnew < mcur - 1e-15:
            Xcur = Xnew
            mcur = float(mnew)
            # continue to next re-selection
        else:
            # no improvement: try one smaller-step retry with larger reg if budget left?
            # do a single retry with reg*8 without re-selecting (using same selection) if not yet tried
            # only once to save time
            if _tm.time() > HARD_END - 30.0:
                break
            # retry with stronger anchoring starting from same X0loc
            reg_alpha2 = reg_alpha * 8.0
            def fun2(p):
                y = p[:-1]; dt = p[-1]
                Ym2 = y.reshape(n, d - 1)
                V2 = X0loc + np.einsum('ijk,ik->ij', Bloc, Ym2)
                nr2 = np.linalg.norm(V2, axis=1); nr2[nr2 == 0] = 1.0
                Xn2 = V2 / nr2[:, None]
                dots2 = np.sum(Xn2[Isel] * Xn2[Jsel], axis=1)
                return np.concatenate([(dots2 - m0loc - dt) * S, reg_alpha2 * y])
            def jac2(p):
                y = p[:-1]
                Ym2 = y.reshape(n, d - 1)
                V2 = X0loc + np.einsum('ijk,ik->ij', Bloc, Ym2)
                nr2 = np.linalg.norm(V2, axis=1); nr2[nr2 == 0] = 1.0
                Xn2 = V2 / nr2[:, None]
                XB2 = np.einsum('ij,ijk->ik', Xn2, Bloc)
                Ci2 = (Bloc - Xn2[:, :, None] * XB2[:, None, :]) / nr2[:, None, None]
                J2 = np.zeros((Ksel + Ntan, Ntan + 1), dtype=float)
                for k in range(Ksel):
                    a = int(Isel[k]); b = int(Jsel[k])
                    J2[k, a*11:(a+1)*11] = (Xn2[b] @ Ci2[a]) * S
                    J2[k, b*11:(b+1)*11] = (Xn2[a] @ Ci2[b]) * S
                    J2[k, Ntan] = -S
                J2[Ksel:, :Ntan] = np.eye(Ntan) * reg_alpha2
                return J2
            try:
                res2 = _ls(fun2, p0, jac=jac2, method='lm',
                           xtol=1e-14, ftol=1e-14, gtol=1e-14,
                           max_nfev=60, x_scale='jac', verbose=0)
                p2 = res2.x
                Ym2 = p2[:-1].reshape(n, d - 1)
                V2 = X0loc + np.einsum('ijk,ik->ij', Bloc, Ym2)
                Xnew2 = _norm_rows(V2)
                mnew2 = _max_cos(Xnew2)
                if np.isfinite(mnew2) and mnew2 < mcur - 1e-15:
                    Xcur = Xnew2
                    mcur = float(mnew2)
                    continue
            except Exception:
                pass
            # if still no improvement, stop (further re-selects would repeat)
            break
        if _tm.time() > HARD_END - 5.0:
            break

    # return best between initial best and Xcur
    Xout = _norm_rows(Xcur if mcur < best_m else bestX)
    return Xout

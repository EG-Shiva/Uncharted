import time
import numpy as np

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n = X.shape[0]
    B = np.empty((n, 4, 3), dtype=np.float64)
    for i in range(n):
        # null space of 1x4 row via SVD; first vh row = X[i]
        _, _, vh = np.linalg.svd(X[i:i+1], full_matrices=True)
        B[i] = vh[1:4, :].T
    return B

def _eval_max(X0, Delta, alpha):
    Z = X0 + alpha * Delta
    Zn = _normed(Z)
    return _max_cos(Zn), Zn

def _lp_polish_tight(x, deadline, r0=1e-5):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _normed(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-15:
        c = (x @ x.T)[iu]
        mval = float(c.max())
        sel = c > mval - max(20.0 * radius, 1e-12)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method='highs')
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, 'status', 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _normed(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    import time
    t0 = time.time()
    TL = 172.0
    deadline = t0 + TL
    # incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (292, 4) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if nr.min() > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        rng = np.random.default_rng(292)
        Xb = _normed(rng.standard_normal((292, 4)))
    else:
        vals = [_max_cos(c) for c in cands]
        Xb = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(Xb)
    Xcur = Xb.copy()
    cur_val = best_val
    n = 292
    q = 3
    NV = n * q
    # active-set equalization loop
    # budget equalization until ~ deadline-25s
    eq_deadline = deadline - 25.0
    it = 0
    try:
        from scipy.optimize import least_squares, nnls
        has_ls = True
    except Exception:
        has_ls = False
    while has_ls and time.time() < eq_deadline and it < 12:
        it += 1
        # build tangent bases at Xcur
        try:
            B = _tangent_bases(Xcur)
        except Exception:
            break
        G = Xcur @ Xcur.T
        iu0 = np.triu_indices(n, 1)
        c_all = G[iu0]
        m = float(c_all.max())
        slack = m - c_all
        order = np.argsort(slack)
        n_thresh = int(np.sum(slack < 5e-7))
        Pwant = 1200
        if n_thresh >= Pwant:
            # take all below thresh but cap at 1600 for tractability
            if n_thresh > 1600:
                sel_idx = order[:1600]
            else:
                sel_idx = order[:n_thresh]
        else:
            sel_idx = order[:Pwant]
        p = len(sel_idx)
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        # least squares equality c_ij(y)-t=0
        v0 = np.zeros(NV + 1, dtype=np.float64)
        v0[-1] = m
        y_bound = 0.02
        lb = np.full(NV + 1, -y_bound)
        ub = np.full(NV + 1, y_bound)
        lb[-1] = m - 0.005
        ub[-1] = m + 1e-9
        # ensure v0 inside
        # closures for speed
        Bc = B
        Ic = I
        Jc = J
        def fun(v):
            y = v[:-1].reshape(n, q)
            t = v[-1]
            Z = Xcur + np.einsum('nik,nk->ni', Bc, y)
            nn = np.sqrt((Z * Z).sum(axis=1))
            nn = np.maximum(nn, 1e-12)
            Xn = Z / nn[:, None]
            Xi = Xn[Ic]
            Xj = Xn[Jc]
            cc = (Xi * Xj).sum(axis=1)
            return cc - t
        def jac(v):
            y = v[:-1].reshape(n, q)
            Z = Xcur + np.einsum('nik,nk->ni', Bc, y)
            nn = np.sqrt((Z * Z).sum(axis=1))
            nn = np.maximum(nn, 1e-12)
            Xn = Z / nn[:, None]
            Xi = Xn[Ic]
            Xj = Xn[Jc]
            cc = (Xi * Xj).sum(axis=1)
            ni = nn[Ic]
            nj = nn[Jc]
            Bi = Bc[Ic]  # p,4,3
            Bj = Bc[Jc]
            # Bi^T Xj etc
            BiTXj = np.einsum('pab,pa->pb', Bi, Xj)
            BiTXi = np.einsum('pab,pa->pb', Bi, Xi)
            BjTXi = np.einsum('pab,pa->pb', Bj, Xi)
            BjTXj = np.einsum('pab,pa->pb', Bj, Xj)
            gi = (BiTXj - BiTXi * cc[:, None]) / ni[:, None]
            gj = (BjTXi - BjTXj * cc[:, None]) / nj[:, None]
            Jac = np.zeros((p, NV + 1), dtype=np.float64)
            rows = np.arange(p)
            # fill blocks without python loop over p
            for a in range(3):
                Jac[rows, Ic * 3 + a] += gi[:, a]
                Jac[rows, Jc * 3 + a] += gj[:, a]
            Jac[:, -1] = -1.0
            return Jac
        try:
            res = least_squares(fun, v0, jac=jac, bounds=(lb, ub),
                                xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                max_nfev=60, verbose=0)
        except Exception:
            break
        if not hasattr(res, 'x'):
            break
        vopt = res.x
        yopt = vopt[:-1].reshape(n, q)
        topt = float(vopt[-1])
        Delta = np.einsum('nik,nk->ni', Bc, yopt)
        if not np.all(np.isfinite(Delta)) or np.linalg.norm(Delta) < 1e-15:
            # nothing to move; try multiplier check then break
            pass
        else:
            # exact line search on true max cosine
            alphas = np.linspace(0.0, 1.5, 31)
            best_a = 0.0
            best_X = Xcur
            best_m = cur_val
            for a in alphas:
                if a == 0.0:
                    continue
                v, Xa = _eval_max(Xcur, Delta, a)
                if v < best_m:
                    best_m = v
                    best_a = a
            if best_a > 0:
                # refine around best_a
                lo = max(0.0, best_a - 0.05)
                hi = best_a + 0.05
                for _ in range(3):
                    fine = np.linspace(lo, hi, 21)
                    bm2 = best_m
                    ba2 = best_a
                    Xa_best = best_X
                    for a in fine:
                        v, Xa = _eval_max(Xcur, Delta, a)
                        if v < bm2:
                            bm2 = v
                            ba2 = a
                            Xa_best = Xa
                    best_m = bm2
                    best_a = ba2
                    if ba2 == best_a:
                        best_X = Xa_best
                    step = (hi - lo) / 20.0
                    lo = max(0.0, ba2 - step * 2)
                    hi = ba2 + step * 2
                    if step < 1e-12:
                        break
                # re-evaluate best_X at best_a to be safe
                _, best_X = _eval_max(Xcur, Delta, best_a)
                if best_m < cur_val - 1e-13:
                    Xcur = _normed(best_X)
                    cur_val = float(best_m)
                    if cur_val < best_val:
                        best_val = cur_val
                        Xb = Xcur.copy()
                    # check equalization: active slacks at new point
                    Gn = Xcur @ Xcur.T
                    cn = Gn[iu0]
                    mn = float(cn.max())
                    sl = mn - cn
                    # take active slacks for old active set
                    # map: need slacks of pairs in Ic/Jc -> find via slack of those indices
                    # sl_active = mn - (Xn[Ic].Xn[Jc]) already; approximate via sl[sel_idx]
                    act_sl = sl[sel_idx]
                    if float(act_sl.max()) < 1e-12:
                        break
                    # KKT multiplier check via NNLS at new point
                    try:
                        # build grad matrix at Xcur (y=0 basis new)
                        B2 = _tangent_bases(Xcur)
                        # recompute grads for current active (top 1200 at new point)
                        order2 = np.argsort(sl)
                        sel2 = order2[:1200]
                        I2 = iu0[0][sel2]; J2 = iu0[1][sel2]
                        p2 = len(sel2)
                        # grads at y=0: dc/dy_i = B_i^T Xj, dc/dy_j = B_j^T Xi
                        Xn2 = Xcur
                        Xi2 = Xn2[I2]; Xj2 = Xn2[J2]
                        Bi2 = B2[I2]; Bj2 = B2[J2]
                        gi2 = np.einsum('pab,pa->pb', Bi2, Xj2)
                        gj2 = np.einsum('pab,pa->pb', Bj2, Xi2)
                        # Gmat 876 x p2
                        Gmat = np.zeros((NV, p2))
                        # scatter: for each a, Gmat[I*3+a, k] = gi...
                        # vectorized via loop over p? use advanced accumulation
                        # do with for a in 3 using np.add.at? simple loop over p2 (1200) is fine
                        for k in range(p2):
                            ii = int(I2[k]); jj = int(J2[k])
                            Gmat[ii*3:(ii*3+3), k] = gi2[k]
                            Gmat[jj*3:(jj*3+3), k] = gj2[k]
                        w = 1.0
                        M = np.vstack([Gmat, np.ones((1, p2)) * w])
                        bb = np.zeros(NV + 1); bb[-1] = w
                        lam, rnorm = nnls(M, bb)
                        # residual norm
                        rvec = M @ lam - bb
                        rn = float(np.linalg.norm(rvec) / max(1.0, abs(w)))
                        # drop zeros: count support
                        sup = int(np.sum(lam > 1e-9))
                        # if stationary (small residual) -> break
                        if rn < 1e-7:
                            break
                        # otherwise continue loop (next iteration will re-select top1200,
                        # implicitly dropping zero-weight pairs if they remain loose)
                    except Exception:
                        pass
                    continue
                else:
                    # no improvement along direction -> try multiplier drop / different basin?
                    # fall through to break to final LP
                    pass
        # no acceptance: check time and break to final polish
        break
    # final tight LP with small initial radius
    try:
        Xb = _lp_polish_tight(Xb if 'Xb' in locals() else Xcur, deadline, r0=1e-5)
    except Exception:
        pass
    # safety: ensure finite, normalized, correct shape
    try:
        Xb = np.asarray(Xb, dtype=np.float64)
        Xb = _normed(Xb)
    except Exception:
        rng = np.random.default_rng(7)
        Xb = _normed(rng.standard_normal((292, 4)))
    return Xb

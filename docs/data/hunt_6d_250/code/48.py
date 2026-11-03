import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def solve():
    t_start = time.time()
    DEADLINE = t_start + 170.0
    n, d = 250, 6
    dd = d - 1  # 5 tangent dims
    # load best
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        rng0 = np.random.default_rng(0)
        z = rng0.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    # pick best
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    X = cands[bi].copy()
    X = _norm_rows(X)
    M = float(vals[bi])
    best_overall = X.copy()
    best_val = M

    # initial active set: all pairs within 1e-5 of max
    G0 = X @ X.T
    iu0 = np.triu_indices(n, 1)
    gall = G0[iu0]
    sel0 = gall > M - 1e-5
    I = iu0[0][sel0].astype(np.int64)
    J = iu0[1][sel0].astype(np.int64)
    streak = np.zeros(len(I), dtype=np.int64)
    # safety: if too few, expand window
    if len(I) < 200:
        sel0 = gall > M - 5e-5
        I = iu0[0][sel0].astype(np.int64)
        J = iu0[1][sel0].astype(np.int64)
        streak = np.zeros(len(I), dtype=np.int64)

    rho = 3e-4
    RHO_MIN_CERT = 1e-5
    RHO_MAX = 1e-2

    nv = n * dd + 1
    c_obj = np.zeros(nv)
    c_obj[-1] = 1.0

    it = 0
    # preallocate iu for reuse
    triu = np.triu_indices(n, 1)

    while time.time() < DEADLINE - 2.0:
        it += 1
        if len(I) == 0:
            # rebuild
            G = X @ X.T
            g = G[triu]
            Mcur = float(np.max(g))
            M = Mcur
            sel = g > Mcur - 1e-5
            I = triu[0][sel].astype(np.int64)
            J = triu[1][sel].astype(np.int64)
            streak = np.zeros(len(I), dtype=np.int64)
            if len(I) == 0:
                break
        p = len(I)
        # prune if absurdly large (keep closest)
        if p > 3500:
            c_sel_tmp = (X[I] * X[J]).sum(axis=1)
            ord_idx = np.argpartition(-c_sel_tmp, 3500)[:3500]
            I = I[ord_idx]; J = J[ord_idx]; streak = streak[ord_idx]
            p = len(I)

        # build Householder data for current X
        sgn = np.sign(X[:, 0])
        sgn[sgn == 0] = 1.0
        W = X.copy()
        W[:, 0] -= sgn
        dden = np.sum(W * W, axis=1)
        degen = dden < 1e-12
        # fix degenerate
        W[degen] = 0.0
        dden[degen] = 1.0

        # current cos for active pairs
        # compute via rows (fast)
        c_sel = np.sum(X[I] * X[J], axis=1)

        # WdotX = W @ X.T  (n x n)
        WdotX = W @ X.T
        # build A_ub rows
        # a_ij = Xj[1:] - 2*W_i[1:]*dot_i, dot_i=(W_i.X_j)/dden_i
        dot_i = WdotX[I, J] / dden[I]  # wait WdotX[i,j]=W_i.X_j? WdotX = W @ X.T so [i,j]=W_i.X_j correct
        dot_j = WdotX[J, I] / dden[J]  # [j,i]=W_j.X_i correct
        # careful: WdotX[J,I] uses row J col I = W_J . X_I good
        Amat = X[J][:, 1:] - 2.0 * W[I][:, 1:] * dot_i[:, None]
        Bmat = X[I][:, 1:] - 2.0 * W[J][:, 1:] * dot_j[:, None]

        rows = np.repeat(np.arange(p), 2 * dd + 1)
        # cols
        base_I = I[:, None] * dd + np.arange(dd)[None, :]
        base_J = J[:, None] * dd + np.arange(dd)[None, :]
        cols_I = base_I.ravel()
        cols_J = base_J.ravel()
        cols = np.empty(p * (2 * dd + 1), dtype=np.int64)
        cols[0::(2*dd+1)] = 0  # placeholder, will overwrite blockwise
        # faster: construct via concatenation
        ones_t = np.full(p, nv - 1, dtype=np.int64)
        cols = np.concatenate([base_I.ravel(), base_J.ravel(), ones_t]) if False else None
        # need row-major interleaving for coo: build explicitly
        # rows already repeat; cols must interleave per row: [I-block, J-block, t]
        cols_arr = np.empty(p * (2 * dd + 1), dtype=np.int64)
        vals_arr = np.empty(p * (2 * dd + 1), dtype=np.float64)
        # fill
        for k in range(p):
            o = k * (2 * dd + 1)
            vals_arr[o:o+dd] = Amat[k]
            cols_arr[o:o+dd] = I[k]*dd + np.arange(dd)
            vals_arr[o+dd:o+2*dd] = Bmat[k]
            cols_arr[o+dd:o+2*dd] = J[k]*dd + np.arange(dd)
            vals_arr[o+2*dd] = -1.0
            cols_arr[o+2*dd] = nv - 1
        rows_arr = np.repeat(np.arange(p), 2*dd+1)
        try:
            A_ub = coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(p, nv)).tocsr()
        except Exception:
            break
        b_ub = -c_sel
        bounds = [(-rho, rho)] * (n * dd) + [(None, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception:
            rho *= 0.5
            if rho < 1e-7:
                break
            continue
        if res.status != 0:
            rho *= 0.5
            if rho < 1e-7:
                # try certify: if already tiny, break
                break
            # if infeasible at tiny rho, break
            if time.time() > DEADLINE - 5:
                break
            continue
        t_pred = float(res.x[-1])
        Ustar = res.x[:-1].reshape(n, dd)

        # dual multipliers for dropping
        try:
            marg = None
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                # res.ineqlin may be dict-like or object
                try:
                    marg = np.asarray(res.ineqlin.marginals, dtype=float)
                except Exception:
                    try:
                        marg = np.asarray(res.ineqlin["marginals"], dtype=float)
                    except Exception:
                        marg = None
            if marg is None or marg.shape[0] != p:
                raise ValueError("no duals")
            is_zero = np.abs(marg) < 1e-9
        except Exception:
            is_zero = np.zeros(p, dtype=bool)

        # certification check: no predicted progress
        if t_pred >= M - 1e-12:
            if rho > RHO_MIN_CERT + 1e-12:
                # shrink to cert radius and re-solve
                rho = RHO_MIN_CERT
                # update streak for dropping before next solve
                streak = np.where(is_zero, streak + 1, 0)
                keep = streak < 3
                if np.count_nonzero(~keep) > 0:
                    I = I[keep]; J = J[keep]; streak = streak[keep]
                continue
            else:
                # certified local optimum at rho=1e-5
                streak = np.where(is_zero, streak + 1, 0)
                break

        # build tangent displacement D from Ustar
        raw = np.sum(W[:, 1:] * Ustar, axis=1)  # (n,)
        coeff = 2.0 * raw / dden
        Dmat = np.zeros((n, d))
        Dmat[:, 1:] = Ustar
        Dmat -= W * coeff[:, None]

        # exact line search over alphas on retracted points
        alphas = (1.0, 0.5, 0.25, 2.0)
        best_a = 0.0
        best_Xa = None
        best_Ma = M
        # evaluate
        for alpha in alphas:
            if time.time() > DEADLINE - 1.5:
                break
            Xa = X + alpha * Dmat
            Xa = _norm_rows(Xa)
            if not np.all(np.isfinite(Xa)):
                continue
            va = _max_cos(Xa)
            if va < best_Ma - 1e-13:
                best_Ma = va
                best_a = alpha
                best_Xa = Xa
        if best_Xa is not None and best_Ma < M - 1e-13:
            # accept
            X = best_Xa
            M = best_Ma
            if M < best_val - 1e-13:
                best_val = M
                best_overall = X.copy()
            # update streak
            streak = np.where(is_zero, streak + 1, 0)
            keep = streak < 3
            I = I[keep]; J = J[keep]; streak = streak[keep]
            # add every pair within 1e-6 of new max
            Gnew = X @ X.T
            gnew = Gnew[triu]
            add_sel = gnew > M - 1e-6
            # build set of existing
            existing = set((int(a)*n + int(b)) for a, b in zip(I, J))
            # also include symmetric? only upper
            nI = triu[0][add_sel]; nJ = triu[1][add_sel]
            newI = []; newJ = []
            for a, b in zip(nI, nJ):
                key = int(a)*n+int(b)
                if key not in existing:
                    newI.append(int(a)); newJ.append(int(b))
            if newI:
                I = np.concatenate([I, np.asarray(newI, dtype=np.int64)])
                J = np.concatenate([J, np.asarray(newJ, dtype=np.int64)])
                streak = np.concatenate([streak, np.zeros(len(newI), dtype=np.int64)])
            # adapt rho up
            rho = min(rho * 1.5, RHO_MAX)
        else:
            # reject: shrink
            streak = np.where(is_zero, streak + 1, 0)
            keep = streak < 3
            I = I[keep]; J = J[keep]; streak = streak[keep]
            rho *= 0.5
            if rho < 1e-7:
                # too small, if no prediction of progress, break
                # force cert check next iter
                if t_pred >= M - 1e-12:
                    break
                rho = 1e-7
            # optional: if active set became tiny, rebuild window around current max
            if len(I) < 100:
                Gc = X @ X.T
                gc = Gc[triu]
                Mc = float(np.max(gc))
                M = Mc
                sel = gc > Mc - 1e-5
                I = triu[0][sel].astype(np.int64)
                J = triu[1][sel].astype(np.int64)
                streak = np.zeros(len(I), dtype=np.int64)
        # time guard + iteration cap
        if it > 300:
            break
        # early exit if deadline near
        if time.time() >= DEADLINE - 1.0:
            break

    # return best found if below initial, else fallback (best_overall already tracks)
    Xout = best_overall / np.maximum(np.linalg.norm(best_overall, axis=1, keepdims=True), 1e-12)
    # final safety finite
    Xout = np.where(np.isfinite(Xout), Xout, 0.0)
    nn = np.linalg.norm(Xout, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    Xout = Xout / nn
    return Xout

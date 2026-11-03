import time
import numpy as np
from scipy.optimize import linprog, lsq_linear
from scipy.sparse import csr_matrix, hstack as sparse_hstack

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_bases(x):
    n, d = x.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    e1 = np.zeros(d); e1[0] = 1.0
    I = np.eye(d)
    for i in range(n):
        xi = x[i]
        s = 1.0 if xi[0] >= 0 else -1.0
        w = xi + s * e1
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            # xi ~ -s*e1, pick another axis
            e2 = np.zeros(d); e2[1] = 1.0
            w = xi + s * e2
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                w = xi.copy()
                nw = np.linalg.norm(w)
        u = w / nw
        H = I - 2.0 * np.outer(u, u)
        # first column is parallel to xi, rest is basis
        # ensure orientation: check which column aligns with xi
        # by construction col 0 ~ -s*xi, so take cols 1:
        B[i] = H[:, 1:]
        # re-orthogonalize against xi for safety
        # B should satisfy B.T @ xi ~ 0
        corr = B[i].T @ xi
        if np.max(np.abs(corr)) > 1e-10:
            # Gram-Schmidt cleanup
            Q, _ = np.linalg.qr(np.concatenate([xi[:, None], B[i]], axis=1))
            B[i] = Q[:, 1:]
    return B

def _sqp_polish(x0, deadline, verbose=False):
    x = x0.copy()
    n, d = x.shape
    dm = d - 1
    N = n * dm
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    best = _max_cos(x)
    best_x = x.copy()
    # trust radii schedule
    sched = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8]
    ri = 0
    radius = sched[0]
    it = 0
    fails = 0
    rng = np.random.default_rng(0)
    # fixed random null direction for determinism
    while time.time() < deadline:
        it += 1
        if ri >= len(sched):
            break
        radius = sched[ri]
        G = x @ x.T
        # max off-diagonal
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        c_all = (x @ x.T)[iu0]  # recompute without diag hack
        # c_all includes diag? iu excludes diag so fine
        slack = m - c_all
        active_tol = max(1e-6, 10.0 * radius)
        sel = slack < active_tol
        # cap to tightest K to keep LP tractable
        KCAP = 1800
        if np.sum(sel) > KCAP:
            # keep tightest KCAP
            idx = np.where(sel)[0]
            order = np.argsort(slack[idx])[:KCAP]
            mask2 = np.zeros_like(sel, dtype=bool)
            mask2[idx[order]] = True
            sel = mask2
        I = iuI[sel]; J = iuJ[sel]
        c_sel = c_all[sel]
        p = len(I)
        if p == 0:
            ri += 1
            continue
        B = _build_bases(x)
        # Build Jacobian p x N
        Jmat = np.zeros((p, N), dtype=float)
        # loop
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            Jmat[k, i*dm:(i+1)*dm] = B[i].T @ x[j]
            Jmat[k, j*dm:(j+1)*dm] = B[j].T @ x[i]
        # rank test every few iters
        rank_def = False
        null_dim = 0
        if it % 3 == 1 and p > 10:
            try:
                # economy SVD for rank estimate (use matrix_rank)
                r = np.linalg.matrix_rank(Jmat, tol=1e-7)
                mr = min(p, N)
                if r < mr - 2:
                    rank_def = True
                    null_dim = N - r
                elif N > p and r < N - 2:
                    rank_def = True
                    null_dim = N - r
            except Exception:
                rank_def = False
        # LP subproblem: min t s.t. Jmat*u - t <= -c_sel
        # A_ub = [Jmat, -1]
        try:
            A_ub = np.concatenate([Jmat, -np.ones((p, 1))], axis=1)
            A_ub_sp = csr_matrix(A_ub)
            b_ub = -c_sel
            c_obj = np.zeros(N+1); c_obj[-1] = 1.0
            bounds = [(-radius, radius)]*N + [(None, None)]
            res = linprog(c_obj, A_ub=A_ub_sp, b_ub=b_ub, bounds=bounds, method="highs",
                          options={"presolve": True})
        except Exception:
            ri += 1
            continue
        if res.status != 0:
            # infeasible/unbounded -> shrink
            ri += 1
            continue
        u_star = res.x[:-1]
        t_star = float(res.x[-1])
        pred_gain = m - t_star
        unorm = float(np.max(np.abs(u_star))) if N>0 else 0.0
        # if predicted gain tiny, trigger null-space equalization instead of collapse
        if pred_gain < 1e-12 or unorm < 0.05*radius:
            # null-space / equalization move via lsq_linear
            # equalize active pairs to their mean
            did_eq = False
            if time.time() < deadline - 1.0:
                try:
                    mean_c = float(np.mean(c_sel))
                    rhs = mean_c - c_sel  # want Jmat*u = rhs
                    # damp: solve bounded LS
                    ls = lsq_linear(Jmat, rhs, bounds=(-radius, radius), lsmr_tol='auto', verbose=0)
                    u_eq = ls.x
                    if np.all(np.finite(u_eq)) and np.max(np.abs(u_eq)) > 1e-13:
                        # line search on equalization: allow tiny uphill but reduce spread
                        # spread metric: std of active true cosines
                        def apply_alpha(al):
                            dx = (B * u_eq.reshape(n,1,dm)).sum(axis=2)  # n x d? check: B n x d x dm * u n x dm
                            # compute: dx_i = B_i @ u_i
                            # above broadcasting: B(n,d,dm)*u(n,1,dm) sum over dm -> (n,d)
                            Y = x + al*dx
                            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                            return Y
                        # precompute dx
                        U3 = u_eq.reshape(n, dm)
                        DX = np.einsum('nik,nk->ni', B, U3)
                        # current spread
                        cur_std = float(np.std(c_sel))
                        best_a = 0.0
                        best_Y = None
                        best_m_eq = m
                        # try alphas
                        for al in [1.0, 0.5, 0.25, 0.125]:
                            Y = x + al*DX
                            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                            Gt = Y @ Y.T
                            np.fill_diagonal(Gt, -2.0)
                            mt = float(np.max(Gt))
                            # active spread after (use same I,J index set)
                            c_after = np.array([Gt[I[k], J[k]] for k in range(min(p, 2000))])
                            std_after = float(np.std(c_after))
                            # accept if not worse by much and spread reduced, or max reduced
                            if mt < m - 1e-13:
                                best_a = al; best_Y = Y; best_m_eq = mt
                                break
                            if mt < m + 5e-9 and std_after < cur_std*(1-1e-4):
                                best_a = al; best_Y = Y; best_m_eq = mt
                                break
                        if best_Y is not None:
                            x = best_Y
                            if best_m_eq < best - 1e-15:
                                best = best_m_eq; best_x = x.copy()
                            # do not shrink radius on successful equalization
                            fails = 0
                            did_eq = True
                except Exception:
                    did_eq = False
            if not did_eq:
                # collapse radius
                ri += 1
                fails += 1
                if fails > 6 and ri < len(sched)-1:
                    # already at small radius, stop?
                    pass
            continue
        # line search with exact renormalization (second-order correction via normalization)
        U3 = u_star.reshape(n, dm)
        DX = np.einsum('nik,nk->ni', B, U3)
        # backtracking
        accepted = False
        for al in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            Y = x + al*DX
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr==0] = 1.0
            Y = Y / nr
            mt = _max_cos(Y)
            # second-order correction check: linear predicted t(al)= m - al*pred_gain? Actually t for scaled step ~ m - al*pred_gain ignoring curvature
            # true should be <= linear due to  -0.5c||dx||^2 denominator (favorable), so accept any true improvement
            if mt < m - 1e-15:
                x = Y
                if mt < best:
                    best = mt; best_x = Y.copy()
                # success: try to keep or expand radius slightly
                if ri > 0 and al >= 0.9:
                    ri = max(0, ri-1)
                fails = 0
                accepted = True
                break
        if not accepted:
            # shrink trust region
            ri += 1
            fails += 1
        # early stop if radius tiny
        if radius < 5e-8 and fails > 3:
            break
        # safety: if time low break
        if time.time() >= deadline - 0.5:
            break
    return best_x, best

def solve():
    T0 = time.time()
    BUDGET = 60.0  # at most 60s for SQP per assignment
    # load incumbent
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.default_rng(0)
        best = rng0.standard_normal(size=(94,13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumb = best.copy(); incumb_val = best_val
    deadline = T0 + BUDGET - 1.0
    try:
        px, pv = _sqp_polish(incumb, deadline)
    except Exception:
        px, pv = incumb, incumb_val
    px /= np.linalg.norm(px, axis=1, keepdims=True)
    # keep only if beats incumbent (strict)
    try:
        if not np.all(np.finite(px)) or px.shape != (94,13):
            return incumb
        fv = _max_cos(px)
        if fv < incumb_val - 1e-15:
            return px
        else:
            return incumb
    except Exception:
        return incumb

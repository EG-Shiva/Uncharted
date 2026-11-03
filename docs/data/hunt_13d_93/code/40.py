import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    # SQP budget ~ up to deadline-3s
    rng = np.random.default_rng(0)
    n, d = 93, 13
    m = d - 1
    nv = n * m + 1
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)

    # load start
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _norm(a)
            G = a @ a.T
            np.fill_diagonal(G, -2.0)
            v = float(G.max())
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)
    X = best.copy()
    cur = float(best_val)
    bestX = X.copy()

    ar12 = np.arange(m)
    # trust region
    radius = 3e-3
    rmax = 1.0e-2
    rmin = 1e-12

    # for linprog
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    c_obj = np.zeros(nv, dtype=np.float64)
    c_obj[-1] = 1.0

    use_filter = False
    filt_thresh = 1e-3
    slow_count = 0

    it = 0
    # preallocate index helpers for full problem
    full_rows_base = np.arange(P)

    while time.time() - t0 < DEADLINE - 3.0:
        it += 1
        if radius < rmin:
            # tiny random tangent kick to try new basin, keep best
            kick = rng.standard_normal(X.shape) * radius * 10.0
            kick -= (np.sum(kick * X, axis=1, keepdims=True)) * X
            X = _norm(X + kick)
            cur = _maxcos(X)
            radius = 1e-4
            if cur < best_val:
                best_val = cur
                bestX = X.copy()
            continue

        # Gram and values
        G = X @ X.T
        # extract pair cosines
        c_all = G[iu0, iu1]  # size P
        cmax = float(c_all.max())
        cur = cmax  # keep synced (X is current iterate)

        # select active set
        if use_filter:
            sel = c_all > cmax - filt_thresh
            # always keep top 600 to retain cloud
            if int(sel.sum()) < 600:
                idx = np.argpartition(-c_all, 600)[:600]
                sel = np.zeros(P, dtype=bool)
                sel[idx] = True
            I = iu0[sel]; J = iu1[sel]
            csub = c_all[sel]
            p = len(I)
        else:
            I = iu0; J = iu1
            csub = c_all
            p = P

        # build tangent bases T: (n,d,m)
        # vectorized per-point QR
        T = np.empty((n, d, m), dtype=np.float64)
        for i in range(n):
            xi = X[i]
            k = int(np.argmin(np.abs(xi)))
            # B columns = e_j - xi_j*xi for j!=k
            # Build B as (d,m)
            # fast: B = -xi[:,None]*xi_row? then set diagonal 1
            # B[a, col] where col corresponds to j
            # Use loop over m (12) - cheap
            B = np.empty((d, m), dtype=np.float64)
            col = 0
            for j in range(d):
                if j == k:
                    continue
                v = -xi * xi[j]
                v[j] += 1.0
                B[:, col] = v
                col += 1
            Q, _ = np.linalg.qr(B)
            T[i] = Q

        # coefficients CI = T_I^T X_J, CJ = T_J^T X_I
        # advanced indexing
        TI = T[I]  # (p,13,12)
        TJ = T[J]
        XI = X[I]  # (p,13)
        XJ = X[J]
        # einsum pdm,pd->pm
        CI = np.einsum('pdm,pd->pm', TI, XJ)
        CJ = np.einsum('pdm,pd->pm', TJ, XI)

        # build sparse A_ub
        # rows: each pair row has 2*m+1 nonzeros
        # construct coo arrays
        nnz = p * (2 * m + 1)
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=np.float64)
        # fill in blocks
        # block1: CI
        # rows block1: repeat each row m times
        r1 = np.repeat(np.arange(p), m)
        # cols block1
        c1 = (I[:, None] * m + ar12[None, :]).ravel()
        # block2
        r2 = np.repeat(np.arange(p), m)
        c2 = (J[:, None] * m + ar12[None, :]).ravel()
        r3 = np.arange(p)
        c3 = np.full(p, nv - 1, dtype=np.int64)
        rows[0:p*m] = r1
        cols[0:p*m] = c1
        data[0:p*m] = CI.ravel()
        rows[p*m:2*p*m] = r2
        cols[p*m:2*p*m] = c2
        data[p*m:2*p*m] = CJ.ravel()
        rows[2*p*m:2*p*m+p] = r3
        cols[2*p*m:2*p*m+p] = c3
        data[2*p*m:2*p*m+p] = -1.0
        A_ub = coo_matrix((data, (rows, cols)), shape=(p, nv)).tocsr()
        b_ub = -csub

        bounds = [(-radius, radius)] * (n * m) + [(-1.0, cur)]
        # solve LP
        t_s = time.time()
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            radius *= 0.5
            continue
        t_solve = time.time() - t_s
        # adapt filtering if too slow
        if t_solve > 0.6:
            slow_count += 1
            if slow_count >= 2:
                use_filter = True
        else:
            if use_filter and t_solve < 0.15 and p < P:
                # occasionally try full again if fast
                pass

        if res.status != 0 or res.x is None:
            radius *= 0.5
            continue
        t_pred = float(res.x[-1])
        pred = cur - t_pred
        if pred < 1e-13:
            # no predicted progress: shrink or kick
            # check tying: count ties
            slack = cur - c_all
            n10 = int((slack < 1e-10).sum())
            # if jammed, try small random move to change active set
            radius *= 0.5
            if radius < 1e-7:
                # random tangent perturbation at current radius scale
                kick = rng.standard_normal(X.shape) * 2e-4
                kick -= (np.sum(kick * X, axis=1, keepdims=True)) * X
                Xt = _norm(X + kick)
                vt = _maxcos(Xt)
                if vt < cur + 1e-6:
                    X = Xt
                    cur = vt
                    radius = 5e-4
            continue

        U = res.x[:-1].reshape(n, m)
        D = np.einsum('ndm,nm->nd', T, U)

        # exact backtracking over alpha
        alphas = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
        accepted = False
        best_alpha_val = cur
        best_alpha_X = None
        best_alpha = 0.0
        for alpha in alphas:
            Xt = _norm(X + alpha * D)
            # exact max
            Gt = Xt @ Xt.T
            np.fill_diagonal(Gt, -2.0)
            vt = float(Gt.max())
            if vt < best_alpha_val - 1e-15:
                best_alpha_val = vt
                best_alpha_X = Xt
                best_alpha = alpha
                # don't break: smaller alpha might be even better? generally larger is better if monotonic, but keep first improvement then check? we want best among alphas
        if best_alpha_X is not None and best_alpha_val < cur - 1e-15:
            ared = cur - best_alpha_val
            rho = ared / max(pred * best_alpha, 1e-18)
            # accept
            X = best_alpha_X
            cur = best_alpha_val
            accepted = True
            if cur < best_val:
                best_val = cur
                bestX = X.copy()
            # radius update
            if rho > 0.75:
                radius = min(radius * 1.5, rmax)
            elif rho > 0.25:
                pass
            else:
                # small ratio but still improved: keep radius
                pass
            # second-order correction recomputed next iter via Gram
        else:
            # no improvement
            radius *= 0.5
            continue

        # optional time check for diagnostics (counts) - no overhead
        if it % 50 == 0:
            if time.time() - t0 > DEADLINE - 5.0:
                break

    return _norm(bestX)

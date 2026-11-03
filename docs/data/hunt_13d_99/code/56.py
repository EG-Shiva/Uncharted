import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _proj_A_cor(G):
    # PSD rank<=13 + unit diagonal via truncated eig + correlation scaling
    Gs = (G + G.T) * 0.5
    w, V = np.linalg.eigh(Gs)
    # top 13
    Vt = V[:, -13:]
    wt = w[-13:]
    wt = np.maximum(wt, 0.0)
    # reconstruct
    Gp = (Vt * wt) @ Vt.T
    d = np.diag(Gp)
    d = np.maximum(d, 1e-12)
    s = 1.0 / np.sqrt(d)
    Gp = Gp * s[:, None] * s[None, :]
    Gp = (Gp + Gp.T) * 0.5
    np.fill_diagonal(Gp, 1.0)
    return Gp

def _proj_B_box(G, t):
    H = np.minimum(G, t)
    np.fill_diagonal(H, 1.0)
    return H

def _proj_B_quant(G, t, rng_vals=None):
    H = np.minimum(G, t)
    np.fill_diagonal(H, 1.0)
    # weak quantisation toward few-distance levels for entries well below t
    levels = np.array([-0.3333333, -0.15, 0.0, 0.12], dtype=float)
    q = 0.18
    # work on off-diagonal only
    n = H.shape[0]
    # mask of entries to quantise (strictly below t, not diagonal)
    # vectorised over full matrix (symmetric, apply symmetrically)
    # compute nearest level
    # to save time do it every call but n=99 so cheap
    # use upper triangle to decide then symmetrise
    # simple full-matrix version:
    # expand diff: (99,99,4)
    D = np.abs(H[:, :, None] - levels[None, None, :])
    nearest = levels[np.argmin(D, axis=2)]
    # only pull if comfortably below t
    pull_mask = (H < (t - 0.025))
    # don't touch diagonal
    np.fill_diagonal(pull_mask, False)
    H[pull_mask] = (1.0 - q) * H[pull_mask] + q * nearest[pull_mask]
    H = (H + H.T) * 0.5
    np.fill_diagonal(H, 1.0)
    # re-enforce box after pull (pull could exceed t slightly? no, levels < t)
    np.minimum(H, t, out=H)
    np.fill_diagonal(H, 1.0)
    return H

def _gram_to_X(G):
    Gs = (G + G.T) * 0.5
    w, V = np.linalg.eigh(Gs)
    Vt = V[:, -13:]
    wt = np.maximum(w[-13:], 0.0)
    sq = np.sqrt(np.maximum(wt, 0.0))
    X = Vt * sq[None, :]
    # rows may have tiny norm; fix
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    X = X / nr
    return _norm_rows(X)

def _tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    am = np.argmax(np.abs(X), axis=1)
    eye = np.eye(dm)
    for i in range(n):
        xi = X[i]
        a = int(am[i])
        mask = np.arange(d) != a
        xj = xi[mask]
        M = -xi[:, None] * xj[None, :]
        M[mask, :] += eye
        Q, _ = np.linalg.qr(M, mode='reduced')
        B[i] = Q
    return B

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    GRAM_BUDGET = 48.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    if xb is not None and np.shape(xb) == (99, 13):
        Xbest0 = _norm_rows(np.asarray(xb, float))
    elif xr is not None and np.shape(xr) == (99, 13):
        Xbest0 = _norm_rows(np.asarray(xr, float))
    else:
        rng0 = np.random.default_rng(0)
        Xbest0 = _norm_rows(rng0.standard_normal(size=(99, 13)))
    if xr is not None and np.shape(xr) == (99, 13):
        Xrec0 = _norm_rows(np.asarray(xr, float))
    else:
        Xrec0 = Xbest0.copy()
    m0 = _max_cos(Xbest0)
    best_X = Xbest0.copy()
    best_m = float(m0)
    if _max_cos(Xrec0) < best_m:
        best_X = Xrec0.copy(); best_m = _max_cos(Xrec0)

    # ---- build diverse initial X ----
    X_inits = []
    X_inits.append(Xbest0.copy())
    X_inits.append(Xrec0.copy())
    for sd in (1, 2, 3):
        rng = np.random.default_rng(1000 + sd)
        X_inits.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    # quantized lattice init {-1,0,1}
    rng = np.random.default_rng(777)
    Xq = np.zeros((99, 13))
    for i in range(99):
        while True:
            v = rng.integers(-1, 2, size=13).astype(float)
            if np.linalg.norm(v) > 1e-9:
                break
        Xq[i] = v / np.linalg.norm(v)
    # small random rotation to break exact degeneracy
    Rq, _ = np.linalg.qr(rng.standard_normal(size=(13, 13)))
    if np.linalg.det(Rq) < 0:
        Rq[:, 0] = -Rq[:, 0]
    Xq = _norm_rows(Xq @ Rq.T + 0.02 * rng.standard_normal(size=(99, 13)))
    X_inits.append(Xq)
    # cyclic C11 structured init (9 reps x 11)
    try:
        rng = np.random.default_rng(999)
        Yf = _norm_rows(rng.standard_normal(size=(9, 13)))
        th = 2 * np.pi * np.arange(11)[:, None] * np.arange(1, 6)[None, :] / 11.0
        Cc = np.cos(th); Ss = np.sin(th)
        Xc = np.zeros((99, 13))
        for g in range(11):
            idx = np.arange(9) * 11 + g
            for p in range(5):
                c = Cc[g, p]; s = Ss[g, p]
                y0 = Yf[:, 2 * p]; y1 = Yf[:, 2 * p + 1]
                Xc[idx, 2 * p] = c * y0 - s * y1
                Xc[idx, 2 * p + 1] = s * y0 + c * y1
            Xc[idx, 10:13] = Yf[:, 10:13]
        Xc = _norm_rows(Xc + 0.01 * rng.standard_normal(size=(99, 13)))
        X_inits.append(Xc)
    except Exception:
        pass
    # antipodal-ish random init
    rng = np.random.default_rng(2024)
    Xa = _norm_rows(rng.standard_normal(size=(99, 13)))
    X_inits.append(Xa)

    G_inits = [np.asarray(X @ X.T, dtype=float) for X in X_inits]
    for G in G_inits:
        np.fill_diagonal(G, 1.0)
        G[:] = (G + G.T) * 0.5

    targets = [0.2445, 0.2435, 0.2425, 0.2410]
    cand_X = []
    # quick Gram alternating projections
    # per (t,init) iters adapt to remaining budget
    n_init = len(G_inits)
    total_runs = len(targets) * n_init
    # iters per run ~ 450 (tunable)
    iters_per = 450
    for ti, t in enumerate(targets):
        for gi, G0 in enumerate(G_inits):
            if time.time() - t0 > GRAM_BUDGET:
                break
            use_dr = (gi % 2 == 0)
            use_q = (gi % 3 == 0)
            Z = G0.copy()
            best_mA = 1e9
            best_A = None
            beta = 1.0
            # local copies for speed
            for k in range(iters_per):
                if not use_dr:
                    A = _proj_A_cor(Z)
                    if use_q:
                        B = _proj_B_quant(A, t)
                    else:
                        B = _proj_B_box(A, t)
                    # max off-diag of A
                    # cheap max: use triu
                    mA = float(np.max(A - np.eye(99) * 2.0))
                    if mA < best_mA:
                        best_mA = mA
                        best_A = A.copy()
                        if best_mA <= t + 1e-9:
                            # feasible-ish, keep iterating a bit to diversify? break early
                            if k > 150:
                                Z = B
                                break
                    Z = B  # beta=1
                else:
                    A = _proj_A_cor(Z)
                    R = 2.0 * A - Z
                    R = (R + R.T) * 0.5
                    np.fill_diagonal(R, 1.0)
                    if use_q:
                        B = _proj_B_quant(R, t)
                    else:
                        B = _proj_B_box(R, t)
                    mA = float(np.max(A - np.eye(99) * 2.0))
                    if mA < best_mA:
                        best_mA = mA
                        best_A = A.copy()
                        if best_mA <= t + 1e-9 and k > 150:
                            Z = Z + (B - A)
                            break
                    Z = Z + (B - A)
                    # stabilise: keep symmetric, diag 1
                    Z = (Z + Z.T) * 0.5
                    np.fill_diagonal(Z, 1.0)
                    # damp runaway
                    # clip extreme values
                    np.clip(Z, -1.0, 1.0, out=Z)
                    np.fill_diagonal(Z, 1.0)
                # time check inside tight loop occasionally
                if (k & 127) == 127 and time.time() - t0 > GRAM_BUDGET:
                    break
            if best_A is not None:
                try:
                    Xc = _gram_to_X(best_A)
                    cand_X.append(Xc)
                except Exception:
                    pass
        if time.time() - t0 > GRAM_BUDGET:
            break

    # evaluate candidates, keep top few + incumbent
    scored = []
    for Xc in cand_X:
        try:
            m = _max_cos(_norm_rows(Xc))
            scored.append((m, Xc))
        except Exception:
            pass
    scored.append((best_m, best_X.copy()))
    scored.sort(key=lambda z: z[0])
    # deduplicate-ish: keep up to 6 best distinct
    keep = []
    for m, Xc in scored:
        Xn = _norm_rows(Xc)
        # check distinct from kept (Frobenius after alignment is expensive; just check max-cos diff + norm diff)
        dup = False
        for _, Yk in keep:
            if np.linalg.norm(Xn - Yk) < 1e-6 or np.linalg.norm(Xn + Yk) < 1e-6:
                dup = True
                break
        if not dup:
            keep.append((m, Xn))
        if len(keep) >= 6:
            break
    # update global best from gram phase (actual renormalised values)
    for m, Xn in keep:
        if m < best_m:
            best_m = m
            best_X = Xn.copy()

    # ---- annealed LogSumExp L-BFGS polish ----
    def make_fun(alpha):
        def fun(vf):
            V = vf.reshape(99, 13)
            nr = np.sqrt(np.sum(V * V, axis=1, keepdims=True))
            nr = np.maximum(nr, 1e-12)
            Y = V / nr
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            E = np.exp(alpha * (G - m))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E) * 0.5) + 1e-300
            L = m + np.log(S) / alpha
            GY = E @ Y
            gY = GY / S
            dot = np.sum(gY * Y, axis=1, keepdims=True)
            gV = (gY - dot * Y) / nr
            return float(L), gV.ravel().astype(float)
        return fun

    alphas = [120.0, 400.0, 1200.0, 3000.0]
    # budget LBFGS: iterate over kept in order of promise
    for ki, (m_init, Xk) in enumerate(list(keep)):
        if time.time() - t0 > DEADLINE - 25.0:
            break
        cur = _norm_rows(Xk.copy())
        cur_m = _max_cos(cur)
        # fewer iters for worse candidates
        maxiter_list = [120, 120, 150, 150] if ki < 2 else [80, 80, 80, 80]
        for ai, alpha in enumerate(alphas):
            if time.time() - t0 > DEADLINE - 25.0:
                break
            fun = make_fun(alpha)
            v0 = cur.ravel().astype(float).copy()
            try:
                r = minimize(fun, v0, jac=True, method='L-BFGS-B',
                             options={'maxiter': maxiter_list[ai], 'ftol': 1e-20, 'gtol': 1e-9,
                                      'maxfun': maxiter_list[ai] * 6, 'maxcor': 12})
                Vopt = r.x.reshape(99, 13)
                Yopt = _norm_rows(Vopt)
                m_new = _max_cos(Yopt)
                # accept if not catastrophically worse (allow uphill on smooth proxy only if max not exploding)
                if m_new < cur_m + 0.002:
                    cur = Yopt
                    cur_m = m_new
                if m_new < best_m:
                    best_m = m_new
                    best_X = Yopt.copy()
            except Exception:
                continue
        # store back
        keep[ki] = (cur_m, cur)

    # ---- final tangent LP polish on global best (wide active set, few steps) ----
    try:
        X = _norm_rows(best_X.copy())
        m_cur = _max_cos(X)
        n, d = 99, 13
        NVAR = n * (d - 1)
        NCOL = NVAR + 1
        iu0, iu1 = np.triu_indices(n, 1)
        ar12 = np.arange(d - 1)
        r = 8e-5
        r_min = 1e-9
        for it in range(40):
            if time.time() - t0 > DEADLINE - 3.0:
                break
            if r < r_min:
                break
            G = X @ X.T
            c_all = G[iu0, iu1]
            m_cur = float(np.max(c_all))
            # keep tightest ~900
            need = 900
            order = np.argpartition(-c_all, min(need, c_all.size - 1))
            sel_idx = order[:need]
            I = iu0[sel_idx].astype(np.int64)
            J = iu1[sel_idx].astype(np.int64)
            c_sel = c_all[sel_idx]
            p = need
            B = _tangent_bases(X)
            XI = X[I]; XJ = X[J]
            BI = B[I]; BJ = B[J]
            aI = np.einsum('pi,pij->pj', XJ, BI)
            aJ = np.einsum('pi,pij->pj', XI, BJ)
            cols_mat = np.empty((p, 25), dtype=np.int64)
            cols_mat[:, :12] = I[:, None] * 12 + ar12
            cols_mat[:, 12:24] = J[:, None] * 12 + ar12
            cols_mat[:, 24] = NVAR
            vals_mat = np.empty((p, 25), dtype=float)
            vals_mat[:, :12] = aI
            vals_mat[:, 12:24] = aJ
            vals_mat[:, 24] = -1.0
            rows_mat = np.repeat(np.arange(p), 25)
            A_ub = coo_matrix((vals_mat.ravel(), (rows_mat, cols_mat.ravel())), shape=(p, NCOL)).tocsr()
            b_ub = -c_sel
            obj = np.zeros(NCOL); obj[-1] = 1.0
            bounds = [(-r, r)] * NVAR + [(None, m_cur)]
            remain = DEADLINE - 3.0 - (time.time() - t0)
            tl = max(1.0, min(6.0, remain / 4.0))
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                              options={"presolve": True, "time_limit": tl})
            except Exception:
                r *= 0.5
                continue
            if res.status != 0 or res.x is None:
                r *= 0.5
                continue
            t_pred = float(res.x[-1])
            if not np.isfinite(t_pred) or t_pred >= m_cur - 1e-12:
                r *= 0.5
                continue
            dvec = np.asarray(res.x[:-1], dtype=float)
            if not np.all(np.isfinite(dvec)):
                r *= 0.5
                continue
            D = dvec.reshape(n, d - 1)
            DX = np.einsum('nij,nj->ni', B, D)
            accepted = False
            for alpha_s in (1.0, 0.5, 0.25):
                Y = _norm_rows(X + alpha_s * DX)
                m_new = _max_cos(Y)
                if m_new < m_cur - 1e-12:
                    X = Y; m_cur = m_new
                    if m_new < best_m:
                        best_m = m_new; best_X = Y.copy()
                    r = max(r * 0.85, r_min * 0.5)
                    accepted = True
                    break
            if not accepted:
                r *= 0.5
                continue
    except Exception:
        pass

    return _norm_rows(np.asarray(best_X, dtype=float))

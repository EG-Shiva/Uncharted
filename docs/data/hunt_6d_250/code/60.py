import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _build_Q(X):
    # vectorised tangent bases via projected random frame + inverse sqrt, einsum only
    n, d = X.shape
    k = d - 1
    rng = np.random.default_rng(0)
    R = rng.standard_normal((d, k))
    S = X @ R  # n x k
    # Braw[n,d,k] = R - X * (X^T R)
    Braw = R[None, :, :] - X[:, :, None] * S[:, None, :]
    Gram = np.einsum('ndk,ndl->nkl', Braw, Braw)
    Gram = Gram + np.eye(k)[None, :, :] * 1e-12
    w, V = np.linalg.eigh(Gram)
    w = np.maximum(w, 1e-12)
    inv = 1.0 / np.sqrt(w)
    # inv_sqrt = V diag(inv) V^T
    inv_sqrt = np.einsum('nkl,nl,nml->nkm', V, inv, V)
    Q = np.einsum('ndl,nlm->ndm', Braw, inv_sqrt)
    return Q

def _smooth_opt(X0, beta, maxiter, maxcor=8):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.exp(D, out=D)
        np.fill_diagonal(E, 0.0)
        Spair = float(np.sum(E) * 0.5)
        if Spair <= 0 or not np.isfinite(Spair):
            Spair = 1e-300
        loss = M + np.log(Spair) / beta
        GX = (E @ X) / Spair
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        if not np.all(np.isfinite(GY)):
            GY = np.nan_to_num(GY, nan=0.0, posinf=0.0, neginf=0.0)
        return float(loss), GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter * 6,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    return _norm_rows(Y)

def _dual_stress(X, Q, I, J, c):
    n, d = X.shape
    k = d - 1
    P = I.shape[0]
    if P == 0:
        return None, None
    QI = Q[I]  # P x d x k
    QJ = Q[J]
    XI = X[I]
    XJ = X[J]
    VI = np.einsum('pdk,pd->pk', QI, XJ)
    VJ = np.einsum('pdk,pd->pk', QJ, XI)
    ar = np.arange(k)
    # A_eq: (n*k+1) x (P+1), vectorised COO
    rows_I = (I[:, None] * k + ar[None, :]).ravel()
    cols_I = np.repeat(np.arange(P), k)
    data_I = VI.ravel()
    rows_J = (J[:, None] * k + ar[None, :]).ravel()
    cols_J = np.repeat(np.arange(P), k)
    data_J = VJ.ravel()
    rows_sum = np.full(P, n * k, dtype=np.int64)
    cols_sum = np.arange(P, dtype=np.int64)
    data_sum = np.ones(P)
    rows_eq = np.concatenate([rows_I, rows_J, rows_sum])
    cols_eq = np.concatenate([cols_I, cols_J, cols_sum])
    data_eq = np.concatenate([data_I, data_J, data_sum])
    try:
        A_eq = coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(n * k + 1, P + 1)).tocsr()
    except Exception:
        return None, None
    b_eq = np.zeros(n * k + 1)
    b_eq[-1] = 1.0
    # A_ub: -w + t <= 0
    rows_ub = np.repeat(np.arange(P), 2)
    cols_ub = np.empty(2 * P, dtype=np.int64)
    cols_ub[0::2] = np.arange(P)
    cols_ub[1::2] = P
    data_ub = np.empty(2 * P)
    data_ub[0::2] = -1.0
    data_ub[1::2] = 1.0
    try:
        A_ub = coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(P, P + 1)).tocsr()
    except Exception:
        return None, None
    b_ub = np.zeros(P)
    cc = np.zeros(P + 1)
    cc[-1] = -1.0
    bounds = [(0.0, None)] * P + [(None, None)]
    try:
        res = linprog(cc, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception:
        return None, None
    if res.status != 0 or res.x is None:
        return None, None
    t = float(res.x[-1])
    w = np.asarray(res.x[:-1]).ravel()
    return t, w

def _primal_step(X, Q, I, J, c, radius):
    n, d = X.shape
    k = d - 1
    P = I.shape[0]
    if P == 0:
        return None, None
    QI = Q[I]
    QJ = Q[J]
    VI = np.einsum('pdk,pd->pk', QI, X[J])
    VJ = np.einsum('pdk,pd->pk', QJ, X[I])
    ar = np.arange(k)
    # build P x (2k+1) mats then ravel -> vectorised COO, no pair loop
    cols_I = (I[:, None] * k + ar[None, :])
    cols_J = (J[:, None] * k + ar[None, :])
    cols_T = np.full((P, 1), n * k, dtype=np.int64)
    ColsMat = np.concatenate([cols_I, cols_J, cols_T], axis=1)
    DataMat = np.concatenate([VI, VJ, -np.ones((P, 1))], axis=1)
    rows_pr = np.repeat(np.arange(P), 2 * k + 1)
    cols_pr = ColsMat.ravel()
    data_pr = DataMat.ravel()
    nv = n * k + 1
    try:
        A_ub = coo_matrix((data_pr, (rows_pr, cols_pr)), shape=(P, nv))
        A_ub = coo_matrix((data_pr, (rows_pr, cols_pr)), shape=(P, nv)).tocsr()
    except Exception:
        return None, None
    b_ub = -c
    cc = np.zeros(nv)
    cc[-1] = 1.0
    Mcur = float(np.max(c))
    bounds = [(-radius, radius)] * (n * k) + [(None, float(Mcur))]
    try:
        res = linprog(cc, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    except Exception:
        return None, None
    if res.status != 0 or res.x is None:
        return None, None
    u = np.asarray(res.x[:-1]).reshape(n, k)
    t = float(res.x[-1])
    return u, t

def _apply_u(X, Q, u, alpha):
    D = np.einsum('ndk,nk->nd', Q, u)
    Y = X + alpha * D
    return _norm_rows(Y)

def solve():
    t0 = time.time()
    DEADLINE = 168.0
    def rem():
        return DEADLINE - (time.time() - t0)
    n, d = 250, 6
    k = d - 1
    rng = np.random.default_rng(777)
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm_rows(z)
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best_val:
                        best_val = v
                        best = z.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        best = _norm_rows(z)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = float(best_val)
    iu0 = np.triu_indices(n, 1)[0]
    iu1 = np.triu_indices(n, 1)[1]

    # ---- exact stress-certification on M-1e-6 shell ----
    lp_deadline = t0 + 150.0
    try:
        G = cur @ cur.T
        gv = G[iu0, iu1]
        M = float(np.max(gv))
        thr = M - 1e-6
        mask = gv >= thr
        I = iu0[mask].astype(np.int64)
        J = iu1[mask].astype(np.int64)
        c = gv[mask].astype(np.float64)
        # cap to 4000 tightest if huge (vectorised top-k)
        if I.shape[0] > 4000:
            order = np.argpartition(-c, 4000)[:4000]
            I = I[order]; J = J[order]; c = c[order]
        P0 = I.shape[0]
        Q = _build_Q(cur)
        t_star, w = None, None
        if rem() > 20 and time.time() < lp_deadline and P0 > 0:
            t_star, w = _dual_stress(cur, Q, I, J, c)
        blocked = False
        if t_star is not None and np.isfinite(t_star) and t_star > 1e-9:
            blocked = True
        if blocked:
            # certify blocked: delete lowest-weight 8% of shell contacts
            try:
                frac = 0.08
                nd = int(max(1, int(frac * P0)))
                order = np.argsort(w)
                drop = order[:nd]
                keep_mask = np.ones(P0, dtype=bool)
                keep_mask[drop] = False
                Ik = I[keep_mask]; Jk = J[keep_mask]; ck = c[keep_mask]
                # deliberate basin change: tangent jitter biased to weak contacts
                deg_weak = np.zeros(n)
                np.add.at(deg_weak, I[drop], 1.0)
                np.add.at(deg_weak, J[drop], 1.0)
                amp = 0.0025
                Z = rng.standard_normal((n, d))
                Z = Z - np.sum(Z * cur, axis=1, keepdims=True) * cur
                sc = (1.0 + deg_weak / max(1.0, float(np.max(deg_weak))))[:, None]
                Xp = _norm_rows(cur + amp * sc * Z)
                vp = _max_cos(Xp)
                # polish perturbed point even if slightly worse (basin hop)
                Xh = Xp
                for beta, mi in ((6000.0, 120), (18000.0, 140)):
                    if rem() < 12:
                        break
                    Xh = _smooth_opt(Xh, beta, maxiter=mi)
                vh = _max_cos(Xh)
                if vh < cur_val:
                    cur_val = vh; cur = Xh.copy()
                    Q = _build_Q(cur)
                    # refresh shell after success
                    G = cur @ cur.T; gv = G[iu0, iu1]; M = float(np.max(gv))
                    thr = M - 1e-6
                    mask = gv >= thr
                    I = iu0[mask].astype(np.int64); J = iu1[mask].astype(np.int64)
                    c = gv[mask].astype(np.float64)
                    if I.shape[0] > 4000:
                        od = np.argpartition(-c, 4000)[:4000]
                        I = I[od]; J = J[od]; c = c[od]
                else:
                    # try primal on reduced set to force topology change
                    Qp = _build_Q(Xh)
                    G2 = Xh @ Xh.T; gv2 = G2[iu0, iu1]; M2 = float(np.max(gv2))
                    thr2 = M2 - 1e-6
                    m2 = gv2 >= thr2
                    I2 = iu0[m2].astype(np.int64); J2 = iu1[m2].astype(np.int64)
                    c2 = gv2[m2].astype(np.float64)
                    if I2.shape[0] > 4000:
                        od = np.argpartition(-c2, 4000)[:4000]
                        I2 = I2[od]; J2 = J2[od]; c2 = c2[od]
                    for rad in (1e-4, 3e-5):
                        if rem() < 10:
                            break
                        pr = _primal_step(Xh, Qp, I2, J2, c2, rad)
                        if pr[0] is None:
                            continue
                        u, tp = pr
                        for a in (1.0, 0.5, 0.25):
                            Y = _apply_u(Xh, Qp, u, a)
                            v = _max_cos(Y)
                            if v < cur_val:
                                cur_val = v; cur = Y.copy()
                                break
                # keep track of kept set for later equalisation (store)
            except Exception:
                pass
        else:
            # unblocked or infeasible: primal descent ladder + exact line search
            try:
                ladder = (1e-4, 3e-5, 1e-5, 3e-6)
                for rad in ladder:
                    if rem() < 12 or time.time() > lp_deadline:
                        break
                    # refresh Q and shell each round
                    Qc = _build_Q(cur)
                    Gc = cur @ cur.T
                    gvc = Gc[iu0, iu1]
                    Mc = float(np.max(gvc))
                    thr = Mc - 1e-6
                    m = gvc >= thr
                    Ic = iu0[m].astype(np.int64); Jc = iu1[m].astype(np.int64)
                    cc_ = gvc[m].astype(np.float64)
                    if Ic.shape[0] == 0:
                        break
                    if Ic.shape[0] > 4000:
                        od = np.argpartition(-cc_, 4000)[:4000]
                        Ic = Ic[od]; Jc = Jc[od]; cc_ = cc_[od]
                    pr = _primal_step(cur, Qc, Ic, Jc, cc_, rad)
                    if pr[0] is None:
                        continue
                    u, tp = pr
                    # exact line search over alpha ladder
                    improved = False
                    for a in (1.0, 0.7, 0.4, 0.2, 0.1):
                        Y = _apply_u(cur, Qc, u, a)
                        v = _max_cos(Y)
                        if v < cur_val - 1e-12:
                            cur_val = v; cur = Y.copy()
                            improved = True
                            break
                    # if no improvement, continue to smaller radius
            except Exception:
                pass
    except Exception:
        pass

    # ---- annealed smoothing polish ----
    try:
        if rem() > 12:
            X = cur.copy()
            for beta, mi in ((5000.0, 150), (15000.0, 180), (35000.0, 180)):
                if rem() < 8:
                    break
                X = _smooth_opt(X, beta, maxiter=mi)
                v = _max_cos(X)
                if v < cur_val:
                    cur_val = v; cur = X.copy()
    except Exception:
        pass

    # ---- short hopping with hole regrowth if time remains ----
    try:
        hop = 0
        while rem() > 20 and hop < 12:
            hop += 1
            G = cur @ cur.T
            gv = G[iu0, iu1]
            M = float(np.max(gv))
            deg = np.zeros(n)
            sel = gv >= M - 2e-6
            Is = iu0[sel]; Js = iu1[sel]
            np.add.at(deg, Is, 1); np.add.at(deg, Js, 1)
            # excise 5 random high-degree points and regrow by farthest sampling
            order = np.argsort(-deg)
            pool = order[:40].tolist()
            rng.shuffle(pool)
            chosen = pool[:5]
            Xw = cur.copy()
            for idx in chosen:
                C = rng.standard_normal((2000, d))
                C = _norm_rows(C)
                dots = C @ Xw.T
                dots[:, idx] = -2.0
                mrow = np.max(dots, axis=1)
                Xw[idx] = C[int(np.argmin(mrow))]
            Xw = _norm_rows(Xw)
            Xh = _smooth_opt(Xw, 8000.0, maxiter=100)
            Xh = _smooth_opt(Xh, 20000.0, maxiter=120)
            vh = _max_cos(Xh)
            if vh < cur_val:
                cur_val = vh; cur = Xh.copy()
                if rem() < 25:
                    break
    except Exception:
        pass

    # ---- final intensive polish ----
    try:
        X = cur.copy()
        for beta, mi in ((8000.0, 180), (25000.0, 200)):
            if rem() < 6:
                break
            Xn = _smooth_opt(X, beta, maxiter=mi)
            v = _max_cos(Xn)
            if v < cur_val:
                cur_val = v; cur = Xn.copy()
            X = Xn
    except Exception:
        pass

    out = _norm_rows(cur.copy())
    # guard against NaN / zero rows
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    out = out / nr
    return out

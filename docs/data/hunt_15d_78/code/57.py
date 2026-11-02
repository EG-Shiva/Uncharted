import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix
try:
    from scipy.linalg import expm
    _HAS_EXPM = True
except Exception:
    _HAS_EXPM = False

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, deadline, r_init=0.008, r_max=0.03, cap=2500, max_it=60):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bx = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    it = 0
    while time.time() + 0.4 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        if int(np.count_nonzero(sel)) == 0:
            radius *= 0.5
            continue
        if int(np.count_nonzero(sel)) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; bx = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return bx, best

def _skew_to_R(s):
    M = np.zeros((6, 6))
    k = 0
    for a in range(6):
        for b in range(a + 1, 6):
            M[a, b] = s[k]; M[b, a] = -s[k]; k += 1
    if _HAS_EXPM:
        try:
            return expm(M)
        except Exception:
            pass
    # Cayley fallback
    try:
        I = np.eye(6)
        return np.linalg.solve(I - M, I + M)
    except Exception:
        return I + M

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    INTERNAL = t0 + 160.0
    n, d = 78, 15
    # load incumbent
    cand = []
    try:
        b = np.load("best.npy")
        if b.shape == (n, d):
            cand.append(b.astype(float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        if r.shape == (n, d):
            cand.append(r.astype(float))
    except Exception:
        pass
    if len(cand) == 0:
        rng0 = np.random.default_rng(0)
        x0 = rng0.standard_normal((n, d)); x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    X_best = None; v_best = 1e9
    for c in cand:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < v_best:
            v_best = v; X_best = c.copy()
    X_inc = X_best.copy()
    # PCA basis from centered cloud
    try:
        mu = X_inc.mean(axis=0, keepdims=True)
        Xc = X_inc - mu
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt.T  # 15x15 orthogonal
    except Exception:
        V, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))
    Y_base = X_inc @ V  # rotated record
    pc1 = (Xc @ V[:, 0]) if 'Xc' in locals() else Y_base[:, 0]
    order = np.argsort(pc1)  # ascending: low dense vs high sparse
    spc1 = pc1[order]
    diffs = np.diff(spc1)
    # largest gap in middle 40..90% to define original strata
    lo, hi = int(n * 0.4), int(n * 0.9)
    seg = diffs[lo:hi]
    if len(seg) > 0:
        gap_rel = int(np.argmax(seg))
        nA0 = lo + gap_rel + 1
    else:
        nA0 = 58
    nA0 = int(np.clip(nA0, 40, 70))
    orig_low_set = set(order[:nA0].tolist())
    muA0 = float(Y_base[order[:nA0], 0].mean())
    muB0 = float(Y_base[order[nA0:], 0].mean())
    rest_norm = np.linalg.norm(Y_base[:, 1:], axis=1)
    rest_norm = np.maximum(rest_norm, 1e-12)
    iu0, iu1 = np.triu_indices(n, 1)

    targets = [65, 61, 55, 52, 49]
    # include original size once
    if nA0 not in targets:
        targets = [nA0] + targets
    # dedup
    seen = set(); tlist = []
    for t in targets:
        if t not in seen and 30 < t < 75:
            seen.add(t); tlist.append(t)
    targets = tlist[:6]

    def build_X(muA, muB, R, low_mask):
        # vectorized two-level construction
        Y = np.empty_like(Y_base)
        mus = np.where(low_mask, muA, muB)
        # clip for sqrt
        mus_c = np.clip(mus, -0.95, 0.95)
        scl = np.sqrt(np.maximum(0.0, 1.0 - mus_c ** 2)) / rest_norm
        Y[:, 0] = mus_c
        Y[:, 1:] = Y_base[:, 1:] * scl[:, None]
        # relative rotation on high stratum block 1:7
        if R is not None:
            hb = Y[~low_mask][:, 1:7]
            if hb.shape[0] > 0:
                Y[~low_mask][:, 1:7] = hb @ R.T
        X = Y @ V.T
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def lse_of_X(X, beta):
        G = X @ X.T
        gv = G[iu0, iu1]
        m = float(gv.max())
        e = np.exp(beta * (gv - m))
        s = float(e.sum())
        return m + np.log(s) / beta, m

    for nA in targets:
        if time.time() > INTERNAL - 30:
            break
        low_mask = np.zeros(n, dtype=bool)
        low_mask[order[:nA]] = True
        new_low_set = set(order[:nA].tolist())
        moved = np.array(sorted(list((new_low_set ^ orig_low_set))), dtype=int) if len(new_low_set ^ orig_low_set) > 0 else np.array([], dtype=int)
        # initial mus: use original stratum means
        muA, muB = muA0, muB0
        s_vec = np.zeros(15)
        # low-dim optimisation: numerical Adam on [muA,muB,s] with beta schedule
        deadline_ld = min(time.time() + 8.0, INTERNAL - 25)
        # quick check initial true max
        R0 = np.eye(6)
        try:
            X0 = build_X(muA, muB, R0, low_mask)
            v0 = _max_cos(X0)
        except Exception:
            continue
        # Adam state for 17 params
        p = np.concatenate([[muA, muB], s_vec]).astype(float)
        m1 = np.zeros_like(p); m2 = np.zeros_like(p)
        step_cnt = 0
        betas = [100, 300, 900]
        iters_per = [40, 30, 20]
        lr0 = 0.02
        eps_fd = 1e-4
        for bi, beta in enumerate(betas):
            nit = iters_per[bi]
            for it in range(nit):
                if time.time() > deadline_ld:
                    break
                step_cnt += 1
                # eval base
                try:
                    Rcur = _skew_to_R(p[2:])
                    Xcur = build_X(float(p[0]), float(p[1]), Rcur, low_mask)
                    L0, _ = lse_of_X(Xcur, beta)
                except Exception:
                    break
                if not np.isfinite(L0):
                    break
                grad = np.zeros_like(p)
                # numerical grad
                for j in range(len(p)):
                    pp = p.copy(); pp[j] += eps_fd
                    try:
                        Rj = _skew_to_R(pp[2:])
                        Xj = build_X(float(pp[0]), float(pp[1]), Rj, low_mask)
                        Lj, _ = lse_of_X(Xj, beta)
                        grad[j] = (Lj - L0) / eps_fd
                    except Exception:
                        grad[j] = 0.0
                # clip grad
                grad = np.clip(grad, -5.0, 5.0)
                b1, b2, eps = 0.9, 0.999, 1e-8
                m1 = b1 * m1 + (1 - b1) * grad
                m2 = b2 * m2 + (1 - b2) * (grad ** 2)
                m1h = m1 / (1 - b1 ** step_cnt)
                m2h = m2 / (1 - b2 ** step_cnt)
                lr = lr0 * (0.7 ** bi)
                p = p - lr * m1h / (np.sqrt(m2h) + eps)
                p[0] = float(np.clip(p[0], -0.95, 0.95))
                p[1] = float(np.clip(p[1], -0.95, 0.95))
                p[2:] = np.clip(p[2:], -0.4, 0.4)
            if time.time() > deadline_ld:
                break
        muA, muB = float(p[0]), float(p[1])
        Rcur = _skew_to_R(p[2:])
        try:
            Xcur = build_X(muA, muB, Rcur, low_mask)
        except Exception:
            continue
        # tangential Adam for moved points only
        if moved.size > 0 and time.time() < deadline_ld:
            beta_m = 300
            mM = np.zeros((moved.size, d)); vM = np.zeros((moved.size, d))
            lr_m = 0.01
            cnt_m = 0
            nsteps = 120
            for st in range(nsteps):
                if time.time() > deadline_ld:
                    break
                cnt_m += 1
                G = Xcur @ Xcur.T
                gv = G[iu0, iu1]
                mx = float(gv.max())
                e = np.exp(beta_m * (gv - mx))
                Ssum = float(e.sum())
                w = e / Ssum
                W = np.zeros((n, n)); W[iu0, iu1] = w; W = W + W.T
                Gr = W @ Xcur
                dot = np.sum(Gr * Xcur, axis=1)
                Gt = Gr - dot[:, None] * Xcur
                gmov = Gt[moved]
                b1, b2, eps = 0.9, 0.999, 1e-8
                mM = b1 * mM + (1 - b1) * gmov
                vM = b2 * vM + (1 - b2) * (gmov ** 2)
                mMh = mM / (1 - b1 ** cnt_m); vMh = vM / (1 - b2 ** cnt_m)
                upd = lr_m * mMh / (np.sqrt(vMh) + eps)
                # project update to tangent
                for k, idx in enumerate(moved):
                    u = upd[k]
                    u = u - np.dot(u, Xcur[idx]) * Xcur[idx]
                    # cap step
                    nu = np.linalg.norm(u)
                    if nu > 0.02:
                        u *= 0.02 / nu
                    Xcur[idx] = Xcur[idx] - u
                Xcur[moved] /= np.linalg.norm(Xcur[moved], axis=1, keepdims=True)
        vcur = _max_cos(Xcur)
        # only candidates below 0.175 get full-vector L-BFGS smoothing
        if vcur < 0.175 and time.time() < INTERNAL - 20:
            beta_f = 300
            y0 = Xcur.ravel().copy()
            def fun(yf):
                Y = yf.reshape(n, d)
                rn = np.linalg.norm(Y, axis=1, keepdims=True)
                rn = np.maximum(rn, 1e-12)
                X = Y / rn
                G = X @ X.T
                gv = G[iu0, iu1]
                mx = float(gv.max())
                e = np.exp(beta_f * (gv - mx))
                Ssum = float(e.sum())
                L = mx + np.log(Ssum) / beta_f
                w = e / Ssum
                W = np.zeros((n, n)); W[iu0, iu1] = w; W = W + W.T
                Gr = W @ X
                dot = np.sum(Gr * X, axis=1)
                Da = (Gr - dot[:, None] * X) / rn
                return float(L), Da.ravel().astype(float)
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": 120, "ftol": 1e-12, "gtol": 1e-10, "maxfun": 150})
                Yf = res.x.reshape(n, d)
                Yf /= np.linalg.norm(Yf, axis=1, keepdims=True)
                vf = _max_cos(Yf)
                if vf < vcur and np.all(np.isfinite(Yf)):
                    Xcur = Yf; vcur = vf
            except Exception:
                pass
        # only candidates below incumbent get LP polish (capped 15s)
        if vcur < v_best - 1e-13 and time.time() < INTERNAL - 5:
            dl = min(time.time() + 15.0, INTERNAL - 1.0, HARD_END - 0.5)
            try:
                Xp, vp = _lp_polish(Xcur, dl, r_init=0.008, r_max=0.03)
                if vp < vcur:
                    Xcur, vcur = Xp, vp
            except Exception:
                pass
        if vcur < v_best - 1e-13 and np.all(np.isfinite(Xcur)):
            X_best = Xcur.copy(); v_best = vcur
    # final light polish of best if time remains
    if time.time() < HARD_END - 2 and X_best is not None:
        try:
            dl = min(time.time() + 6.0, HARD_END - 0.4)
            Xp, vp = _lp_polish(X_best, dl, r_init=0.002, r_max=0.015)
            if vp < v_best:
                X_best, v_best = Xp, vp
        except Exception:
            pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

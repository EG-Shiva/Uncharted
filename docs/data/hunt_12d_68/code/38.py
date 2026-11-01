import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish_full(x, deadline, r0=8e-4, cap_p=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def _full_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_full(X0, betas=(100.0, 300.0, 600.0), maxiter=120):
    n, d = X0.shape
    IU = np.triu_indices(n, 1)
    y = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    for b in betas:
        fg = _full_fg(n, d, IU, b)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

# ---- 4x17 cyclic symmetry helpers ----
def _build_CS(mult):
    # mult: (6,) ints mod 17
    K = 17
    C = np.empty((6, K)); S = np.empty((6, K))
    for i, m in enumerate(mult):
        for k in range(K):
            th = 2.0 * np.pi * float(m) * float(k) / 17.0
            C[i, k] = np.cos(th); S[i, k] = np.sin(th)
    return C, S

def _expand_sym(Sn, C, S_):
    # Sn (4,12) unit rows; C,S_ (6,17); -> X (68,12)
    S6 = Sn.reshape(4, 6, 2)
    S0 = S6[:, :, 0]  # (4,6)
    S1 = S6[:, :, 1]
    CT = C.T  # (17,6)
    ST = S_.T
    X0 = S0[:, None, :] * CT[None, :, :] - S1[:, None, :] * ST[None, :, :]
    X1 = S0[:, None, :] * ST[None, :, :] + S1[:, None, :] * CT[None, :, :]
    X6 = np.empty((4, 17, 6, 2))
    X6[:, :, :, 0] = X0
    X6[:, :, :, 1] = X1
    return X6.reshape(68, 12)

def _make_sym_fg(mult, C, S_, beta, IU):
    def fg(y_raw):
        y_raw = np.asarray(y_raw, dtype=np.float64)
        Y = y_raw.reshape(4, 12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Sn = Y / nr
        X = _expand_sym(Sn, C, S_)
        g = (X @ X.T)[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        Sm = float(np.sum(e)) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X  # (68,12)
        G6 = GX.reshape(4, 17, 6, 2)
        G0 = G6[:, :, :, 0]  # (4,17,6)
        G1 = G6[:, :, :, 1]
        CT = C.T  # (17,6)
        ST = S_.T
        # dS0[j,i] = sum_k C[i,k] G0 + S[i,k] G1
        # broadcast: C-> (1,17,6)
        Cb = C.T[None, :, :]  # (1,17,6)
        Sb = S_.T[None, :, :]
        dS0 = np.sum(Cb * G0 + Sb * G1, axis=1)  # (4,6)
        dS1 = np.sum(-Sb * G0 + Cb * G1, axis=1)
        dS6 = np.empty((4, 6, 2))
        dS6[:, :, 0] = dS0; dS6[:, :, 1] = dS1
        dS = dS6.reshape(4, 12)
        proj = np.sum(dS * Sn, axis=1, keepdims=True)
        GY = (dS - proj * Sn) / nr
        return F, GY.ravel()
    return fg

def _opt_sym(y0, mult, C, S_, betas=(100.0, 300.0, 1000.0, 3000.0), maxiter=150):
    y = y0.copy()
    for b in betas:
        IU = np.triu_indices(68, 1)
        fg = _make_sym_fg(mult, C, S_, b, IU)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(4, 12)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    X = _expand_sym(Y, C, S_)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, Y, y

def _sym_lp_polish(Seeds, mult, C, S_, deadline, r0=5e-4, cap_p=2000):
    Scur = Seeds.copy()
    Scur /= np.linalg.norm(Scur, axis=1, keepdims=True)
    X = _expand_sym(Scur, C, S_)
    best = _max_cos(X)
    radius = r0
    n = 68
    iu = np.triu_indices(n, 1)
    # map flat idx -> (j,k)
    Jof = np.repeat(np.arange(4), 17)
    Kof = np.tile(np.arange(17), 4)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        # Build A_ub: vars d (48) + t(1); a1 = R_{k1}^T X2, a2 = R_{k2}^T X1
        # Compute R^T X for needed pairs: need function apply RT
        # Precompute RTx for all points? For each flat idx q with (j,k), RT applied to vector v:
        # (RT v)[plane i] = [c*g0+s*g1, -s*g0+c*g1] with c,s = C[i,k],S_[i,k]
        # So for pair (I,J): a1 = RT_{k1}(X_J), a2 = RT_{k2}(X_I)
        try:
            rows_list = []
            cols_list = []
            vals_list = []
            for r in range(p):
                i1 = int(I[r]); j1 = int(Jof[i1]); k1 = int(Kof[i1])
                i2 = int(J[r]); j2 = int(Jof[i2]); k2 = int(Kof[i2])
                x1 = X[i1]; x2 = X[i2]
                x1m = x1.reshape(6, 2); x2m = x2.reshape(6, 2)
                # a1
                cc1 = C[:, k1]; ss1 = S_[:, k1]
                a1_0 = cc1 * x2m[:, 0] + ss1 * x2m[:, 1]
                a1_1 = -ss1 * x2m[:, 0] + cc1 * x2m[:, 1]
                a1 = np.empty(12); a1[0::2] = a1_0; a1[1::2] = a1_1
                cc2 = C[:, k2]; ss2 = S_[:, k2]
                a2_0 = cc2 * x1m[:, 0] + ss2 * x1m[:, 1]
                a2_1 = -ss2 * x1m[:, 0] + cc2 * x1m[:, 1]
                a2 = np.empty(12); a2[0::2] = a2_0; a2[1::2] = a2_1
                base1 = j1 * 12; base2 = j2 * 12
                for dd in range(12):
                    rows_list.append(r); cols_list.append(base1 + dd); vals_list.append(a1[dd])
                    rows_list.append(r); cols_list.append(base2 + dd); vals_list.append(a2[dd])
                rows_list.append(r); cols_list.append(48); vals_list.append(-1.0)
            nv = 49
            A_ub = coo_matrix((np.array(vals_list), (np.array(rows_list), np.array(cols_list))), shape=(p, nv)).tocsr()
            # A_eq: Scur[j] . d_j = 0 (4 rows)
            er, ec, ev = [], [], []
            for j in range(4):
                for dd in range(12):
                    er.append(j); ec.append(j*12+dd); ev.append(float(Scur[j, dd]))
            A_eq = coo_matrix((np.array(ev), (np.array(er), np.array(ec))), shape=(4, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(4),
                          bounds=[(-radius, radius)] * 48 + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        d = res.x[:48].reshape(4, 12)
        Snew = Scur + d
        Snew /= np.linalg.norm(Snew, axis=1, keepdims=True)
        Xnew = _expand_sym(Snew, C, S_)
        v = _max_cos(Xnew)
        if v < best - 1e-15:
            Scur, X, best = Snew, Xnew, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                St = Scur + d * sh
                St /= np.linalg.norm(St, axis=1, keepdims=True)
                Xt = _expand_sym(St, C, S_)
                vt = _max_cos(Xt)
                if vt < best:
                    Scur, X, best = St, Xt, vt
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return X, Scur

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    rng = np.random.default_rng(20261031)
    n, d = 68, 12
    bestX = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < best and np.all(np.isfinite(a)):
                    best = s
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = _max_cos(bestX)
    incumbent = bestX.copy()
    inc_score = best

    # build ~20 multiplier sets
    mult_list = []
    mult_list.append(np.array([1, 2, 3, 4, 5, 6], dtype=int))
    mult_list.append(np.array([1, 1, 2, 3, 5, 8], dtype=int))
    mult_list.append(np.array([1, 2, 4, 8, 3, 6], dtype=int))
    # random rest
    while len(mult_list) < 20:
        m = rng.integers(1, 17, size=6)
        # avoid all-zero (never zero here) and avoid duplicates-heavy? allow
        # canonicalize sign: map >8 to 17-m to reduce duplicates? keep as is for diversity
        mult_list.append(m.astype(int))
    # apply random sign flips / permutation implicitly by random mult; also permute planes randomly
    for i in range(len(mult_list)):
        if i >= 3:
            perm = rng.permutation(6)
            mult_list[i] = mult_list[i][perm]
            # random sign: replace m by 17-m with prob 0.5
            fl = rng.random(6) < 0.5
            mult_list[i][fl] = 17 - mult_list[i][fl]

    cands = []  # (score, X, Seeds, mult, C, S_)
    scan_deadline = min(t0 + 120.0, T_END - 50.0)
    for mi, mult in enumerate(mult_list):
        if time.time() > scan_deadline:
            break
        # skip degenerate with zero mod17
        if np.any(mult % 17 == 0):
            continue
        C, S_ = _build_CS(mult % 17)
        y0 = rng.standard_normal(48)
        # quick sym opt with reduced iters to fit 20 in time
        Xsym, Seeds, _ = _opt_sym(y0, mult, C, S_,
                                  betas=(100.0, 300.0, 1000.0, 3000.0), maxiter=120)
        s = _max_cos(Xsym)
        cands.append((s, Xsym, Seeds, mult.copy(), C, S_))
        # early stop if time over
    cands.sort(key=lambda t: t[0])
    # polish best two with symmetry-preserving LP
    for k in range(min(2, len(cands))):
        if time.time() > T_END - 45.0:
            break
        s, Xsym, Seeds, mult, C, S_ = cands[k]
        Xp, Sp = _sym_lp_polish(Seeds, mult, C, S_, min(time.time() + 6.0, T_END - 35.0))
        sp = _max_cos(Xp)
        # update candidate in place
        cands[k] = (sp, Xp, Sp, mult, C, S_)

    cands.sort(key=lambda t: t[0])
    # decide release: only if symmetric within 5e-4 of incumbent
    for k in range(min(2, len(cands))):
        if time.time() > T_END - 30.0:
            break
        s, Xsym, Seeds, mult, C, S_ = cands[k]
        if s > inc_score + 5e-4:
            continue
        # release symmetry: full LBFGS + full LP
        Xr = _run_full(Xsym, betas=(100.0, 300.0, 600.0), maxiter=100)
        if time.time() < T_END - 15.0:
            Xr = _lp_polish_full(Xr, min(time.time() + 7.0, T_END - 8.0))
        sr = _max_cos(Xr)
        if sr < inc_score - 1e-12 and np.all(np.isfinite(Xr)):
            inc_score = sr
            incumbent = Xr.copy()

    # if no symmetric candidate qualified, still try quick release of overall best if close? (respect rule)
    # fallback: ensure incumbent at least LP-polished lightly if time remains and we improved? Don't waste.
    # Final: return best among incumbent (never worse than loaded best)
    incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    # safety: if somehow incumbent invalid, return bestX
    if not np.all(np.isfinite(incumbent)):
        return bestX
    # guarantee no worse than input best: if our search failed, incumbent == input best
    return incumbent

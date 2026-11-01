import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

# ---------- full space helpers ----------
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

def _run_full(X0, betas, maxiter=100):
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

# ---------- 4x17 cyclic symmetry ----------
def _build_CS(mult):
    K = 17
    C = np.empty((6, K)); Sm = np.empty((6, K))
    for i, m in enumerate(mult):
        for k in range(K):
            th = 2.0 * np.pi * float(int(m)) * float(k) / 17.0
            C[i, k] = np.cos(th); Sm[i, k] = np.sin(th)
    return C, Sm

def _expand_sym(Sn, C, S_):
    # Sn (4,12) -> X (68,12)
    S0 = Sn[:, 0::2]  # (4,6)
    S1 = Sn[:, 1::2]
    CT = C.T  # (17,6)
    ST = S_.T
    X0 = S0[:, None, :] * CT[None, :, :] - S1[:, None, :] * ST[None, :, :]
    X1 = S0[:, None, :] * ST[None, :, :] + S1[:, None, :] * CT[None, :, :]
    X = np.empty((4, 17, 12))
    X[:, :, 0::2] = X0
    X[:, :, 1::2] = X1
    return X.reshape(68, 12)

def _make_sym_fg(C, S_, beta, IU):
    CT = C.T
    ST = S_.T
    def fg(y_raw):
        Y = np.asarray(y_raw, dtype=np.float64).reshape(4, 12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Sn = Y / nr
        X = _expand_sym(Sn, C, S_)
        g = (X @ X.T)[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        Sm_ = float(np.sum(e)) + 1e-300
        F = m + np.log(Sm_) / beta
        w = e / Sm_
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        Gxt = GX - np.sum(GX * X, axis=1, keepdims=True) * X
        G4 = Gxt.reshape(4, 17, 12)
        G0 = G4[:, :, 0::2]  # (4,17,6)
        G1 = G4[:, :, 1::2]
        # back-rotate: ds0 = C*g0 + S*g1, ds1 = -S*g0 + C*g1, sum over k
        # vectorized with einsum
        # C,S_: (6,17)
        GS0 = np.einsum('jk,ajk->aj', C, np.einsum('ajk,jk->ajk', np.ones((4,17,6)), G0) if False else None
        # explicit einsum
        # GS0[a,j] = sum_k C[j,k]*G0[a,k,j] + S_[j,k]*G1[a,k,j]
        GS0 = np.einsum('jk,akj->aj', C, np.transpose(G0, (0, 2, 1))) + np.einsum('jk,akj->aj', S_, np.transpose(G1, (0, 2, 1)))
        GS1 = -np.einsum('jk,akj->aj', S_, np.transpose(G0, (0, 2, 1))) + np.einsum('jk,akj->aj', C, np.transpose(G1, (0, 2, 1)))
        GS = np.empty((4, 12))
        GS[:, 0::2] = GS0
        GS[:, 1::2] = GS1
        GY = (GS - np.sum(GS * Sn, axis=1, keepdims=True) * Sn) / nr
        return F, GY.ravel()
    return fg

def _run_sym(y0, C, S_, IU, betas, maxiters):
    y = y0.copy()
    for beta, mi in zip(betas, maxiters):
        fg = _make_sym_fg(C, S_, beta, IU)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': mi*3})
            y = r.x.copy()
        except Exception:
            break
    return y

def _sym_lp_polish(Sn0, C, S_, deadline, r0=8e-4, cap_p=2500):
    Sn = Sn0 / np.linalg.norm(Sn0, axis=1, keepdims=True)
    X = _expand_sym(Sn, C, S_)
    best = _max_cos(X)
    radius = r0
    iu = np.triu_indices(68, 1)
    nvar = 48
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p0 = len(I)
        if p0 == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p0 > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p0 = cap_p
        A = I // 17; Kk = I % 17
        B = J // 17; Ll = J % 17
        Xf = X  # (68,12)
        Xp = Xf[I]  # (p,12)
        Xq = Xf[J]
        Xp0 = Xp[:, 0::2]; Xp1 = Xp[:, 1::2]  # (p,6)
        Xq0 = Xq[:, 0::2]; Xq1 = Xq[:, 1::2]
        # gather C,S for k and l: shapes (p,6)
        # C is (6,17)
        Ck = C[:, Kk].T  # (p? ) C[:,Kk] is (6,p) -> T (p,6)
        Sk = S_[:, Kk].T
        Cl = C[:, Ll].T
        Sl = S_[:, Ll].T
        # coeff_a = R(k)^T Xq
        Ca0 = Ck * Xq0 + Sk * Xq1
        Ca1 = -Sk * Xq0 + Ck * Xq1
        Cb0 = Cl * Xp0 + Sl * Xp1
        Cb1 = -Sl * Xp0 + Cl * Xp1
        Ca = np.empty((p0, 12)); Cb = np.empty((p0, 12))
        Ca[:, 0::2] = Ca0; Ca[:, 1::2] = Ca1
        Cb[:, 0::2] = Cb0; Cb[:, 1::2] = Cb1
        # build dense A_ub (p,49)
        Aub = np.zeros((p0, nvar + 1))
        for t in range(p0):
            a = int(A[t]); b = int(B[t])
            if a == b:
                Aub[t, a*12:(a+1)*12] = Ca[t] + Cb[t]
            else:
                Aub[t, a*12:(a+1)*12] = Ca[t]
                Aub[t, b*12:(b+1)*12] = Cb[t]
        Aub[:, -1] = -1.0
        bub = -csel
        Aeq = np.zeros((4, nvar + 1))
        for a in range(4):
            Aeq[a, a*12:(a+1)*12] = Sn[a]
        beq = np.zeros(4)
        obj = np.zeros(nvar + 1); obj[-1] = 1.0
        bounds = [(-radius, radius)] * nvar + [(None, None)]
        try:
            res = linprog(obj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        DS = res.x[:nvar].reshape(4, 12)
        SnT = Sn + DS
        SnT /= np.linalg.norm(SnT, axis=1, keepdims=True)
        XT = _expand_sym(SnT, C, S_)
        v = _max_cos(XT)
        if v < best - 1e-15:
            Sn, X, best = SnT, XT, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for sh in (0.3, 0.1):
                SnS = Sn + DS * sh
                SnS /= np.linalg.norm(SnS, axis=1, keepdims=True)
                XS = _expand_sym(SnS, C, S_)
                vs = _max_cos(XS)
                if vs < best:
                    Sn, X, best = SnS, XS, vs
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return Sn, X, best

def solve():
    t0 = time.time()
    TL = 172.0
    rng = np.random.default_rng(0)
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    except Exception:
        pass
    if not cands:
        x0 = rng.standard_normal((68, 12))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    x_best = cands[0].copy()
    best_val = _max_cos(x_best)
    for cc in cands[1:]:
        v = _max_cos(cc)
        if v < best_val:
            best_val = v
            x_best = cc.copy()

    IU = np.triu_indices(68, 1)

    # multiplier sets: 6 values mod 17, non-zero
    mult_list = []
    mult_list.append(np.array([1, 2, 3, 4, 5, 6], dtype=int))
    mult_list.append(np.array([1, 2, 4, 8, 16, 9], dtype=int))
    mult_list.append(np.array([1, 3, 5, 7, 2, 6], dtype=int))
    mult_list.append(np.array([1, 2, 3, 5, 8, 13], dtype=int))
    mult_list.append(np.array([1, 4, 6, 9, 11, 16], dtype=int))
    mult_list.append(np.array([2, 3, 5, 7, 11, 13], dtype=int))
    # random magnitude+sign sets
    for _ in range(16):
        mag = rng.integers(1, 9, size=6)
        sgn = rng.choice(np.array([-1, 1]), size=6)
        m = (sgn * mag) % 17
        # avoid zeros (mag>=1 so no zero unless 17, impossible)
        m[m == 0] = 1
        mult_list.append(m.astype(int))

    sym_cands = []  # (val, Sn, mult, C, S_)
    sym_deadline = t0 + 112.0
    for mi, mult in enumerate(mult_list):
        if time.time() > sym_deadline:
            break
        # time-adaptive maxiters
        remain = sym_deadline - time.time()
        if remain < 3.0:
            break
        C, S_ = _build_CS(mult)
        y0 = rng.standard_normal(48)
        # betas per spec: 100,300,1000,3000
        if remain < 20:
            betas = [100.0, 300.0, 1000.0]
            mis = [50, 60, 60]
        else:
            betas = [100.0, 300.0, 1000.0, 3000.0]
            mis = [60, 70, 80, 80]
        try:
            yopt = _run_sym(y0, C, S_, IU, betas, mis)
        except Exception:
            continue
        Y = yopt.reshape(4, 12)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        try:
            X = _expand_sym(Y, C, S_)
        except Exception:
            continue
        # check degeneracies / NaN
        if not np.all(np.isfinite(X)):
            continue
        v = _max_cos(X)
        sym_cands.append((v, Y.copy(), mult.copy(), C, S_))
    sym_cands.sort(key=lambda t: t[0])

    # polish best two with symmetry-preserving LP
    polished = []
    for k in range(min(2, len(sym_cands))):
        if time.time() > t0 + 135.0:
            break
        v, Sn, mult, C, S_ = sym_cands[k]
        # quick extra LBFGS at high beta before LP if time allows
        try:
            y0 = Sn.ravel().copy()
            yopt = _run_sym(y0, C, S_, IU, [3000.0], [60])
            Sn2 = yopt.reshape(4, 12)
            Sn2 /= np.linalg.norm(Sn2, axis=1, keepdims=True)
        except Exception:
            Sn2 = Sn
        dl = min(time.time() + 5.0, t0 + 138.0)
        try:
            SnP, XP, vP = _sym_lp_polish(Sn2, C, S_, dl, r0=8e-4)
        except Exception:
            vP = _max_cos(_expand_sym(Sn2, C, S_)); XP = _expand_sym(Sn2, C, S_); SnP = Sn2
        polished.append((vP, XP, SnP, mult, C, S_))
        if vP < best_val:
            best_val = vP
            x_best = XP.copy()
    polished.sort(key=lambda t: t[0])

    # release symmetry for final LP only if within 5e-4 of incumbent
    # incumbent threshold: best_val at this point
    thresh = best_val + 5e-4
    # note: best_val may have been updated by symmetric polish
    released_done = False
    if polished and polished[0][0] <= thresh + 1e-12:
        for k in range(min(2, len(polished))):
            if time.time() > t0 + TL - 12.0:
                break
            vP, XP, SnP, mult, C, S_ = polished[k]
            if vP > thresh + 1e-12:
                continue
            Xr = XP.copy()
            # full L-BFGS release
            try:
                Xr = _run_full(Xr, betas=(300.0, 800.0), maxiter=120)
            except Exception:
                pass
            try:
                Xr = _lp_polish_full(Xr, min(time.time() + 6.0, t0 + TL - 6.0))
            except Exception:
                pass
            vr = _max_cos(Xr)
            released_done = True
            if vr < best_val:
                best_val = vr
                x_best = Xr.copy()
            # only polish best; if first succeeded break to save time for incumbent squeeze
            break

    # final squeeze of incumbent (does not violate symmetry rule: direct incumbent polish)
    if time.time() < t0 + TL - 8.0:
        try:
            # short full LP squeeze to eke out 1e-12
            x_best = _lp_polish_full(x_best, min(time.time() + 5.0, t0 + TL - 2.0), r0=5e-4, cap_p=2500)
            best_val = _max_cos(x_best)
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # shrink by tiny margin guard (renormalize keeps constraints strict)
    return x_best

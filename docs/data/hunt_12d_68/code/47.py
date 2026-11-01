import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish_full(x, deadline, r0=8e-4, cap_p=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.35:
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

def _full_fg_factory(IU, beta, n=68, d=12):
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

def _run_full(X0, betas, maxiter):
    n, d = X0.shape
    IU = np.triu_indices(n, 1)
    y = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    for b in betas:
        fg = _full_fg_factory(IU, b, n, d)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _build_CS(mult):
    C = np.empty((6, 17)); S = np.empty((6, 17))
    for i, m in enumerate(mult):
        mm = int(m) % 17
        for k in range(17):
            th = 2.0 * np.pi * float(mm) * float(k) / 17.0
            C[i, k] = np.cos(th); S[i, k] = np.sin(th)
    return C, S

def _expand_sym(Sn, C, Sm):
    S6 = Sn.reshape(4, 6, 2)
    S0 = S6[:, :, 0]; S1 = S6[:, :, 1]
    CT = C.T; ST = Sm.T
    X0 = S0[:, None, :] * CT[None, :, :] - S1[:, None, :] * ST[None, :, :]
    X1 = S0[:, None, :] * ST[None, :, :] + S1[:, None, :] * CT[None, :, :]
    X6 = np.empty((4, 17, 6, 2))
    X6[:, :, :, 0] = X0; X6[:, :, :, 1] = X1
    return X6.reshape(68, 12)

def _make_sym_fg(C, Sm, beta, IU):
    CTb = C.T[None, :, :]
    STb = Sm.T[None, :, :]
    def fg(y_raw):
        Y = np.asarray(y_raw, dtype=np.float64).reshape(4, 12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Sn = Y / nr
        X = _expand_sym(Sn, C, Sm)
        g = (X @ X.T)[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        Ssum = float(np.sum(e)) + 1e-300
        F = m + np.log(Ssum) / beta
        w = e / Ssum
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        G6 = GX.reshape(4, 17, 6, 2)
        G0 = G6[:, :, :, 0]; G1 = G6[:, :, :, 1]
        dS0 = np.sum(CTb * G0 + STb * G1, axis=1)
        dS1 = np.sum(-STb * G0 + CTb * G1, axis=1)
        dS6 = np.empty((4, 6, 2))
        dS6[:, :, 0] = dS0; dS6[:, :, 1] = dS1
        dS = dS6.reshape(4, 12)
        proj = np.sum(dS * Sn, axis=1, keepdims=True)
        GY = (dS - proj * Sn) / nr
        return F, GY.ravel()
    return fg

def _opt_sym(y0, C, Sm, betas, maxiter):
    IU = np.triu_indices(68, 1)
    y = y0.copy()
    for b in betas:
        fg = _make_sym_fg(C, Sm, b, IU)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(4, 12)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    X = _expand_sym(Y, C, Sm)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, Y, y

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

    mult_list = []
    mult_list.append(np.array([1, 2, 3, 4, 5, 6], dtype=int))
    mult_list.append(np.array([1, 1, 2, 3, 5, 8], dtype=int))
    mult_list.append(np.array([1, 2, 4, 8, 3, 6], dtype=int))
    mult_list.append(np.array([1, 3, 5, 7, 9, 11], dtype=int))
    mult_list.append(np.array([2, 3, 5, 7, 11, 13], dtype=int))
    while len(mult_list) < 20:
        m = rng.integers(1, 17, size=6).astype(int)
        perm = rng.permutation(6)
        m = m[perm]
        fl = rng.random(6) < 0.5
        m[fl] = 17 - m[fl]
        # avoid zero
        m[m == 0] = 1
        m[m == 17] = 16
        mult_list.append(m)

    cands = []
    scan_deadline = min(t0 + 115.0, T_END - 55.0)
    for mi, mult in enumerate(mult_list):
        if time.time() > scan_deadline:
            break
        C, Sm = _build_CS(mult)
        ntry = 2 if mi < 4 else 1
        for t in range(ntry):
            if time.time() > scan_deadline:
                break
            S0 = rng.standard_normal((4, 12))
            S0 /= np.linalg.norm(S0, axis=1, keepdims=True)
            y0 = S0.ravel().copy()
            # quick symmetric schedule
            try:
                X, Seeds, _ = _opt_sym(y0, C, Sm, betas=(100.0, 300.0, 1000.0), maxiter=60)
            except Exception:
                continue
            s = _max_cos(X)
            cands.append((s, X.copy(), Seeds.copy(), mult.copy(), C.copy(), Sm.copy()))
            # early extra beta for promising
            if s < inc_score + 5e-4 and time.time() < scan_deadline - 5:
                try:
                    X2, S2, _ = _opt_sym(Seeds.ravel().copy(), C, Sm, betas=(3000.0,), maxiter=60)
                    s2 = _max_cos(X2)
                    cands.append((s2, X2.copy(), S2.copy(), mult.copy(), C.copy(), Sm.copy()))
                except Exception:
                    pass
    cands.sort(key=lambda z: z[0])
    # keep top 2 within window + incumbent
    top = []
    for c in cands[:6]:
        if c[0] < inc_score + 5e-4:
            top.append(c)
        if len(top) >= 2:
            break
    # if none within window, still take best one for diagnostics but will return incumbent
    if len(top) == 0 and len(cands) > 0:
        top = cands[:1]

    # release symmetry: full LSE + LP on top candidates
    cur_bestX = incumbent.copy()
    cur_best = inc_score
    # budget for release
    for ci, (s, Xsym, Seeds, mult, C, Sm) in enumerate(top):
        if time.time() > T_END - 25.0:
            break
        # full LBFGS from symmetric point
        try:
            Xf = _run_full(Xsym, betas=(300.0, 1000.0, 3000.0), maxiter=80)
        except Exception:
            Xf = Xsym
        sf = _max_cos(Xf)
        # LP polish with remaining time slice
        dl = min(time.time() + 12.0, T_END - 8.0)
        try:
            Xp = _lp_polish_full(Xf, dl, r0=8e-4)
        except Exception:
            Xp = Xf
        sp = _max_cos(Xp)
        if sp < cur_best - 1e-15:
            cur_best = sp
            cur_bestX = Xp.copy()
        elif sf < cur_best - 1e-15:
            cur_best = sf
            cur_bestX = Xf.copy()
        # small kick + re-polish to change basin (topology change)
        if time.time() < T_END - 20.0:
            for sig in (0.002, 0.006):
                if time.time() > T_END - 18.0:
                    break
                try:
                    Xk = Xp + sig * rng.standard_normal((n, d))
                    Xk /= np.linalg.norm(Xk, axis=1, keepdims=True)
                    Xk2 = _run_full(Xk, betas=(800.0, 2500.0), maxiter=60)
                    dl2 = min(time.time() + 6.0, T_END - 8.0)
                    Xk3 = _lp_polish_full(Xk2, dl2, r0=5e-4)
                    sk = _max_cos(Xk3)
                    if sk < cur_best:
                        cur_best = sk
                        cur_bestX = Xk3.copy()
                        Xp = Xk3.copy()
                except Exception:
                    continue

    # also try direct kicks from incumbent (escape jammed basin)
    if time.time() < T_END - 15.0:
        for sig in (0.0015, 0.004, 0.01):
            if time.time() > T_END - 10.0:
                break
            try:
                Xk = incumbent + sig * rng.standard_normal((n, d))
                Xk /= np.linalg.norm(Xk, axis=1, keepdims=True)
                Xk2 = _run_full(Xk, betas=(500.0, 2000.0), maxiter=50)
                dl2 = min(time.time() + 4.0, T_END - 3.0)
                Xk3 = _lp_polish_full(Xk2, dl2, r0=5e-4)
                sk = _max_cos(Xk3)
                if sk < cur_best:
                    cur_best = sk
                    cur_bestX = Xk3.copy()
            except Exception:
                continue

    # final squeeze of global best
    if time.time() < T_END - 3.0:
        try:
            cur_bestX = _lp_polish_full(cur_bestX, T_END - 0.5, r0=3e-4)
            cur_best = _max_cos(cur_bestX)
        except Exception:
            pass
    # safety: return incumbent if none improved
    if cur_best > inc_score and bestX is not None:
        cur_bestX = incumbent.copy()
    cur_bestX = cur_bestX / np.linalg.norm(cur_bestX, axis=1, keepdims=True)
    return cur_bestX

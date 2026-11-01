import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline, r0=8e-4, cap_p=2000):
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

def _build_M(D):
    d = len(D)
    M = np.ones((d, d))
    for k in range(1, d):
        prev = M[k-1]
        cur = np.empty(d)
        for i in range(d):
            cur[i] = D[(i-1) % d] * prev[(i-1) % d]
        M[k] = cur
    return M

def _expand_from_seeds(S5, A4list, M):
    d = 12
    X = np.empty((68, d))
    for i in range(5):
        s = S5[i]
        for k in range(12):
            Mk = M[k]
            idx = (np.arange(d) - k) % d
            X[i*12+k] = Mk * s[idx]
    for j, a in enumerate(A4list):
        sfull = np.tile(a, 3) / np.sqrt(3.0)
        base = 60 + j*4
        for k in range(4):
            Mk = M[k]
            idx = (np.arange(d) - k) % d
            X[base+k] = Mk * sfull[idx]
    return X

def _seeds_from_raw(y_raw):
    S = []
    for i in range(5):
        v = y_raw[i*12:(i+1)*12]
        n = max(float(np.linalg.norm(v)), 1e-12)
        S.append(v / n)
    A = []
    for j in range(2):
        v = y_raw[60+j*4:64+j*4]
        n = max(float(np.linalg.norm(v)), 1e-12)
        A.append(v / n)
    return S, A

def _make_seed_fg(M, beta):
    n, d = 68, 12
    IU = np.triu_indices(n, 1)
    def fg(y_raw):
        y_raw = np.asarray(y_raw, dtype=np.float64)
        S5, A4 = _seeds_from_raw(y_raw)
        Sfull_list = [np.tile(a, 3)/np.sqrt(3.0) for a in A4]
        Sall = S5 + Sfull_list  # len 7, first 5 dim12, last 2 dim12 tiled
        X = _expand_from_seeds(S5, A4, M)
        # LSE over pairs
        G = X @ X.T
        g = G[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        Sm = float(np.sum(e)) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X  # dF/dX
        # backprop to seeds
        # dF/dS for 5 full seeds
        dS5 = []
        for i in range(5):
            acc = np.zeros(d)
            for k in range(12):
                row = GX[i*12+k]  # d
                Mk = M[k]
                # X[i,k][j]=Mk[j]*S[(j-k)%d]; so dF/dS[l]+=GX[j=(l+k)%d]*Mk[j]
                for l in range(d):
                    j = (l + k) % d
                    acc[l] += row[j] * Mk[j]
            dS5.append(acc)
        dSfull = []
        for j in range(2):
            acc = np.zeros(d)
            base = 60 + j*4
            for k in range(4):
                row = GX[base+k]
                Mk = M[k]
                for l in range(d):
                    jj = (l + k) % d
                    acc[l] += row[jj] * Mk[jj]
            dSfull.append(acc)
        # to raw
        grad = np.zeros_like(y_raw)
        for i in range(5):
            v = y_raw[i*12:(i+1)*12]
            nv = max(float(np.linalg.norm(v)), 1e-12)
            s = v / nv
            ds = dS5[i]
            proj = float(ds @ s)
            gv = (ds - proj * s) / nv
            grad[i*12:(i+1)*12] = gv
        for j in range(2):
            # dSfull -> da (4-dim) via tiling
            dsf = dSfull[j]
            # sfull = tile(a,3)/sqrt3
            da_tilde = np.zeros(4)
            for r in range(4):
                da_tilde[r] = (dsf[r] + dsf[r+4] + dsf[r+8]) / np.sqrt(3.0)
            v = y_raw[60+j*4:64+j*4]
            nv = max(float(np.linalg.norm(v)), 1e-12)
            a = v / nv
            proj = float(da_tilde @ a)
            ga = (da_tilde - proj * a) / nv
            grad[60+j*4:64+j*4] = ga
        return F, grad
    return fg

def _opt_seeds(y0, M, betas=(80.0, 200.0, 400.0), maxiter=120):
    y = y0.copy()
    for b in betas:
        fg = _make_seed_fg(M, b)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
            y = r.x.copy()
        except Exception:
            break
    S5, A4 = _seeds_from_raw(y)
    X = _expand_from_seeds(S5, A4, M)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, y

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

def _run_full(X0, betas=(100.0, 300.0, 600.0), maxiter=150):
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

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    rng = np.random.default_rng(1234567)
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

    # symmetry scan: several sign patterns D with prod +1
    Dall = []
    Dall.append(np.ones(12))
    D = np.ones(12); D[0] = -1; D[6] = -1; Dall.append(D.copy())
    D = np.ones(12); D[0] = -1; D[1] = -1; D[6] = -1; D[7] = -1; Dall.append(D.copy())
    D = np.ones(12)
    for i in range(0, 12, 2):
        D[i] = -1
    # product of alternating: (+? 6 negatives => prod +1)
    Dall.append(D.copy())
    D = np.ones(12); D[0] = -1; D[3] = -1; D[6] = -1; D[9] = -1; Dall.append(D.copy())
    D = np.ones(12); D[1] = -1; D[5] = -1; Dall.append(D.copy())

    sym_cands = []
    # time box for sym scan ~ 60s
    sym_deadline = min(T_END - 90.0, t0 + 70.0)
    for di, Dv in enumerate(Dall):
        if time.time() > sym_deadline:
            break
        M = _build_M(Dv)
        for rep in range(3):
            if time.time() > sym_deadline:
                break
            y0 = rng.standard_normal(68)
            # seed from best projected? occasionally project best onto seeds via random pick
            if rep == 0 and bestX is not None:
                # init seeds from random rows of bestX
                idx = rng.choice(n, size=7, replace=False)
                y0 = np.concatenate([bestX[idx[i]].ravel() if i < 5 else bestX[idx[i]][:4].ravel() for i in range(7)]).copy()
                # ensure length 68: 5*12+2*4=68
                if y0.size != 68:
                    y0 = rng.standard_normal(68)
            Xsym, _ = _opt_seeds(y0, M, betas=(80.0, 220.0, 450.0), maxiter=100)
            s = _max_cos(Xsym)
            sym_cands.append((s, Xsym, di))
            # quick release test for promising ones: full LBFGS short
            if s < best + 0.02 and time.time() < sym_deadline - 5:
                Xr = _run_full(Xsym, betas=(150.0, 400.0), maxiter=80)
                sr = _max_cos(Xr)
                sym_cands.append((sr, Xr, di))
    # pick best symmetric-derived candidate
    sym_cands.sort(key=lambda t: t[0])
    incumbent = bestX.copy()
    inc_score = best
    # release top 2 sym candidates with polish
    for k in range(min(2, len(sym_cands))):
        if time.time() > T_END - 55.0:
            break
        _, Xc, _ = sym_cands[k]
        Xr = _run_full(Xc, betas=(100.0, 300.0, 600.0), maxiter=120)
        # short LP
        Xr = _lp_polish(Xr, min(time.time() + 8.0, T_END - 40.0))
        sr = _max_cos(Xr)
        if sr < inc_score and np.all(np.isfinite(Xr)):
            inc_score = sr
            incumbent = Xr.copy()
    if inc_score < best:
        best, bestX = inc_score, incumbent.copy()
    # final basin hops around incumbent: random kicks + full LBFGS + LP
    # also LP-polish incumbent once
    if time.time() < T_END - 20.0:
        bestX = _lp_polish(bestX, min(time.time() + 10.0, T_END - 12.0))
        best = _max_cos(bestX)
    kick = 0
    while time.time() < T_END - 6.0 and kick < 12:
        kick += 1
        if time.time() > T_END - 8.0:
            break
        # random perturbation strength
        amp = 0.01 if kick % 2 == 0 else 0.03
        Xt = bestX + amp * rng.standard_normal(bestX.shape)
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        Xt = _run_full(Xt, betas=(200.0, 500.0), maxiter=70)
        if _max_cos(Xt) < best + 0.002:
            Xt = _lp_polish(Xt, min(time.time() + 4.0, T_END - 4.0))
        s = _max_cos(Xt)
        if s < best - 1e-12 and np.all(np.isfinite(Xt)):
            best = s
            bestX = Xt.copy()
    if time.time() < T_END - 2.0:
        # ensure LP squeeze remainder
        remain = T_END - time.time() - 1.0
        if remain > 3.0:
            bestX = _lp_polish(bestX, time.time() + min(remain, 12.0))
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

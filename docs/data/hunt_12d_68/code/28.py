import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, deadline, r0=8e-4, cap_p=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.25:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for shrink in (0.3, 0.1):
                yy = x + res.x[:-1].reshape(n, d) * shrink
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def _full_fg_factory(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
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

def _run_full_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = _full_fg_factory(n, d, IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': maxiter * 3})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _build_M(D):
    d = len(D)
    K = d
    M = np.ones((K, d))
    for k in range(1, K):
        prev = M[k - 1]
        cur = np.empty(d)
        for i in range(d):
            cur[i] = D[(i - 1) % d] * prev[(i - 1) % d]
        M[k] = cur
    return M

def _expand_mixed(y_raw, M):
    d = 12
    parts12 = [y_raw[i * 12:(i + 1) * 12] for i in range(5)]
    y5 = y_raw[60:64]
    y6 = y_raw[64:68]
    S = []
    for v in parts12:
        n = max(np.linalg.norm(v), 1e-12)
        S.append(v / n)
    for v in (y5, y6):
        n = max(np.linalg.norm(v), 1e-12)
        a = v / n
        S.append(np.tile(a, 3) / np.sqrt(3.0))
    X = np.empty((68, d))
    for i in range(5):
        s = S[i]
        for k in range(12):
            Mk = M[k]
            idx = (np.arange(d) - k) % d
            X[i * 12 + k] = Mk * s[idx]
    for j, sj in enumerate([S[5], S[6]]):
        base = 60 + j * 4
        for k in range(4):
            X[base + k] = np.roll(sj, k)
    return X, S

def _seed_fg_factory(M, beta):
    d = 12
    def fg(y_raw):
        y_raw = np.asarray(y_raw)
        norms5 = []
        S = []
        for i in range(5):
            v = y_raw[i * 12:(i + 1) * 12]
            n = max(float(np.linalg.norm(v)), 1e-12)
            norms5.append(n)
            S.append(v / n)
        raws = [y_raw[60:64], y_raw[64:68]]
        normsA = []
        Aunit = []
        Sfull = []
        for v in raws:
            n = max(float(np.linalg.norm(v)), 1e-12)
            normsA.append(n)
            a = v / n
            Aunit.append(a)
            Sfull.append(np.tile(a, 3) / np.sqrt(3.0))
        Sall = S + Sfull
        X = np.empty((68, d))
        for i in range(5):
            s = Sall[i]
            for k in range(12):
                Mk = M[k]
                idx = (np.arange(d) - k) % d
                X[i * 12 + k] = Mk * s[idx]
        for j in range(2):
            sj = Sall[5 + j]
            base = 60 + j * 4
            for k in range(4):
                X[base + k] = np.roll(sj, k)
        G = X @ X.T
        iu = np.triu_indices(68, 1)
        g = G[iu]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        Ssum = float(e.sum()) + 1e-300
        F = m + np.log(Ssum) / beta
        w = e / Ssum
        P = np.zeros((68, 68))
        P[iu] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        T = GX - dot * X
        grad = np.zeros_like(y_raw)
        for i in range(5):
            acc = np.zeros(d)
            for k in range(12):
                t = T[i * 12 + k]
                Mk = M[k]
                for j in range(d):
                    jj = (j + k) % d
                    acc[j] += Mk[jj] * t[jj]
            grad[i * 12:(i + 1) * 12] = acc / norms5[i]
        for j in range(2):
            base = 60 + j * 4
            accf = np.zeros(d)
            for k in range(4):
                t = T[base + k]
                accf += np.roll(t, -k)
            h = np.zeros(4)
            for p in range(4):
                h[p] = (accf[p] + accf[p + 4] + accf[p + 8]) / np.sqrt(3.0)
            a = Aunit[j]
            hn = float(h @ a)
            ht = h - hn * a
            grad[60 + j * 4:60 + (j + 1) * 4] = ht / normsA[j]
        return F, grad
    return fg

def _run_seed_lbfgs(y0, M, beta, maxiter):
    fg = _seed_fg_factory(M, beta)
    try:
        r = minimize(fg, y0.copy(), jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        return r.x
    except Exception:
        return y0

def solve():
    t0 = time.time()
    TLIM = 172.0
    n, d = 68, 12
    x_load = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < bv:
                    bv = s
                    x_load = a.copy()
        except Exception:
            pass
    if x_load is None:
        rng0 = np.random.default_rng(0)
        x_load = rng0.standard_normal((n, d))
        x_load /= np.linalg.norm(x_load, axis=1, keepdims=True)
        bv = _max_cos(x_load)
    x_global_best = x_load.copy()
    global_best = bv

    D0 = np.ones(12)
    D1 = np.ones(12); D1[0] = -1; D1[6] = -1
    D2 = np.ones(12); D2[0] = -1; D2[1] = -1; D2[6] = -1; D2[7] = -1
    D3 = np.ones(12); D3[3] = -1; D3[9] = -1
    patterns = [D0, D1, D2, D3]
    Ms = [_build_M(D) for D in patterns]

    sym_deadline = t0 + 55.0
    best_sym_X = None
    best_sym_val = 1e9
    rng = np.random.default_rng(777)
    trial = 0
    for pi, M in enumerate(Ms):
        for rep in range(2):
            if time.time() > sym_deadline:
                break
            trial += 1
            y0 = rng.standard_normal(68)
            # seed L-BFGS escalating sharpness on 68-dim seed space
            for beta, it in [(60.0, 60), (250.0, 80), (1000.0, 80)]:
                if time.time() > sym_deadline:
                    break
                y0 = _run_seed_lbfgs(y0, M, beta, it)
            Xsym, _ = _expand_mixed(y0, M)
            v = _max_cos(Xsym)
            # release: short full polish to judge basin
            if time.time() < sym_deadline:
                Xr = _run_full_lbfgs(Xsym, 600.0, 60)
                vr = _max_cos(Xr)
                if vr < v:
                    Xsym, v = Xr, vr
            if v < best_sym_val:
                best_sym_val = v
                best_sym_X = Xsym.copy()
            # early keep global comparison
            if v < global_best:
                x_global_best = Xsym.copy()
                global_best = v
        if time.time() > sym_deadline:
            break

    # release + polish best symmetric descendant more thoroughly
    sym_desc = None
    sym_desc_val = 1e9
    if best_sym_X is not None:
        try:
            y = _run_full_lbfgs(best_sym_X, 2500.0, 120)
            v = _max_cos(y)
            if v < best_sym_val:
                best_sym_X, best_sym_val = y, v
        except Exception:
            pass
        if time.time() < t0 + 75.0:
            rem = max(0.5, min(6.0, (t0 + 75.0) - time.time()))
            try:
                y2 = _lp_polish(best_sym_X, time.time() + rem, r0=5e-4)
                v2 = _max_cos(y2)
                if v2 < best_sym_val:
                    best_sym_X, best_sym_val = y2, v2
            except Exception:
                pass
        sym_desc = best_sym_X.copy()
        sym_desc_val = best_sym_val
        if sym_desc_val < global_best:
            x_global_best = sym_desc.copy()
            global_best = sym_desc_val

    # final intensive squeeze: polish both the symmetric descendant and the loaded best, keep min
    # polish current global best with escalating sharpness + LP
    remain = TLIM - (time.time() - t0)
    if remain > 8:
        try:
            y = _run_full_lbfgs(x_global_best, 3000.0, 130)
            if _max_cos(y) < global_best:
                x_global_best = y; global_best = _max_cos(y)
        except Exception:
            pass
    remain = TLIM - (time.time() - t0)
    if remain > 8:
        try:
            y = _run_full_lbfgs(x_global_best, 9000.0, 110)
            if _max_cos(y) < global_best:
                x_global_best = y; global_best = _max_cos(y)
        except Exception:
            pass
    # also give the loaded record a short polish in case symmetric basin is worse, then take min
    if sym_desc is not None and sym_desc_val > bv + 2e-6:
        # symmetric worse: re-polish original
        try:
            y = _run_full_lbfgs(x_load, 2500.0, 100)
            v = _max_cos(y)
            if v < global_best:
                x_global_best = y; global_best = v
        except Exception:
            pass
    remain = TLIM - (time.time() - t0)
    if remain > 3:
        try:
            x_global_best = _lp_polish(x_global_best, time.time() + (remain - 1.0), r0=3e-4)
            global_best = _max_cos(x_global_best)
        except Exception:
            pass
    x_global_best = x_global_best / np.linalg.norm(x_global_best, axis=1, keepdims=True)
    return x_global_best

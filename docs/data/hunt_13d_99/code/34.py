import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def norm_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def gen_C11(Y):
    nrep, G = 9, 11
    X = np.zeros((nrep * G, 13))
    thetas = 2 * np.pi * np.arange(G)[:, None] * np.arange(1, 6)[None, :] / 11.0
    C = np.cos(thetas); S = np.sin(thetas)
    for g in range(G):
        idx = np.arange(nrep) * G + g
        for p in range(5):
            c, s = C[g, p], S[g, p]
            y0 = Y[:, 2 * p]; y1 = Y[:, 2 * p + 1]
            X[idx, 2 * p] = c * y0 - s * y1
            X[idx, 2 * p + 1] = s * y0 + c * y1
        X[idx, 10:13] = Y[:, 10:13]
    return X

def gen_C9(Y):
    nrep, G = 11, 9
    X = np.zeros((nrep * G, 13))
    thetas = 2 * np.pi * np.arange(G)[:, None] * np.arange(1, 5)[None, :] / 9.0
    C = np.cos(thetas); S = np.sin(thetas)
    for g in range(G):
        idx = np.arange(nrep) * G + g
        for p in range(4):
            c, s = C[g, p], S[g, p]
            y0 = Y[:, 2 * p]; y1 = Y[:, 2 * p + 1]
            X[idx, 2 * p] = c * y0 - s * y1
            X[idx, 2 * p + 1] = s * y0 + c * y1
        X[idx, 8:13] = Y[:, 8:13]
    return X

def accum_C11(gradX):
    nrep, G = 9, 11
    gY = np.zeros((9, 13))
    thetas = 2 * np.pi * np.arange(G)[:, None] * np.arange(1, 6)[None, :] / 11.0
    C = np.cos(thetas); S = np.sin(thetas)
    for g in range(G):
        idx = np.arange(nrep) * G + g
        for p in range(5):
            c, s = C[g, p], S[g, p]
            gx0 = gradX[idx, 2 * p]; gx1 = gradX[idx, 2 * p + 1]
            gY[:, 2 * p] += c * gx0 + s * gx1
            gY[:, 2 * p + 1] += -s * gx0 + c * gx1
        gY[:, 10:13] += gradX[idx, 10:13]
    return gY

def accum_C9(gradX):
    nrep, G = 11, 9
    gY = np.zeros((11, 13))
    thetas = 2 * np.pi * np.arange(G)[:, None] * np.arange(1, 5)[None, :] / 9.0
    C = np.cos(thetas); S = np.sin(thetas)
    for g in range(G):
        idx = np.arange(nrep) * G + g
        for p in range(4):
            c, s = C[g, p], S[g, p]
            gx0 = gradX[idx, 2 * p]; gx1 = gradX[idx, 2 * p + 1]
            gY[:, 2 * p] += c * gx0 + s * gx1
            gY[:, 2 * p + 1] += -s * gx0 + c * gx1
        gY[:, 8:13] += gradX[idx, 8:13]
    return gY

def lse_full_grad(X, alpha):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    E = np.exp(alpha * (G - m))
    np.fill_diagonal(E, 0.0)
    Shalf = E.sum() / 2.0 + 1e-300
    L = m + np.log(Shalf) / alpha
    gradX = (E @ X) / Shalf
    return L, gradX, m

def make_sym_obj(kind, alpha):
    if kind == 'C11':
        nrep = 9
        gen, accum = gen_C11, accum_C11
    else:
        nrep = 11
        gen, accum = gen_C9, accum_C9
    def fun(vflat):
        V = vflat.reshape(nrep, 13)
        nr = np.linalg.norm(V, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y = V / nr
        X = gen(Y)
        L, gradX, _ = lse_full_grad(X, alpha)
        gY = accum(gradX)
        dot = (gY * Y).sum(axis=1, keepdims=True)
        gV = (gY - dot * Y) / nr
        return float(L), gV.ravel().astype(float)
    return fun, nrep

def make_full_obj(alpha):
    def fun(vflat):
        V = vflat.reshape(99, 13)
        nr = np.linalg.norm(V, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y = V / nr
        L, gradX, _ = lse_full_grad(Y, alpha)
        dot = (gradX * Y).sum(axis=1, keepdims=True)
        gV = (gradX - dot * Y) / nr
        return float(L), gV.ravel().astype(float)
    return fun

def run_lbfgs(fun, x0, maxiter):
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12, 'maxfun': maxiter * 5, 'maxcor': 20})
        return res.x, float(res.fun)
    except Exception:
        return x0, float('inf')

def lp_polish_wide(x, seconds):
    x = norm_rows(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 5e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(30.0 * radius, 3e-7)
        sel = c > m - thr
        if int(sel.sum()) < 900:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:900]] = True
            sel = sel | sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            order = np.argsort(c[sel])[-4000:]
            I, J = I[order], J[order]
            p = 4000
            csub = c[sel][order]
        else:
            csub = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(20261103)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        try:
            if xb.ndim == 2:
                m0 = min(xb.shape[0], 99); m1 = min(xb.shape[1], 13)
                tmp[:m0, :m1] = xb[:m0, :m1]
        except Exception:
            pass
        xb = tmp
    xb = norm_rows(xb)
    best_x = xb.copy()
    best_val = max_cos(best_x)
    try:
        xr = np.load("record.npy").astype(float)
        xr = norm_rows(xr)
        rv = max_cos(xr)
        if rv < best_val:
            best_val = rv; best_x = xr.copy()
    except Exception:
        pass
    # ---- symmetric search from scratch ----
    sym_cands = []
    kinds = ['C11', 'C9', 'C11', 'C9', 'C11', 'C9']
    for ti, kind in enumerate(kinds):
        if time.time() - t0 > 70.0:
            break
        fun100, nrep = make_sym_obj(kind, 100.0)
        V0 = rng.standard_normal(size=(nrep * 13,))
        V0 = V0 / (np.linalg.norm(V0.reshape(nrep, 13), axis=1).repeat(13) + 1e-12)
        V0 = V0.astype(float)
        xopt, _ = run_lbfgs(fun100, V0, 250)
        if time.time() - t0 > 75.0:
            pass
        fun400, _ = make_sym_obj(kind, 400.0)
        xopt2, _ = run_lbfgs(fun400, xopt, 200)
        try:
            V = xopt2.reshape(nrep, 13)
            V = norm_rows(V)
            X = gen_C11(V) if kind == 'C11' else gen_C9(V)
            X = norm_rows(X)
            sc = max_cos(X)
            sym_cands.append((sc, X.copy(), kind))
            if sc < best_val:
                best_val = sc; best_x = X.copy()
        except Exception:
            continue
    sym_cands.sort(key=lambda t: t[0])
    # ---- release symmetry: annealed full LogSumExp ----
    release_seeds = []
    for sc, X, k in sym_cands[:2]:
        release_seeds.append(X.copy())
    release_seeds.append(best_x.copy())
    # add one perturbed record seed for diversity if time allows
    full_alphas = [120.0, 250.0, 500.0]
    full_iters = [250, 200, 150]
    for si, seed in enumerate(release_seeds):
        if time.time() - t0 > 128.0:
            break
        cur = norm_rows(seed.copy())
        # small shake to break exact symmetry except for last (record) seed
        if si < len(release_seeds) - 1:
            cur = cur + rng.standard_normal(size=cur.shape) * 0.004
            cur = norm_rows(cur)
        vflat = cur.ravel().astype(float)
        for a, mi in zip(full_alphas, full_iters):
            if time.time() - t0 > 130.0:
                break
            fun = make_full_obj(a)
            vflat, _ = run_lbfgs(fun, vflat, mi)
        try:
            Xf = norm_rows(vflat.reshape(99, 13))
            sc = max_cos(Xf)
            if sc < best_val:
                best_val = sc; best_x = Xf.copy()
        except Exception:
            continue
        # one extra shake + re-optim at mid alpha to try new basin
        if time.time() - t0 < 125.0:
            try:
                cur2 = Xf + rng.standard_normal(size=Xf.shape) * 0.01
                cur2 = norm_rows(cur2)
                fun = make_full_obj(250.0)
                vf2, _ = run_lbfgs(fun, cur2.ravel().astype(float), 150)
                Xf2 = norm_rows(vf2.reshape(99, 13))
                sc2 = max_cos(Xf2)
                if sc2 < best_val:
                    best_val = sc2; best_x = Xf2.copy()
            except Exception:
                pass
    rem = BUDGET - (time.time() - t0)
    rem = max(5.0, min(rem, 42.0))
    try:
        polished = lp_polish_wide(best_x, rem - 0.5)
        pv = max_cos(polished)
        if pv <= best_val + 1e-12:
            # keep polished even if tie within 1e-9 (more rigid)
            if pv < best_val + 1e-9:
                best_x = polished
                best_val = min(best_val, pv)
    except Exception:
        pass
    best_x = norm_rows(best_x)
    return best_x

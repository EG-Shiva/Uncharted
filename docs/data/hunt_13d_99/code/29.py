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

# ---------- symmetric generation ----------
def gen_C11(Y):
    # Y (9,13), returns X (99,13), order j*11+g
    nrep, G = 9, 11
    X = np.zeros((nrep * G, 13))
    thetas = 2 * np.pi * np.arange(G)[:, None] * np.arange(1, 6)[None, :] / 11.0  # (11,5)
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
    # Y (11,13), X (99,13), 4 planes +5 fixed
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
        for p in range(4 - 4):  # noop
            pass
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
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = G.max()
    E = np.exp(alpha * (G - m))
    np.fill_diagonal(E, 0.0)
    # sum over i<j
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
    d = 13
    def fun(vflat):
        V = vflat.reshape(nrep, d)
        nr = np.linalg.norm(V, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y = V / nr
        X = gen(Y)
        L, gradX, _ = lse_full_grad(X, alpha)
        gY = accum(gradX)
        # chain Y->V
        dot = (gY * Y).sum(axis=1, keepdims=True)
        gV = (gY - dot * Y) / nr
        return float(L), gV.ravel().astype(float)
    return fun

def make_full_obj(alpha):
    n, d = 99, 13
    def fun(vflat):
        V = vflat.reshape(n, d)
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
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12, 'maxfun': maxiter * 5})
        return res.x, float(res.fun)
    except Exception:
        return x0, float('inf')

def lp_polish(x, seconds, wide=False):
    x = norm_rows(np.asarray(x, dtype=float))
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        if wide:
            thr = max(30 * radius, 2e-7)
        else:
            thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints to avoid huge LP
        if p > 4000:
            # keep tightest 4000
            order = np.argsort(c[sel])[::-1][-4000:]
            # actually want largest c; argsort ascending, take last 4000
            I, J = I[order], J[order]
            p = len(I)
            csel = c[sel][order]
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
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
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(20261103)
    # load incumbent
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
        xb = norm_rows(tmp)
    else:
        xb = norm_rows(xb)
    best_val = max_cos(xb)
    best_x = xb.copy()

    cands = []  # list of (val, X)

    # ---- symmetry-restricted search ----
    # C11 seeds
    try:
        for seed in [11, 22]:
            if time.time() - t0 > 70:
                break
            r = np.random.default_rng(seed)
            V0 = r.standard_normal(size=(9, 13))
            V0 = norm_rows(V0)
            x0 = V0.ravel()
            for alpha, mi in [(100.0, 250), (400.0, 250)]:
                if time.time() - t0 > 75:
                    break
                fun = make_sym_obj('C11', alpha)
                xopt, _ = run_lbfgs(fun, x0, mi)
                x0 = xopt
            Vopt = x0.reshape(9, 13)
            Vopt = norm_rows(Vopt)
            Xsym = gen_C11(Vopt)
            Xsym = norm_rows(Xsym)
            v = max_cos(Xsym)
            cands.append((v, Xsym.copy()))
            if v < best_val:
                best_val = v; best_x = Xsym.copy()
        # C9 seed
        if time.time() - t0 < 95:
            r = np.random.default_rng(33)
            V0 = r.standard_normal(size=(11, 13))
            V0 = norm_rows(V0)
            x0 = V0.ravel()
            for alpha, mi in [(100.0, 250), (400.0, 250)]:
                if time.time() - t0 > 100:
                    break
                fun = make_sym_obj('C9', alpha)
                xopt, _ = run_lbfgs(fun, x0, mi)
                x0 = xopt
            Vopt = x0.reshape(11, 13)
            Vopt = norm_rows(Vopt)
            Xsym = gen_C9(Vopt)
            Xsym = norm_rows(Xsym)
            v = max_cos(Xsym)
            cands.append((v, Xsym.copy()))
            if v < best_val:
                best_val = v; best_x = Xsym.copy()
    except Exception:
        pass

    # sort candidates, keep top 2 for release (even if worse by 1e-3)
    cands.sort(key=lambda t: t[0])
    # always also include incumbent as release seed
    release_seeds = []
    for v, X in cands[:2]:
        # keep if within 0.01 of best (sym seed 1e-4 worse still valuable)
        if v < best_val + 0.01:
            release_seeds.append(X.copy())
    # add incumbent perturbed slightly to allow new basin via release? use incumbent directly
    # release + annealed LogSumExp
    try:
        for rs in release_seeds:
            if time.time() - t0 > 140:
                break
            cur = rs.copy()
            # small symmetry-breaking noise so L-BFGS can leave symmetric subspace
            cur = cur + rng.standard_normal(size=cur.shape) * 1e-4
            cur = norm_rows(cur)
            x0 = cur.ravel()
            for alpha, mi in [(100.0, 120), (250.0, 120), (600.0, 150)]:
                if time.time() - t0 > 145:
                    break
                fun = make_full_obj(alpha)
                xopt, _ = run_lbfgs(fun, x0, mi)
                x0 = xopt
                Xc = norm_rows(x0.reshape(99, 13))
                v = max_cos(Xc)
                if v < best_val:
                    best_val = v; best_x = Xc.copy()
            # wide-active LP on this branch if time allows
            if time.time() - t0 < 150:
                Xc = norm_rows(x0.reshape(99, 13))
                Xp, vp = lp_polish(Xc, min(8.0, max(1.0, 150 - (time.time() - t0))), wide=True)
                if vp < best_val:
                    best_val = vp; best_x = Xp.copy()
    except Exception:
        pass

    # ---- final polish of global best ----
    try:
        remain = BUDGET - (time.time() - t0) - 1.0
        if remain > 3:
            Xp, vp = lp_polish(best_x, min(remain, 25.0), wide=True)
            if vp < best_val:
                best_val = vp; best_x = Xp.copy()
        # extra tight LP
        remain = BUDGET - (time.time() - t0) - 0.5
        if remain > 2:
            Xp2, vp2 = lp_polish(best_x, remain, wide=False)
            if vp2 < best_val:
                best_val = vp2; best_x = Xp2.copy()
    except Exception:
        pass

    best_x = norm_rows(np.asarray(best_x, dtype=float))
    # strict validity: no NaN, unit rows
    best_x = np.nan_to_num(best_x, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    return best_x

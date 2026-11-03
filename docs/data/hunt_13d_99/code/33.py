import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _build_Rs_C11():
    d = 13; m = 11
    Rs = np.zeros((m, d, d))
    for j in range(m):
        R = np.eye(d)
        for p in range(5):
            k = p + 1
            th = 2.0 * np.pi * j * k / 11.0
            c, s = np.cos(th), np.sin(th)
            a, b = 2 * p, 2 * p + 1
            R[a, a] = c; R[a, b] = -s
            R[b, a] = s; R[b, b] = c
        Rs[j] = R
    return Rs

def _build_Rs_C9():
    d = 13; m = 9
    Rs = np.zeros((m, d, d))
    for j in range(m):
        R = np.eye(d)
        for p in range(4):
            k = p + 1
            th = 2.0 * np.pi * j * k / 9.0
            c, s = np.cos(th), np.sin(th)
            a, b = 2 * p, 2 * p + 1
            R[a, a] = c; R[a, b] = -s
            R[b, a] = s; R[b, b] = c
        Rs[j] = R
    return Rs

# precomputed triu for N=99
_NIU = np.triu_indices(99, 1)

def _lse_grad_X(X, alpha):
    G = X @ X.T
    c = G[_NIU]
    m = float(np.max(c))
    e = np.exp(alpha * (c - m))
    S = float(np.sum(e))
    f = m + np.log(S) / alpha
    w = e / S
    W = np.zeros_like(G)
    W[_NIU[0], _NIU[1]] = w
    W[_NIU[1], _NIU[0]] = w
    Grad = W @ X
    return f, Grad

def _sym_fun_factory(Rs, nrep, d, alpha):
    m = Rs.shape[0]
    def fun(Uflat):
        U = Uflat.reshape(nrep, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        V = U / nr
        # X[a,j,d]
        X3 = np.einsum('jdc,ac->ajd', Rs, V)
        X = X3.reshape(nrep * m, d)
        f, GradX = _lse_grad_X(X, alpha)
        G3 = GradX.reshape(nrep, m, d)
        GradV = np.einsum('jdc,ajd->ac', Rs, G3)
        dot = np.sum(GradV * V, axis=1, keepdims=True)
        GradU = (GradV - dot * V) / nr.ravel()[:, None]
        return f, GradU.ravel()
    return fun

def _full_fun_factory(N, d, alpha):
    def fun(Uflat):
        U = Uflat.reshape(N, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = U / nr
        f, GradX = _lse_grad_X(X, alpha)
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        GradU = (GradX - dot * X) / nr
        return f, GradU.ravel()
    return fun

def _run_sym(Rs, nrep, d, U0flat, alpha, maxiter):
    fun = _sym_fun_factory(Rs, nrep, d, alpha)
    try:
        res = minimize(fun, U0flat, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-9, 'maxfun': maxiter * 8, 'maxcor': 12})
        return res.x
    except Exception:
        return U0flat

def _run_full(X0, alpha, maxiter):
    N, d = X0.shape
    U0 = X0.ravel().copy()
    fun = _full_fun_factory(N, d, alpha)
    try:
        res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-9, 'maxfun': maxiter * 8, 'maxcor': 12})
        U = res.x.reshape(N, d)
        return _norm_rows(U)
    except Exception:
        return X0.copy()

def _expand_sym(Uflat, Rs, nrep, d):
    U = Uflat.reshape(nrep, d)
    V = _norm_rows(U)
    X3 = np.einsum('jdc,ac->ajd', Rs, V)
    return _norm_rows(X3.reshape(nrep * Rs.shape[0], d))

def _lp_wide(x, seconds, kmin=900, wmin=3e-7):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 5e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thresh = m - max(30.0 * radius, wmin)
        sel = c > thresh
        if int(np.sum(sel)) < kmin:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:kmin]] = True
            sel = sel | sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if best < _max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    d = 13; N = 99
    rng0 = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng0.standard_normal(size=(N, d))
    xb = _norm_rows(np.asarray(xb, dtype=float))
    if xb.shape != (99, 13):
        tmp = rng0.standard_normal(size=(99, 13))
        xb = _norm_rows(tmp)
    best_val = _max_cos(xb)
    best_x = xb.copy()

    Rs11 = _build_Rs_C11()
    Rs9 = _build_Rs_C9()

    sym_cands = []
    # C11: 9 reps; C9: 11 reps
    configs = []
    for s in range(5):
        configs.append(('c11', Rs11, 9, 1000 + s))
    for s in range(4):
        configs.append(('c9', Rs9, 11, 2000 + s))

    for name, Rs, nrep, seed in configs:
        if time.time() - t0 > 75.0:
            break
        rng = np.random.default_rng(seed)
        U0 = rng.standard_normal(size=(nrep, d))
        # normalize roughly
        U0 = U0 / np.linalg.norm(U0, axis=1, keepdims=True)
        U0f = U0.ravel()
        # alpha 100
        U1 = _run_sym(Rs, nrep, d, U0f, 100.0, 220)
        if time.time() - t0 > 80.0:
            X1 = _expand_sym(U1, Rs, nrep, d)
            sym_cands.append((_max_cos(X1), X1))
            break
        U2 = _run_sym(Rs, nrep, d, U1, 400.0, 220)
        X2 = _expand_sym(U2, Rs, nrep, d)
        sym_cands.append((_max_cos(X2), X2))
        # also keep intermediate
        X1 = _expand_sym(U1, Rs, nrep, d)
        sym_cands.append((_max_cos(X1), X1))

    sym_cands = sorted(sym_cands, key=lambda z: z[0])
    # release symmetry: perturb + annealed full LSE
    rel_cands = []
    nrel = min(3, len(sym_cands))
    sigmas = [0.002, 0.006, 0.015]
    for i in range(nrel):
        if time.time() - t0 > 125.0:
            break
        _, Xs = sym_cands[i]
        for sig in sigmas:
            if time.time() - t0 > 125.0:
                break
            rng = np.random.default_rng(5000 + i * 10 + int(sig * 10000))
            Xp = Xs + rng.standard_normal(size=Xs.shape) * sig
            Xp = _norm_rows(Xp)
            # anneal alphas
            for alpha, mit in [(80.0, 120), (200.0, 120), (450.0, 150)]:
                if time.time() - t0 > 128.0:
                    break
                Xp = _run_full(Xp, alpha, mit)
            v = _max_cos(Xp)
            rel_cands.append((v, Xp.copy()))
            if v < best_val:
                best_val = v
                best_x = Xp.copy()

    # also try releasing best.npy with tiny shake + anneal (one shot) if time
    if time.time() - t0 < 130.0:
        try:
            rng = np.random.default_rng(999)
            Xp = best_x + rng.standard_normal(size=best_x.shape) * 0.003
            Xp = _norm_rows(Xp)
            for alpha, mit in [(150.0, 100), (400.0, 120)]:
                if time.time() - t0 > 140.0:
                    break
                Xp = _run_full(Xp, alpha, mit)
            v = _max_cos(Xp)
            rel_cands.append((v, Xp.copy()))
            if v < best_val:
                best_val = v
                best_x = Xp.copy()
        except Exception:
            pass

    # pick top 2 overall (sym-release + original) for LP polish
    allc = [(best_val, best_x)]
    for v, X in rel_cands:
        allc.append((v, X))
    for v, X in sym_cands[:2]:
        allc.append((v, X))
    allc = sorted(allc, key=lambda z: z[0])[:3]

    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 6.0:
        # polish best with wide LP
        t_lp = remain * 0.65
        try:
            Xp = _lp_wide(allc[0][1], t_lp, kmin=900)
            v = _max_cos(Xp)
            if v < best_val:
                best_val = v
                best_x = Xp
                allc[0] = (v, Xp)
        except Exception:
            pass
        remain2 = BUDGET - (time.time() - t0) - 1.0
        if remain2 > 7.0 and len(allc) > 1:
            try:
                Xq = _lp_wide(allc[1][1], min(remain2 * 0.5, 12.0), kmin=900)
                vq = _max_cos(Xq)
                if vq < best_val:
                    best_val = vq
                    best_x = Xq
            except Exception:
                pass

    best_x = _norm_rows(best_x)
    return best_x

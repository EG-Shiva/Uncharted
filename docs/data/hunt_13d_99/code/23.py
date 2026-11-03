import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _make_group(order, ks):
    ks = np.asarray(ks, dtype=float)
    s_idx = np.arange(order)[:, None]
    k_row = ks[None, :]
    ang = 2.0 * np.pi * s_idx * k_row / float(order)
    return np.cos(ang), np.sin(ang)

def _sym_expand(R, C, S, nfix):
    nr = R.shape[0]
    order, np_ = C.shape
    Rp = R[:, :2*np_].reshape(nr, np_, 2)
    Rf = R[:, 2*np_:]
    Rx = Rp[:, :, 0]
    Ry = Rp[:, :, 1]
    Xe = Rx[:, None, :]
    Ye = Ry[:, None, :]
    Ce = C[None, :, :]
    Se = S[None, :, :]
    Px = Ce * Xe - Se * Ye
    Py = Se * Xe + Ce * Ye
    Pplanes = np.stack([Px, Py], axis=-1)
    Pplanes = Pplanes.reshape(nr * order, np_ * 2)
    if nfix > 0:
        Pfixed = np.repeat(Rf[:, None, :], order, axis=1).reshape(nr * order, nfix)
        P = np.concatenate([Pplanes, Pfixed], axis=1)
    else:
        P = Pplanes
    return P

def _sym_grad_pullback(Grad_P, C, S, nr, nfix):
    order, np_ = C.shape
    d = 2 * np_ + nfix
    GP = Grad_P.reshape(nr, order, d)
    Gpl = GP[:, :, :2*np_].reshape(nr, order, np_, 2)
    Gx = Gpl[:, :, :, 0]
    Gy = Gpl[:, :, :, 1]
    Ce = C[None, :, :]
    Se = S[None, :, :]
    Bx_s = Ce * Gx + Se * Gy
    By_s = -Se * Gx + Ce * Gy
    Bx = Bx_s.sum(axis=1)
    By = By_s.sum(axis=1)
    Gp_back = np.stack([Bx, By], axis=-1).reshape(nr, 2*np_)
    if nfix > 0:
        Gf = GP[:, :, 2*np_:].sum(axis=1)
        return np.concatenate([Gp_back, Gf], axis=1)
    else:
        return Gp_back

def _sym_obj_grad(Yflat, C, S, nr, d, nfix, alpha):
    Y = Yflat.reshape(nr, d)
    norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
    R = Y / norms
    P = _sym_expand(R, C, S, nfix)
    Pn = P.shape[0]
    Pnorm = np.linalg.norm(P, axis=1, keepdims=True) + 1e-16
    Pnrm = P / Pnorm
    G = Pnrm @ Pnrm.T
    iu = np.triu_indices(Pn, 1)
    gtri = G[iu]
    m = gtri.max()
    E = alpha * (gtri - m)
    Wtri = np.exp(E)
    ssum = Wtri.sum() + 1e-300
    f = float(m + np.log(ssum) / alpha)
    wtri = Wtri / ssum
    W = np.zeros((Pn, Pn))
    W[iu] = wtri
    W = W + W.T
    Grad_Pn = W @ Pnrm
    # chain through P normalization (P already unit since R unit + orthogonal, but keep)
    # P = expand(R) linear, Pnrm = P/|P| ; |P|=1 so skip
    Grad_R = _sym_grad_pullback(Grad_Pn, C, S, nr, nfix)
    # chain R=Y/|Y|
    dot = (Grad_R * R).sum(axis=1, keepdims=True)
    Grad_Y = (Grad_R - dot * R) / norms
    return f, Grad_Y.ravel()

def _full_obj_grad(Yflat, n, d, alpha):
    Y = Yflat.reshape(n, d)
    norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
    R = Y / norms
    G = R @ R.T
    iu = np.triu_indices(n, 1)
    gtri = G[iu]
    m = gtri.max()
    E = alpha * (gtri - m)
    Wtri = np.exp(E)
    ssum = Wtri.sum() + 1e-300
    f = float(m + np.log(ssum) / alpha)
    wtri = Wtri / ssum
    W = np.zeros((n, n))
    W[iu] = wtri
    W = W + W.T
    Grad_R = W @ R
    dot = (Grad_R * R).sum(axis=1, keepdims=True)
    Grad_Y = (Grad_R - dot * R) / norms
    return f, Grad_Y.ravel()

def _run_sym(Y0, C, S, nr, d, nfix, alpha, maxiter):
    def fun(x):
        return _sym_obj_grad(x, C, S, nr, d, nfix, alpha)
    try:
        res = minimize(fun, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": maxiter * 5, "ftol": 1e-14, "gtol": 1e-8})
        return res.x.reshape(nr, d), float(res.fun)
    except Exception:
        return Y0, 1e9

def _run_full(P0, alphas, maxiter_each):
    n, d = P0.shape
    Y = P0.copy()
    for a in alphas:
        def fun(x, _a=a):
            return _full_obj_grad(x, n, d, _a)
        try:
            res = minimize(fun, Y.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter_each, "maxfun": maxiter_each * 5, "ftol": 1e-14, "gtol": 1e-8})
            Y = res.x.reshape(n, d)
            nrm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
            Y = Y / nrm
        except Exception:
            break
    nrm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
    return Y / nrm

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # subsample worst to keep LP tractable but wide
            idx = np.argsort(c[sel])[-4000:]
            I, J = I[idx], J[idx]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][-4000:] if p == 4000 else -c[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    n, d = 99, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.standard_normal(size=(n, d))
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_x = xb.copy()
    best_v = _max_cos(best_x)

    configs = [
        (11, [1, 2, 3, 4, 5], 9),
        (9, [1, 2, 4, 5], 11),
        (9, [1, 2, 3, 4], 11),
    ]
    sym_cands = []
    # symmetric phase: budget ~75s
    for (order, ks, nr) in configs:
        np_ = len(ks)
        nfix = d - 2 * np_
        if nfix < 0:
            continue
        C, S = _make_group(order, ks)
        for restart in range(2):
            if time.time() - t0 > 78.0:
                break
            Y0 = rng.standard_normal(size=(nr, d))
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            Y0 = Y0 * 1.0
            # alpha 100
            Y1, _ = _run_sym(Y0, C, S, nr, d, nfix, 100.0, 250)
            if time.time() - t0 > 80.0:
                # skip second stage, evaluate directly
                R1 = Y1 / (np.linalg.norm(Y1, axis=1, keepdims=True) + 1e-16)
                P1 = _sym_expand(R1, C, S, nfix)
                P1 /= np.linalg.norm(P1, axis=1, keepdims=True)
                v1 = _max_cos(P1)
                sym_cands.append((v1, P1.copy()))
                continue
            Y2, _ = _run_sym(Y1, C, S, nr, d, nfix, 400.0, 250)
            R2 = Y2 / (np.linalg.norm(Y2, axis=1, keepdims=True) + 1e-16)
            P2 = _sym_expand(R2, C, S, nfix)
            P2 /= np.linalg.norm(P2, axis=1, keepdims=True)
            v2 = _max_cos(P2)
            sym_cands.append((v2, P2.copy()))
            if v2 < best_v:
                best_v = v2
                best_x = P2.copy()
    sym_cands.sort(key=lambda t: t[0])
    # release symmetry + annealed full optimization
    K = min(4, len(sym_cands))
    for k in range(K):
        if time.time() - t0 > 128.0:
            break
        _, P0 = sym_cands[k]
        # break symmetry with small noise
        amp = 0.002 if k < 2 else 0.008
        Pn = P0 + rng.standard_normal(size=P0.shape) * amp
        Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        # annealed alphas 100 -> 600
        if k == 0:
            alphas = [100.0, 250.0, 450.0, 600.0]
            mit = 180
        else:
            alphas = [120.0, 300.0, 550.0]
            mit = 150
        Pf = _run_full(Pn, alphas, mit)
        vf = _max_cos(Pf)
        if vf < best_v:
            best_v = vf
            best_x = Pf.copy()
        # also try a second noise level for the best symmetric seed
        if k == 0 and (time.time() - t0 < 115.0):
            Pn2 = P0 + rng.standard_normal(size=P0.shape) * 0.015
            Pn2 /= np.linalg.norm(Pn2, axis=1, keepdims=True)
            Pf2 = _run_full(Pn2, [100.0, 250.0, 500.0], 150)
            vf2 = _max_cos(Pf2)
            if vf2 < best_v:
                best_v = vf2
                best_x = Pf2.copy()
    # opportunistic anneal of record with tiny shake if time remains
    if time.time() - t0 < 120.0:
        try:
            Pn = best_x + rng.standard_normal(size=best_x.shape) * 0.001
            Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
            Pf = _run_full(Pn, [200.0, 500.0], 120)
            vf = _max_cos(Pf)
            if vf < best_v:
                best_v = vf
                best_x = Pf.copy()
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0)
    remain = max(5.0, min(38.0, remain))
    try:
        polished = _lp_polish(best_x, remain)
        vp = _max_cos(polished)
        if vp < best_v:
            best_x = polished
            best_v = vp
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

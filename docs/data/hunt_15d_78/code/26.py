import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

N, D = 78, 15
NO = 6
ORB = 13  # 6*13=78

IU = np.triu_indices(N, 1)

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _gen_c13(P):
    # P (6,15) -> Y (78,15) unnormalized
    Y = np.empty((N, D))
    for o in range(NO):
        a = P[o, :13]
        b = P[o, 13:15]
        base = o * ORB
        for k in range(ORB):
            Y[base + k, :13] = np.roll(a, k)
            Y[base + k, 13:15] = b
    return Y

def _lse_full_obj_grad(Yflat, beta):
    Y = Yflat.reshape(N, D)
    nrm = np.sqrt(np.sum(Y * Y, axis=1))
    nrm = np.maximum(nrm, 1e-12)
    X = Y / nrm[:, None]
    G = X @ X.T
    g = G[IU]
    m = g.max()
    e = np.exp(beta * (g - m))
    S = e.sum()
    f = m + np.log(S) / beta
    w = e / S
    W = np.zeros((N, N))
    W[IU] = w
    W = W + W.T
    GX = W @ X
    dot = np.sum(GX * X, axis=1)
    GY = (GX - X * dot[:, None]) / nrm[:, None]
    return f, GY.ravel()

def _lse_sym_obj_grad(Pflat, beta):
    P = Pflat.reshape(NO, D)
    Y = _gen_c13(P)
    nrm = np.sqrt(np.sum(Y * Y, axis=1))
    nrm = np.maximum(nrm, 1e-12)
    X = Y / nrm[:, None]
    G = X @ X.T
    g = G[IU]
    m = g.max()
    e = np.exp(beta * (g - m))
    S = e.sum()
    f = m + np.log(S) / beta
    w = e / S
    W = np.zeros((N, N))
    W[IU] = w
    W = W + W.T
    GX = W @ X
    dot = np.sum(GX * X, axis=1)
    GY = (GX - X * dot[:, None]) / nrm[:, None]
    GP = np.zeros((NO, D))
    for o in range(NO):
        base = o * ORB
        s13 = np.zeros(13)
        s2 = np.zeros(2)
        for k in range(ORB):
            v = GY[base + k]
            s13 += np.roll(v[:13], -k)
            s2 += v[13:15]
        GP[o, :13] = s13
        GP[o, 13:15] = s2
    return f, GP.ravel()

def _run_lbfgs_sym(P0, beta, maxiter):
    try:
        r = minimize(lambda p: _lse_sym_obj_grad(p, beta), P0.ravel(),
                     method='L-BFGS-B', jac=True,
                     options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-7, 'maxfun': maxiter * 5})
        P = r.x.reshape(NO, D)
    except Exception:
        P = P0
    Y = _gen_c13(P)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-20
    return P, Y, _max_cos(Y)

def _run_lbfgs_full(X0, beta, maxiter):
    try:
        r = minimize(lambda p: _lse_full_obj_grad(p, beta), X0.ravel(),
                     method='L-BFGS-B', jac=True,
                     options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-7, 'maxfun': maxiter * 5})
        X = r.x.reshape(N, D)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-20
    except Exception:
        X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    return X, _max_cos(X)

def _lp_polish(x, deadline, r_init=0.01, r_max=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    while time.time() < deadline and radius > 1e-13:
        if time.time() + 0.5 > deadline:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        if np.count_nonzero(sel) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.finite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return x, best

def _weight3_init(rng):
    X = np.zeros((N, D))
    s3 = 1.0 / np.sqrt(3.0)
    for i in range(N):
        supp = rng.choice(D, 3, replace=False)
        sgn = rng.choice(np.array([-1.0, 1.0]), 3)
        X[i, supp] = sgn * s3
    # tiny jitter to break discreteness
    X = X + 1e-3 * rng.standard_normal((N, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    # record only as threshold
    try:
        rec = np.load("record.npy")
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
        thresh = _max_cos(rec)
    except Exception:
        thresh = 0.168274222547

    # Phase 1: C13 symmetric screening at low beta
    NSYM = 32
    sym_cands = []
    P_inits = [rng.standard_normal((NO, D)) for _ in range(NSYM)]
    for P0 in P_inits:
        if time.time() > HARD_END - 90:
            break
        _, Xs, v = _run_lbfgs_sym(P0, beta=50, maxiter=60)
        sym_cands.append((v, P0, Xs))
    sym_cands.sort(key=lambda t: t[0])
    # refine top 8 at beta 200
    refined = []
    for _, P0b, _ in sym_cands[:8]:
        if time.time() > HARD_END - 80:
            break
        # warm start from previous best P: re-optimize starting at P0b fitted?
        # use P0b as start (random) -> better: start from optimized P of low-beta run
        # re-run: first get optimized P by re-running quickly? use stored Xs to infer P?
        # Simply start from P0b with higher beta (cheap) - but better start from optimized.
        # We stored only init; redo: run beta50 again to get Popt then beta200
        Popt, _, _ = _run_lbfgs_sym(P0b, beta=50, maxiter=60)
        Popt2, Xs2, v2 = _run_lbfgs_sym(Popt, beta=200, maxiter=100)
        refined.append((v2, Popt2, Xs2))
    if len(refined) == 0:
        refined = sym_cands[:3]
    refined.sort(key=lambda t: t[0])
    # top 3 at beta 800
    top_sym = []
    for _, Popt2, _ in refined[:3]:
        if time.time() > HARD_END - 70:
            break
        Popt3, Xs3, v3 = _run_lbfgs_sym(Popt2, beta=800, maxiter=120)
        top_sym.append((v3, Popt3, Xs3))
    if len(top_sym) == 0:
        top_sym = refined[:3]
    top_sym.sort(key=lambda t: t[0])

    # Phase 2: discrete weight-3 variant, full optimization
    disc_cands = []
    for k in range(12):
        if time.time() > HARD_END - 60:
            break
        X0 = _weight3_init(rng)
        X1, v1 = _run_lbfgs_full(X0, beta=50, maxiter=80)
        disc_cands.append((v1, X1))
    disc_cands.sort(key=lambda t: t[0])
    disc_ref = []
    for _, X1 in disc_cands[:4]:
        if time.time() > HARD_END - 50:
            break
        X2, v2 = _run_lbfgs_full(X1, beta=200, maxiter=100)
        disc_ref.append((v2, X2))
    disc_ref.sort(key=lambda t: t[0])

    # Phase 3: release symmetric best to unconstrained
    rel_cands = []
    for _, _, Xs in top_sym[:3]:
        if time.time() > HARD_END - 40:
            break
        Xc = Xs.copy()
        Xc, v = _run_lbfgs_full(Xc, beta=200, maxiter=120)
        if time.time() > HARD_END - 35:
            rel_cands.append((v, Xc))
            continue
        Xc, v = _run_lbfgs_full(Xc, beta=800, maxiter=120)
        rel_cands.append((v, Xc))
    for _, Xd in disc_ref[:2]:
        if time.time() > HARD_END - 35:
            break
        Xc, v = _run_lbfgs_full(Xd, beta=800, maxiter=120)
        rel_cands.append((v, Xc))
    # also include unreleased sym/disc for safety
    for _, _, Xs in top_sym[:1]:
        rel_cands.append((_max_cos(Xs), Xs))
    for _, Xd in disc_ref[:1]:
        rel_cands.append((_max_cos(Xd), Xd))
    rel_cands.sort(key=lambda t: t[0])
    best_v, best_X = rel_cands[0][0], rel_cands[0][1].copy()

    # Phase 4: LP polish best with remaining time, keep whichever release scores best
    # quick polish of top 2 if time allows, else polish best only
    if time.time() < HARD_END - 2:
        # polish best first
        bx, bv = _lp_polish(best_X, min(HARD_END - 0.4, time.time() + max(8.0, (HARD_END - time.time()) * 0.7)), r_init=0.008, r_max=0.03)
        if bv < best_v:
            best_v, best_X = bv, bx
        # try second candidate briefly if still much time and close
        if len(rel_cands) > 1 and time.time() < HARD_END - 15:
            v2, x2 = rel_cands[1][0], rel_cands[1][1]
            if v2 < best_v + 0.005:
                x2p, v2p = _lp_polish(x2, time.time() + 8.0, r_init=0.008, r_max=0.03)
                if v2p < best_v:
                    best_v, best_X = v2p, x2p
        # final fine squeeze
        if time.time() < HARD_END - 1:
            bx, bv = _lp_polish(best_X, HARD_END - 0.3, r_init=0.001, r_max=0.008)
            if bv < best_v:
                best_v, best_X = bv, bx

    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # strict validity
    best_X = np.nan_to_num(best_X, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best_X = best_X / nr
    return best_X

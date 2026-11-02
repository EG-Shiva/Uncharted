import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix
from scipy.linalg import expm


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lse_loss_grad(Y, beta, iu0, iu1):
    n = Y.shape[0]
    G = Y @ Y.T
    gv = G[iu0, iu1]
    m = float(gv.max())
    e = np.exp(beta * (gv - m))
    S = float(e.sum())
    F = m + np.log(S) / beta
    w = e / S
    W = np.zeros((n, n), dtype=np.float64)
    W[iu0, iu1] = w
    W = W + W.T
    GradY = W @ Y
    return F, GradY, m, w


def _lp_polish_deadline(X, deadline, r_init=1e-3, r_max=0.05):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bestx = X.copy()
    radius = r_init
    iu = np.triu_indices(n, 1)
    while time.time() + 0.3 < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)):
            radius *= 0.5
            continue
        y = y / np.maximum(yn, 1e-12)
        v = _max_cos(y)
        if v < best:
            X, best, bestx = y, v, y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return bestx, best


def _smooth_refine(X0, betas, maxiter_each, deadline):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Xcur = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xflat = Xcur.ravel().copy()
    for beta in betas:
        if time.time() + 0.5 > deadline:
            break
        def fun(g):
            Y0 = g.reshape(n, d)
            nr = np.linalg.norm(Y0, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Y0 / nr
            G = Y @ Y.T
            gv = G[iu0, iu1]
            m = float(gv.max())
            e = np.exp(beta * (gv - m))
            S = float(e.sum())
            F = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GradY = W @ Y
            dot = (GradY * Y).sum(axis=1, keepdims=True)
            GradX = (GradY - dot * Y) / nr
            return float(F), GradX.ravel()
        try:
            res = minimize(fun, xflat, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-7})
            xflat = res.x
        except Exception:
            break
        if time.time() + 0.5 > deadline:
            break
    Xo = xflat.reshape(n, d)
    Xo = Xo / np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
    return Xo


def _skew_from_vec(v):
    S = np.zeros((6, 6))
    k = 0
    for p in range(6):
        for q in range(p + 1, 6):
            S[p, q] = v[k]
            S[q, p] = -v[k]
            k += 1
    return S


def _lowdim_optimize(X_init, mu, Vmat, low_idx, high_idx, moved_idx, budget, t_end):
    n, d = X_init.shape
    iu0, iu1 = np.triu_indices(n, 1)
    C0_rot = (X_init - mu) @ Vmat
    v0 = Vmat[:, 0].copy()
    d0 = 0.0
    d1 = 0.0
    s = np.zeros(15, dtype=np.float64)
    T = np.zeros((n, d), dtype=np.float64)
    md0 = vd0 = 0.0
    md1 = vd1 = 0.0
    ms = np.zeros(15)
    vs = np.zeros(15)
    mT = np.zeros((n, d))
    vT = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    # learning rates
    lr_d, lr_s, lr_t = 0.01, 0.02, 0.01
    betas = [100.0, 300.0, 900.0]
    # split budget across betas
    t_start = time.time()
    for beta in betas:
        # iterations proportional; check time
        while True:
            if time.time() > min(t_end, t_start + budget):
                break
            if time.time() + 0.05 > t_end:
                break
            step += 1
            # forward
            Smat = _skew_from_vec(s)
            try:
                R = expm(Smat)
            except Exception:
                R = np.eye(6) + Smat
            C_rot = C0_rot.copy()
            C_rot[low_idx, 0] += d0
            C_rot[high_idx, 0] += d1
            A6_high = C_rot[high_idx][:, :6].copy()
            # apply relative rotation to high stratum leading block
            # B = A @ R.T
            C_rot[np.ix_(high_idx, np.arange(6))] = A6_high @ R.T
            C = C_rot @ Vmat.T
            X = C + mu
            if len(moved_idx) > 0:
                X[moved_idx] += T[moved_idx]
            rn = np.linalg.norm(X, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Y = X / rn
            F, GradY, mcur, w = _lse_loss_grad(Y, beta, iu0, iu1)
            dot = (GradY * Y).sum(axis=1, keepdims=True)
            gX = (GradY - dot * Y) / rn  # dL/dX
            # grads
            gd0 = float(np.dot(gX[low_idx].sum(axis=0), v0))
            gd1 = float(np.dot(gX[high_idx].sum(axis=0), v0))
            # rotation grad approx M - M^T
            # g_rot for high = gX_high @ Vmat ; take first 6 cols
            Grot_high = gX[high_idx] @ Vmat  # (nh,15)
            Gb = Grot_high[:, :6]
            # M = Gb.T @ A6_high_rot? use current rotated A? use A before rotation for approx
            # use A6_high (pre-rotation base incl shift) and Gb
            M = Gb.T @ A6_high  # 6x6
            Gskew = M - M.T
            gs = np.zeros(15)
            k = 0
            for p in range(6):
                for q in range(p + 1, 6):
                    gs[k] = Gskew[p, q]
                    k += 1
            gT = np.zeros((n, d))
            if len(moved_idx) > 0:
                gT[moved_idx] = gX[moved_idx]
            # Adam updates
            md0 = b1 * md0 + (1 - b1) * gd0
            vd0 = b2 * vd0 + (1 - b2) * gd0 * gd0
            md1 = b1 * md1 + (1 - b1) * gd1
            vd1 = b2 * vd1 + (1 - b2) * gd1 * gd1
            ms = b1 * ms + (1 - b1) * gs
            vs = b2 * vs + (1 - b2) * gs * gs
            mT = b1 * mT + (1 - b1) * gT
            vT = b2 * vT + (1 - b2) * gT * gT
            c1 = 1 - b1 ** step
            c2 = 1 - b2 ** step
            d0 -= lr_d * (md0 / c1) / (np.sqrt(vd0 / c2) + eps)
            d1 -= lr_d * (md1 / c1) / (np.sqrt(vd1 / c2) + eps)
            s -= lr_s * (ms / c1) / (np.sqrt(vs / c2) + eps)
            # clip rotation angles
            s = np.clip(s, -0.5, 0.5)
            T -= lr_t * (mT / c1) / (np.sqrt(vT / c2) + eps)
            # keep T bounded
            if len(moved_idx) > 0:
                nm = np.linalg.norm(T[moved_idx], axis=1, keepdims=True)
                # clip to 0.3
                over = nm > 0.3
                if np.any(over):
                    T[moved_idx] = T[moved_idx] / np.maximum(nm / 0.3, 1.0)
            # clip level shifts
            d0 = float(np.clip(d0, -0.5, 0.5))
            d1 = float(np.clip(d1, -0.5, 0.5))
            # stop conditions per beta: fixed counts to share budget
            # roughly 250 steps per beta max
            if step % 250 == 0:
                break
            # early time check every 50 steps
            if step % 50 == 0 and time.time() > t_start + budget * (betas.index(beta) + 1) / 3.0:
                break
        if time.time() > min(t_end, t_start + budget):
            break
    # final build
    Smat = _skew_from_vec(s)
    try:
        R = expm(Smat)
    except Exception:
        R = np.eye(6) + Smat
    C_rot = C0_rot.copy()
    C_rot[low_idx, 0] += d0
    C_rot[high_idx, 0] += d1
    A6_high = C_rot[high_idx][:, :6].copy()
    C_rot[np.ix_(high_idx, np.arange(6))] = A6_high @ R.T
    C = C_rot @ Vmat.T
    X = C + mu
    if len(moved_idx) > 0:
        X[moved_idx] += T[moved_idx]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def solve():
    t0 = time.time()
    HARD = t0 + 160.0
    LIMIT_SMOOTH = HARD - 20.0
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incumbentV = float(bestV)
    # PCA basis of incumbent
    mu = bestX.mean(axis=0)
    Xc = bestX - mu
    try:
        U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vmat = Vt.T.copy()
    except Exception:
        Vmat = np.eye(15)
    pc1 = Xc @ Vmat[:, 0]
    order = np.argsort(pc1, kind='stable')
    n_low_base = 58
    low_base = set(order[:n_low_base].tolist())
    # enumerate allocations 65/13,61/17,55/23,52/26,49/29 (n_low, n_high)
    allocs = [(65, 13), (61, 17), (55, 23), (52, 26), (49, 29)]
    for (n_low, n_high) in allocs:
        if time.time() + 12.0 > HARD:
            break
        low_idx = order[:n_low].tolist()
        high_idx = order[n_low:].tolist()
        low_set = set(low_idx)
        moved = sorted(list((low_set ^ low_base)))
        moved_idx = np.array(moved, dtype=int) if len(moved) > 0 else np.array([], dtype=int)
        # build X_init: move boundary points' PC1 to destination mean, preserve orthogonal, rescale
        C_rot0 = (bestX - mu) @ Vmat
        # destination levels from staying members
        # original means
        # compute means of pc1 for base groups
        pc_vals = C_rot0[:, 0].copy()
        # for moved points set to destination mean
        if len(moved_idx) > 0:
            # means of staying groups
            stay_low = [i for i in low_idx if i in low_base]
            stay_high = [i for i in high_idx if i not in low_base]
            mean_low = float(pc_vals[stay_low].mean()) if len(stay_low) > 0 else float(pc_vals[low_idx].mean())
            mean_high = float(pc_vals[stay_high].mean()) if len(stay_high) > 0 else float(pc_vals[high_idx].mean())
            for i in moved_idx:
                if i in low_set:
                    C_rot0[i, 0] = mean_low
                else:
                    C_rot0[i, 0] = mean_high
        C = C_rot0 @ Vmat.T
        X_init = C + mu
        X_init = X_init / np.maximum(np.linalg.norm(X_init, axis=1, keepdims=True), 1e-12)
        # low-dim optimize, at most 8s per allocation
        budget = 8.0
        # leave room for smoothing/polish
        remain = HARD - time.time()
        if remain < 10:
            break
        budget = min(budget, remain - 8.0)
        if budget < 1.0:
            continue
        t_end_alloc = min(time.time() + budget, HARD - 6.0)
        try:
            Xcand = _lowdim_optimize(X_init, mu, Vmat,
                                     np.array(low_idx, dtype=int),
                                     np.array(high_idx, dtype=int),
                                     moved_idx, budget, t_end_alloc)
        except Exception:
            continue
        vcand = _max_cos(Xcand)
        if vcand >= 0.175:
            continue
        # full-vector L-BFGS smoothing
        try:
            dl = min(HARD - time.time() - 2.0, 6.0)
            if dl > 1.5:
                Xsm = _smooth_refine(Xcand, betas=[150.0, 500.0], maxiter_each=180,
                                     deadline=time.time() + dl)
            else:
                Xsm = Xcand
        except Exception:
            Xsm = Xcand
        vsm = _max_cos(Xsm)
        if vsm < bestV - 1e-12 and vsm < incumbentV - 1e-12 + 1e-12:
            # only candidates below incumbent get capped 15s LP polish
            # check strictly below incumbent (allow tiny tol)
            if vsm < bestV:
                try:
                    lp_dead = min(HARD - 0.5, time.time() + 15.0)
                    if lp_dead > time.time() + 1.0:
                        Xpol, vpol = _lp_polish_deadline(Xsm, lp_dead)
                    else:
                        Xpol, vpol = Xsm, vsm
                except Exception:
                    Xpol, vpol = Xsm, vsm
                if vpol < bestV:
                    bestX = Xpol.copy()
                    bestV = float(vpol)
            else:
                if vsm < bestV:
                    bestX = Xsm.copy()
                    bestV = float(vsm)
        # also track best even if not below incumbent? spec says only below incumbent gets LP;
        # but keep best fully evaluated: if smoothed is better than best but not incumbent? incumbent==best initially
        # so covered. If we improved once, further improvements also get polish.
        if time.time() > HARD - 2.0:
            break
    bestX = bestX / np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX

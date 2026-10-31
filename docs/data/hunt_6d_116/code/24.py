import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-6):
    G = X @ X.T
    m = np.max(G - 2.0*np.eye(G.shape[0]))
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    sel = np.where(c > m - tol)[0]
    s = set((int(iu[0][k])*1000 + int(iu[1][k])) for k in sel)
    pairs = [(int(iu[0][k]), int(iu[1][k]), float(c[k])) for k in sel]
    return s, pairs, float(m), G

def _jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return inter / union

def _tangent_bases(X, rng):
    n = X.shape[0]
    T = np.empty((n, 6, 5))
    for i in range(n):
        x = X[i]
        M = rng.standard_normal((6, 5))
        M = M - np.outer(x, x @ M)
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _soft_kick(X, T, pairs, G, rng, rms_target):
    n = X.shape[0]
    p = len(pairs)
    N = n * 5
    if p == 0:
        dX = rng.standard_normal(X.shape)
        dX -= (dX * X).sum(axis=1, keepdims=True) * X
        rms = np.sqrt(np.mean((dX**2).sum(axis=1))) + 1e-15
        return _norm(X + dX / rms * rms_target)
    R = np.zeros((p, N))
    for q, (i, j, c) in enumerate(pairs):
        vi = X[j] - c * X[i]
        vj = X[i] - c * X[j]
        ci = T[i].T @ vi
        cj = T[j].T @ vj
        R[q, i*5:(i+1)*5] = ci
        R[q, j*5:(j+1)*5] = cj
    try:
        Gram = R.T @ R
        # symmetrize
        Gram = 0.5 * (Gram + Gram.T)
        w, V = np.linalg.eigh(Gram)
        # V columns ascending
    except Exception:
        dX = rng.standard_normal(X.shape)
        dX -= (dX * X).sum(axis=1, keepdims=True) * X
        rms = np.sqrt(np.mean((dX**2).sum(axis=1))) + 1e-15
        return _norm(X + dX / rms * rms_target)
    K = int(rng.integers(20, 41))
    K = min(K, N)
    Vsmall = V[:, :K]
    coeff = rng.standard_normal(K)
    # weight by 1/(eig+eps) to favor softer? use uniform as spec says random combo
    uflat = Vsmall @ coeff
    U = uflat.reshape(n, 5)
    dX = np.einsum('ijk,ik->ij', T, U)
    # dual-force separation component: top S closest pairs
    S = int(rng.integers(5, 11))
    order = np.argsort([-c for (_, _, c) in pairs])
    sep = np.zeros_like(X)
    for t in range(min(S, len(order))):
        i, j, c = pairs[order[t]]
        ti = X[j] - c * X[i]
        tj = X[i] - c * X[j]
        ni = float(np.linalg.norm(ti)) + 1e-15
        nj = float(np.linalg.norm(tj)) + 1e-15
        sep[i] += -ti / ni
        sep[j] += -tj / nj
    # project sep tangent
    sep -= (sep * X).sum(axis=1, keepdims=True) * X
    rms_soft = np.sqrt(np.mean((dX**2).sum(axis=1))) + 1e-15
    dX = dX / rms_soft * rms_target
    rms_sep = np.sqrt(np.mean((sep**2).sum(axis=1))) + 1e-15
    if rms_sep > 1e-12 and np.all(np.isfinite(sep)):
        gamma = float(rng.uniform(0.2, 0.6))
        sep = sep / rms_sep * (rms_target * gamma)
        # random sign? always separating, add
        dX = dX + sep
        # rescale total to rms_target*~1.0 to keep 0.05-0.25 range
        rms_tot = np.sqrt(np.mean((dX**2).sum(axis=1))) + 1e-15
        dX = dX / rms_tot * rms_target * float(rng.uniform(0.9, 1.15))
    Y = X + dX
    return _norm(Y)

def _hinge_adam(X, t_target, iters, lr, rng=None):
    X = _norm(X.copy())
    n = X.shape[0]
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters+1):
        G = X @ X.T
        Rmat = G - t_target
        np.fill_diagonal(Rmat, 0.0)
        Rmat[Rmat < 0] = 0.0
        Grad = 2.0 * (Rmat @ X)
        proj = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - proj * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * Gt * Gt
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # clamp step norm per point to avoid blowup
        X = X - step
        X = _norm(X)
    return X

def _softmax_adam(X, iters, a0, a1, lr0, lr1):
    X = _norm(X.copy())
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters+1):
        frac = it / float(iters)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Grad = W @ X
        proj = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - proj * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X = _norm(X)
    return X

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bestX = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if len(idx) > 1600:
            order = np.argsort(c[idx])[-1600:]
            idx = idx[order]
        if len(idx) == 0:
            break
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX

def solve():
    t0 = time.time()
    BUDGET = 172.0
    def _clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nrm = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(nrm < 1e-12):
            return None
        return a / nrm
    cands = []
    try:
        cands.append(_clean(np.load("best.npy")))
    except Exception:
        pass
    try:
        cands.append(_clean(np.load("record.npy")))
    except Exception:
        pass
    cands = [c for c in cands if c is not None]
    if not cands:
        rng0 = np.random.default_rng(0)
        cands.append(_norm(rng0.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = x_best.copy()
    cur_val = best_val
    rng = np.random.default_rng(11606)
    # initial basin contacts
    pre_set_all, _, _, _ = _contacts(cur_x, tol=1e-6)
    explore_end = t0 + BUDGET - 30.0
    it = 0
    T_cache = None
    # temperature for basin exploration
    temp0, temp1 = 0.0015, 0.0001
    while time.time() < explore_end:
        it += 1
        frac = min(1.0, (time.time() - t0) / max(explore_end - t0, 1e-9))
        temp = temp0 + (temp1 - temp0) * frac
        # contacts of current basin
        cset, pairs, mcur, Gcur = _contacts(cur_x, tol=1e-6)
        # build tangent bases (fresh rng draws for diversity)
        try:
            T = _tangent_bases(cur_x, rng)
        except Exception:
            T = None
        if T is None or len(pairs) == 0:
            sigma = float(rng.uniform(0.05, 0.2))
            kicked = _norm(cur_x + sigma * rng.standard_normal(cur_x.shape))
        else:
            rms_target = float(rng.uniform(0.05, 0.25))
            try:
                kicked = _soft_kick(cur_x, T, pairs, Gcur, rng, rms_target)
            except Exception:
                sigma = rms_target
                kicked = _norm(cur_x + sigma * rng.standard_normal(cur_x.shape))
        # feasibility phase toward 0.5958
        h_iters = int(rng.integers(500, 1001))
        # adapt to save time late
        if time.time() > explore_end - 10:
            h_iters = 400
        h_lr = float(rng.uniform(0.006, 0.014))
        try:
            trial = _hinge_adam(kicked, 0.5958, h_iters, h_lr)
        except Exception:
            trial = kicked
        # annealed softmax
        try:
            trial = _softmax_adam(trial, 300, 30, 220, 0.02, 0.002)
        except Exception:
            pass
        # Jaccard gate vs pre-kick basin
        try:
            post_set, _, _, _ = _contacts(trial, tol=1e-6)
            jac = _jaccard(cset, post_set)
        except Exception:
            jac = 0.0
        if jac >= 0.9:
            # same basin, slide back -> discard, occasionally force bigger kick next time
            # still allow if it improved best (polish slipped)
            v_tmp = _maxcos(trial)
            if v_tmp < best_val - 1e-12:
                best_val = v_tmp
                x_best = trial.copy()
                cur_x = trial.copy()
                cur_val = v_tmp
            continue
        v = _maxcos(trial)
        # short LP polish if promising and diverse
        if v < best_val + 0.003 and (time.time() < explore_end - 2):
            try:
                tp = _lp_polish(trial, 0.9)
                vp = _maxcos(tp)
            except Exception:
                tp, vp = trial, v
            trial, v = tp, vp
            try:
                post_set2, _, _, _ = _contacts(trial, tol=1e-6)
                jac2 = _jaccard(cset, post_set2)
            except Exception:
                jac2 = jac
            # require still diverse after polish? if it snapped back, discard
            if jac2 >= 0.92:
                if v < best_val - 1e-12:
                    best_val = v
                    x_best = trial.copy()
                continue
        if v < best_val - 1e-12:
            best_val = v
            x_best = trial.copy()
            cur_x = trial.copy()
            cur_val = v
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(temp, 1e-12)):
            if v < cur_val + 0.02:
                cur_x = trial.copy()
                cur_val = v
        if it % 60 == 0 and time.time() < explore_end:
            # occasional return to best but with fresh soft kick (keeps topology change requirement)
            if rng.random() < 0.4:
                cur_x = x_best.copy()
                cur_val = best_val
        if time.time() >= explore_end:
            break
    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 4:
        try:
            x_best = _lp_polish(x_best, remain)
        except Exception:
            x_best = _norm(x_best)
    else:
        x_best = _norm(x_best)
    x_best = _norm(x_best)
    return x_best

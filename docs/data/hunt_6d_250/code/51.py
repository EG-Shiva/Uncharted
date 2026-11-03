import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    T = np.empty((n, d, d - 1), dtype=np.float64)
    Efull = np.eye(d, dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        E = np.delete(Efull, k, axis=1)  # d x (d-1)
        coeff = x @ E  # (d-1,)
        Ep = E - np.outer(x, coeff)
        Q, _ = np.linalg.qr(Ep)
        # ensure orthogonal to x (fix sign drift)
        # Q span is Ep span which is orthogonal to x
        T[i] = Q
    return T

def _adam_polish(X0, steps=300, beta0=500.0, beta1=5000.0, lr=0.004):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    grow = (beta1 / beta0) ** (1.0 / max(1, steps))
    beta = beta0
    for t in range(1, steps + 1):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = (G - m) * beta
        np.clip(D, -30.0, 0.0, out=D)
        W = np.exp(D)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        Gw = W @ X
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        g = 2.0 * (Gw - dot * X)
        # Adam
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        # project step to tangent to avoid radial waste
        sd = np.sum(step * X, axis=1, keepdims=True)
        step = step - sd * X
        # clip step norm per point for stability
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 0.05
        f = np.minimum(1.0, cap / np.maximum(sn, 1e-12))
        step = step * f
        X = X - step
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        beta *= grow
        if not np.all(np.isfinite(X)):
            break
    return X

def _lp_polish(x, deadline, k_top=4500, radius_init=2.0e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0, iu1 = np.triu_indices(n, 1)
    ntri = iu0.shape[0]
    K = min(k_top, ntri)
    while time.time() < deadline - 0.5 and radius > 1e-13:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        if K < ntri:
            part = np.argpartition(-c_all, K)[:K]
            I = iu0[part]; J = iu1[part]; c_sel = c_all[part]
        else:
            I = iu0; J = iu1; c_sel = c_all
        p = I.shape[0]
        nv = n * d + 1
        cols_blocks = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols_blocks)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        val = _max_cos(y)
        if val < best - 1e-12:
            x, best = y, val
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x

def _adj_set(X, tol=1e-4):
    G = X @ X.T
    iu0, iu1 = np.triu_indices(X.shape[0], 1)
    c = G[iu0, iu1]
    m = float(np.max(c))
    sel = c >= m - tol
    return set((iu0[sel] * X.shape[0] + iu1[sel]).tolist()), m

def _jacc_dist(s0, s1):
    if not s0 and not s1:
        return 0.0
    inter = len(s0 & s1)
    union = len(s0 | s1)
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def solve():
    t0 = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t0)
    n, d = 250, 6
    best = None; best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                v = _max_cos(z)
                if v < best_val:
                    best_val = v; best = z.copy()
        except Exception:
            continue
    if best is None:
        rng0 = np.random.default_rng(0)
        best = rng0.standard_normal((n, d))
        best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
        best_val = _max_cos(best)
    X0 = best.copy(); m0 = best_val
    adj0, _ = _adj_set(X0, tol=1e-4)

    # ---- build plateau Jacobian A ----
    try:
        G0 = X0 @ X0.T
        iu0_all, iu1_all = np.triu_indices(n, 1)
        c_all = G0[iu0_all, iu1_all]
        m_all = float(np.max(c_all))
        for thr in (2e-6, 5e-6, 1e-5, 2e-5):
            sel = c_all >= m_all - thr
            cnt = int(np.count_nonzero(sel))
            if cnt >= 800:
                break
        if cnt < 300:
            sel = c_all >= m_all - 5e-5
            cnt = int(np.count_nonzero(sel))
        # cap to keep SVD tractable
        if cnt > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:3500]
            keep = np.zeros_like(sel); keep[idx[order]] = True
            sel = keep
            cnt = 3500
        I = iu0_all[sel]; J = iu1_all[sel]
        p = cnt
        T = _tangent_bases(X0)  # n x 6 x 5
        dd = d - 1
        A = np.zeros((p, n * dd), dtype=np.float64)
        # Tt_x: for speed precompute? loop
        for s in range(p):
            i = int(I[s]); j = int(J[s])
            A[s, i*dd:(i+1)*dd] = T[i].T @ X0[j]
            A[s, j*dd:(j+1)*dd] = T[j].T @ X0[i]
        # smallest right singular vectors via ATA eig
        ATA = A.T @ A
        # regularize tiny
        w, V = np.linalg.eigh(ATA)
        # w ascending; take 30 smallest
        Kmode = min(30, V.shape[1])
        Modes = V[:, :Kmode]  # 1250 x K
    except Exception:
        Modes = None; T = None

    cands_pre = []
    if Modes is not None:
        try:
            dd = d - 1
            Kmode = Modes.shape[1]
            step_sizes = (0.02, 0.05, 0.10, 0.20)
            # precompute per-mode per-point norms/directions
            for k in range(Kmode):
                if rem() < 90:
                    break
                v = Modes[:, k]  # 1250
                C = v.reshape(n, dd)  # coeffs
                U = np.einsum('nij,nj->ni', T, C)  # tangent vecs 250x6
                r = np.linalg.norm(U, axis=1)
                rmax = float(np.max(r))
                if rmax < 1e-12:
                    continue
                That = np.zeros_like(U)
                nz = r > 1e-14
                That[nz] = U[nz] / r[nz, None]
                for sign in (1.0, -1.0):
                    Uh = sign * That
                    for S in step_sizes:
                        theta = S * r / rmax  # max = S
                        ct = np.cos(theta)[:, None]
                        st = np.sin(theta)[:, None]
                        Xn = ct * X0 + st * Uh
                        # for zero-norm points Uh=0 -> Xn = ct*X0, renorm fixes
                        Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
                        G = Xn @ Xn.T
                        np.fill_diagonal(G, -2.0)
                        mc = float(np.max(G))
                        cands_pre.append((mc, k, sign, S, Xn))
        except Exception:
            pass

    # rank pre-candidates by max cosine
    cands_pre.sort(key=lambda t: t[0])
    # compute adjacency difference and filter >=5%
    filtered = []
    if cands_pre:
        # evaluate adjacency only for top 80 to save time
        for (mc, k, sign, S, Xn) in cands_pre[:80]:
            try:
                s, _ = _adj_set(Xn, tol=1e-4)
                jd = _jacc_dist(adj0, s)
            except Exception:
                jd = 0.0
            filtered.append((mc, jd, k, sign, S, Xn))
    # keep diverse: jd >= 0.05 first
    diverse = [f for f in filtered if f[1] >= 0.05]
    if len(diverse) < 6:
        # relax to 0.02
        diverse = [f for f in filtered if f[1] >= 0.02]
    if not diverse:
        diverse = filtered[:12]
    # take top ~12 by mc among diverse (already sorted by mc)
    diverse.sort(key=lambda t: t[0])
    shortlist = diverse[:12]

    polished = []
    for (mc, jd, k, sign, S, Xn) in shortlist:
        if rem() < 60:
            break
        try:
            Y = _adam_polish(Xn, steps=300, beta0=500.0, beta1=5000.0, lr=0.004)
            v = _max_cos(Y)
            try:
                s, _ = _adj_set(Y, tol=1e-4)
                jd2 = _jacc_dist(adj0, s)
            except Exception:
                jd2 = jd
            polished.append((v, jd2, Y))
            if v < best_val - 1e-12:
                best_val = v; best = Y.copy()
        except Exception:
            continue

    # choose two most distinct improvers for LP polish
    # improvers = polished with v < m0; if <2, take best polished
    polished.sort(key=lambda t: t[0])
    improvers = [q for q in polished if q[0] < m0 - 1e-12]
    if len(improvers) >= 2:
        improvers.sort(key=lambda t: -t[1])  # most distinct first
        lp_targets = [improvers[0][2], improvers[1][2]]
    elif len(polished) >= 2:
        # most distinct among best few
        cand = polished[:4]
        cand.sort(key=lambda t: -t[1])
        lp_targets = [cand[0][2], cand[1][2] if len(cand) > 1 else cand[0][2]]
    elif len(polished) == 1:
        lp_targets = [polished[0][2]]
    else:
        lp_targets = []

    for Xt in lp_targets:
        if rem() < 12:
            break
        try:
            Y = _lp_polish(Xt, time.time() + rem() - 1.0)
            v = _max_cos(Y)
            if v < best_val:
                best_val = v; best = Y.copy()
        except Exception:
            continue

    # final LP squeeze on global best if time remains
    try:
        if rem() > 10:
            best = _lp_polish(best, time.time() + rem() - 1.0)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best

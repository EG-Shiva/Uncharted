import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _hinge_quench(X, iters=150, lr0=0.02, band=0.004):
    X = _norm(X.copy())
    n = X.shape[0]
    lr = lr0
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        thr = M - band
        Mk = G > thr
        A = Mk @ X
        w = np.sum(Mk * G, axis=1, keepdims=True)
        Gt = A - w * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn < 1e-14 or not np.isfinite(gn):
            break
        mx = 0.03
        step = lr
        if gn * step > mx:
            step = mx / gn
        X = X - step * Gt
        X = _norm(X)
        lr *= 0.995
    return X

def _lp_polish(X, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X, _maxcos(X)
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.5:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu0[0][sel]; J = iu0[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
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
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    import time as _tm
    t0 = _tm.time()
    HARD = t0 + 168.0
    rng = np.random.default_rng(0)
    # load incumbent
    def _load(f):
        try:
            a = np.load(f)
            a = np.asarray(a, dtype=float)
            if a.shape != (116, 6) or not np.all(np.isfinite(a)):
                return None
            nn = np.linalg.norm(a, axis=1, keepdims=True)
            if np.any(nn < 1e-12):
                return None
            return a / nn
        except Exception:
            return None
    Xb = _load("best.npy")
    Xr = _load("record.npy")
    cands0 = []
    if Xb is not None:
        cands0.append(Xb)
    if Xr is not None:
        cands0.append(Xr)
    if len(cands0) == 0:
        Xf = rng.standard_normal((116, 6))
        Xf = _norm(Xf)
    else:
        vals = [_maxcos(X) for X in cands0]
        Xf = cands0[int(np.argmin(vals))].copy()
    X0 = _norm(Xf.copy())
    cmax0 = _maxcos(X0)
    best_val = cmax0
    best_X = X0.copy()
    try:
        n, d = 116, 6
        td = d - 1
        N = n * td
        G0 = X0 @ X0.T
        iu = np.triu_indices(n, 1)
        c = G0[iu]
        order = np.argsort(-c)
        csorted = c[order]
        # active set slack < 1e-9
        slack = cmax0 - csorted
        m_active = int(np.sum(slack < 1e-9))
        if m_active < 400:
            m_active = min(560, len(csorted))
            # take tightest m_active
        if m_active > 650:
            m_active = 650
        m = m_active
        sel_pairs = order[:m]
        Ii = iu[0][sel_pairs]
        Ji = iu[1][sel_pairs]
        # tangent bases via QR (orthonormal, one per point)
        B = np.empty((n, d, td))
        for i in range(n):
            xi = X0[i]
            k = int(np.argmax(np.abs(xi)))
            cols = [j for j in range(d) if j != k]
            A = np.eye(d)[:, cols]  # 6x5
            coeff = xi @ A  # (5,)
            Ap = A - np.outer(xi, coeff)
            Q, _ = np.linalg.qr(Ap)
            B[i] = Q
        # Jacobian m x N
        J = np.zeros((m, N), dtype=float)
        # vectorize using loop (m~540 fine)
        for p in range(m):
            i = int(Ii[p]); j = int(Ji[p])
            J[p, i*td:(i+1)*td] = B[i].T @ X0[j]
            J[p, j*td:(j+1)*td] = B[j].T @ X0[i]
        # single dense SVD
        U, S, Vt = np.linalg.svd(J, full_matrices=True)
        # rank
        if S.size == 0 or not np.all(np.isfinite(S)):
            raise RuntimeError("svd bad")
        tol = max(1e-7, S[0] * 1e-9)
        rank = int(np.sum(S > tol))
        rank = min(rank, m)
        # null basis rows Vt[rank:]
        null_dim = N - rank
        if null_dim < 5:
            raise RuntimeError("no nullspace")
        # rotation subspace removal
        Rmat = np.zeros((N, 15), dtype=float)
        col = 0
        for pp in range(d):
            for qq in range(pp+1, d):
                for i in range(n):
                    xi = X0[i]
                    dx = np.zeros(d)
                    dx[pp] = xi[qq]
                    dx[qq] = -xi[pp]
                    # dx = A @ xi with A=Eppqq - Eqqpp? check: (A x)_p = x_q, (A x)_q=-x_p
                    Rmat[i*td:(i+1)*td, col] = B[i].T @ dx
                col += 1
        Qr, _ = np.linalg.qr(Rmat)
        def _proj(u):
            return u - Qr @ (Qr.T @ u)
        # collect directions
        dirs = []
        # lowest non-zero singular vectors (up to 10)
        nlow = min(10, rank)
        for k in range(nlow):
            v = Vt[rank-1-k, :].copy()
            v = _proj(v)
            nn = float(np.linalg.norm(v))
            if nn > 1e-12 and np.isfinite(nn):
                dirs.append(v / nn)
        # random null combos
        null_basis = Vt[rank:, :]  # (null_dim x N)
        # project null basis orthogonal to rotations implicitly via _proj on combos
        n_rand = 50
        for _ in range(n_rand):
            coeff = rng.standard_normal(null_dim)
            v = coeff @ null_basis
            v = _proj(v)
            nn = float(np.linalg.norm(v))
            if nn > 1e-12 and np.isfinite(nn):
                dirs.append(v / nn)
            if _tm.time() > HARD - 120:
                break
        if len(dirs) == 0:
            raise RuntimeError("no dirs")
        # line search scoring true max + 600th
        K = 600
        # baseline Kth
        cs0 = np.sort(c)[::-1]
        ck0 = float(cs0[K-1]) if len(cs0) >= K else float(cs0[-1])
        steps = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], dtype=float)
        # pre-reshape B for fast apply: B (n,6,5)
        scored_moves = []
        for di, vv in enumerate(dirs):
            if _tm.time() > HARD - 110:
                break
            Umat = vv.reshape(n, td)  # per-point tangent coeffs
            # per-point displacement in R6 (unit dir): D_i = B_i @ Umat_i
            D = np.einsum('nij,nj->ni', B, Umat)
            for s in steps:
                for sgn in (1.0, -1.0):
                    alpha = sgn * s
                    Y = X0 + alpha * D
                    Y = _norm(Y)
                    G = Y @ Y.T
                    np.fill_diagonal(G, -2.0)
                    cmx = float(np.max(G))
                    # kth largest via partition on upper tri
                    tri = G[iu]
                    # partial: use partition for speed
                    kth = float(-np.partition(-tri, K-1)[K-1])
                    # keep if not raising max too much and reduces shell
                    if cmx <= cmax0 + 1e-7 and kth < ck0 - 1e-9:
                        scored_moves.append((kth, cmx, alpha, di, Y))
                    if _tm.time() > HARD - 105:
                        break
                if _tm.time() > HARD - 105:
                    break
        # select best few by (cmax, kth)
        if len(scored_moves) > 0:
            scored_moves.sort(key=lambda t: (t[1], t[0]))
            # dedupe / take top 4 with distinct dirs
            seen = set()
            top = []
            for kth, cmx, alpha, di, Y in scored_moves:
                if di not in seen:
                    top.append((kth, cmx, alpha, di, Y))
                    seen.add(di)
                if len(top) >= 4:
                    break
            # hinge quench each
            for kth, cmx, alpha, di, Y in top:
                if _tm.time() > HARD - 90:
                    break
                Yq = _hinge_quench(Y, iters=120, lr0=0.02, band=0.004)
                vq = _maxcos(Yq)
                if vq < best_val:
                    best_val = vq
                    best_X = Yq.copy()
            # also try quench of best shell reducer regardless of max
            scored_moves.sort(key=lambda t: t[0])
            Ybest_shell = scored_moves[0][4]
            if _tm.time() < HARD - 95:
                Yq2 = _hinge_quench(Ybest_shell, iters=150, lr0=0.015, band=0.006)
                vq2 = _maxcos(Yq2)
                if vq2 < best_val:
                    best_val = vq2
                    best_X = Yq2.copy()
        # single trust-region LP re-jam on current best (with time budget)
        remain = HARD - _tm.time()
        if remain > 8.0:
            budget = min(remain - 2.0, 40.0)
            Xp, vp = _lp_polish(best_X, _tm.time() + budget)
            if vp < best_val and np.all(np.isfinite(Xp)):
                best_val = vp
                best_X = Xp.copy()
        # final safety: if best not better, keep incumbent (still valid)
        if _maxcos(best_X) > cmax0 + 1e-12:
            best_X = X0.copy()
    except Exception:
        best_X = X0.copy()
    best_X = _norm(best_X)
    # strict finite check
    if not np.all(np.isfinite(best_X)):
        best_X = X0.copy()
    return best_X

import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import svds
from scipy.optimize import linprog

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_B(X):
    n, d = X.shape
    k = d - 1
    B = np.zeros((n, d, k))
    for i in range(n):
        x = X[i]
        amax = int(np.argmax(np.abs(x)))
        cols = []
        for j in range(d):
            if j == amax:
                continue
            e = np.zeros(d)
            e[j] = 1.0
            v = e - float(e @ x) * x
            cols.append(v)
        M = np.stack(cols, axis=1)  # 6x5
        Q, _ = np.linalg.qr(M)
        B[i] = Q
    return B

def _lp_polish(x_init, deadline, t0, BUDGET, iu0, iu1, r_init=1.0e-3, max_steps=25):
    n, d = x_init.shape
    x = _norm(x_init.copy())
    best = _max_cos(x)
    bx = x.copy()
    radius = r_init
    steps = 0
    while time.time() < deadline and (time.time() - t0) < BUDGET - 1.0 and steps < max_steps:
        steps += 1
        if radius < 1e-14:
            break
        Gf = x @ x.T
        c = Gf[iu0, iu1]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[sel]; J = iu1[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            kk = 3500
            part = np.argpartition(-csel, kk - 1)[:kk]
            I, J, csel = I[part], J[part], csel[part]
            p = kk
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cA = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vA, (rows, cA)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            bx = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if p < 10 and radius < 1e-6:
            break
    return bx, best

def solve():
    t0 = time.time()
    BUDGET = 170.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        rng0 = np.random.RandomState(0)
        x_best = _norm(rng0.randn(n, d))
        best_val = _max_cos(x_best)

    X0 = x_best.copy()
    m0 = best_val

    # identify rattler: fewest contacts within 1e-9
    try:
        G0 = X0 @ X0.T
        np.fill_diagonal(G0, -2.0)
        cnt0 = np.sum(G0 > m0 - 1e-9, axis=1)
        rattler = int(np.argmin(cnt0))
    except Exception:
        rattler = int(np.argmin(np.max(X0 @ X0.T - 2*np.eye(n), axis=1)))

    # tangent bases
    try:
        B = _build_B(X0)
    except Exception:
        return _norm(x_best)

    # pairs above maxcos-1e-6
    try:
        Gf = X0 @ X0.T
        c_all = Gf[iu0, iu1]
        thrA = m0 - 1e-6
        selA = c_all > thrA
        IA = iu0[selA]; JA = iu1[selA]
        m_pairs = len(IA)
        if m_pairs < 50 or m_pairs > 8000:
            # adjust window to get reasonable size
            order = np.argsort(-c_all)
            m_pairs = int(np.clip(1100, 50, len(c_all)))
            pick = order[:m_pairs]
            IA = iu0[pick]; JA = iu1[pick]
            m_pairs = len(IA)
    except Exception:
        return _norm(x_best)

    # column map freezing rattler
    idx_map = np.full(n, -1, dtype=int)
    cur = 0
    for i in range(n):
        if i == rattler:
            continue
        idx_map[i] = cur
        cur += 1
    Np = (n - 1) * 5

    # build sparse Jacobian A (m_pairs x Np)
    rows_l = []
    cols_l = []
    data_l = []
    for k in range(m_pairs):
        i = int(IA[k]); j = int(JA[k])
        if i != rattler:
            bi = idx_map[i] * 5
            vec = X0[j] @ B[i]  # shape (5,)
            for t in range(5):
                rows_l.append(k); cols_l.append(bi + t); data_l.append(float(vec[t]))
        if j != rattler:
            bj = idx_map[j] * 5
            vec2 = X0[i] @ B[j]
            for t in range(5):
                rows_l.append(k); cols_l.append(bj + t); data_l.append(float(vec2[t]))
    try:
        A = coo_matrix((np.array(data_l), (np.array(rows_l), np.array(cols_l))), shape=(m_pairs, Np)).tocsr()
    except Exception:
        return _norm(x_best)

    # rotation modes (Np x 15)
    rots = []
    for a in range(d):
        for b in range(a + 1, d):
            rots.append((a, b))
    nrot = len(rots)  # 15
    Rmat = np.zeros((Np, nrot))
    for kk, (a, b) in enumerate(rots):
        for i in range(n):
            if i == rattler:
                continue
            bi = idx_map[i] * 5
            xi = X0[i]
            dd = np.zeros(d)
            dd[a] = xi[b]
            dd[b] = -xi[a]
            v = B[i].T @ dd
            Rmat[bi:bi + 5, kk] = v
    try:
        Qrot, _ = np.linalg.qr(Rmat)
    except Exception:
        Qrot = None

    # augmented matrix [A ; beta Q^T]
    beta = 1.0
    try:
        if Qrot is not None:
            Qcsr = csr_matrix(beta * Qrot.T)
            Aaug = vstack([A, Qcsr], format="csr")
        else:
            Aaug = A
    except Exception:
        Aaug = A

    # smallest singular vectors
    K = 30
    try:
        Mdim, Ndim = Aaug.shape
        K = int(min(K, min(Mdim, Ndim) - 2))
        if K < 8:
            K = int(min(Mdim, Ndim) - 2)
        if K <= 2:
            raise RuntimeError("too small")
        rng_v = np.random.RandomState(12345)
        v0 = rng_v.randn(Ndim)
        # svds for smallest
        U, S, Vt = svds(Aaug, k=K, which="SM", tol=1e-7, maxiter=20000, v0=v0)
        # sort ascending (svds returns ascending? ensure)
        order_s = np.argsort(S)
        S = S[order_s]; Vt = Vt[order_s]
    except Exception:
        try:
            # dense fallback: eig of A^T A on small subspace via random projection
            rng_f = np.random.RandomState(7)
            # use dense A for speed if feasible
            Ad = A.toarray() if m_pairs * Np < 5_000_000 else None
            if Ad is None:
                return _norm(x_best)
            # project out rotations
            if Qrot is not None:
                Ad = Ad - (Ad @ Qrot) @ Qrot.T
                # restrict: already orthogonal complement via projection + penalty
            # randomized: compute Vt via full SVD of Ad (1080x1090) dense
            _, _, Vt_full = np.linalg.svd(Ad, full_matrices=False)
            # smallest are last rows
            Vt = Vt_full[-K:, :]
            S = np.zeros(K)
        except Exception:
            return _norm(x_best)

    # generate trials: each mode, each sign, epsilons
    eps_list = [0.003, 0.01, 0.03]
    # precompute full V vectors
    trials = []
    # exact threshold for counting
    count_thr = m0 - 1e-7
    for mi in range(Vt.shape[0]):
        v = Vt[mi]
        # orthogonalize against rotations for safety
        if Qrot is not None:
            try:
                coeff = Qrot.T @ v
                v = v - Qrot @ coeff
                nv = float(np.linalg.norm(v))
                if nv < 1e-12:
                    continue
                v = v / nv
            except Exception:
                pass
        else:
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                continue
            v = v / nv
        # expand to per-point tangent coeffs
        # build displacement directions D_i = B_i @ v_i
        D = np.zeros((n, d))
        for i in range(n):
            if i == rattler:
                continue
            bi = idx_map[i] * 5
            vi = v[bi:bi + 5]
            D[i] = B[i] @ vi
        for sgn in (1.0, -1.0):
            Ds = sgn * D
            for eps in eps_list:
                if time.time() - t0 > 60:
                    break
                Xt = _norm(X0 + eps * Ds)
                Gt = Xt @ Xt.T
                np.fill_diagonal(Gt, -2.0)
                mx = float(np.max(Gt))
                cnt = int(np.sum(Gt > count_thr))
                # cnt counts ordered pairs (2x); use as is for ranking
                trials.append((cnt, mx, mi, sgn, eps, Xt))
            if time.time() - t0 > 60:
                break
        if time.time() - t0 > 60:
            break

    if len(trials) == 0:
        return _norm(x_best)

    # rank by (count, maxcos)
    trials.sort(key=lambda t: (t[0], t[1]))

    # pick best few diverse (limit 6)
    best_few = []
    seen = set()
    for t in trials:
        cnt, mx, mi, sgn, eps, Xt = t
        # diversify by mode index
        key = (mi, sgn)
        if key in seen and len(best_few) > 2:
            # allow at most 2 eps per (mode,sign)? skip duplicates beyond first
            continue
        seen.add(key)
        best_few.append(t)
        if len(best_few) >= 7:
            break
    # ensure at least the very best is included
    # time left for polishing
    time_left = BUDGET - (time.time() - t0) - 2.0
    if time_left < 10:
        time_left = 10.0
    per_trial = time_left / max(1, len(best_few))

    overall_best = x_best.copy()
    overall_val = best_val

    for idx, t in enumerate(best_few):
        cnt, mx, mi, sgn, eps, Xt = t
        # quick skip if mx is terrible (> m0+0.02) and cnt huge? still polish maybe
        deadline = time.time() + per_trial * 0.9
        # cap deadline by global budget
        try:
            Xp, vp = _lp_polish(Xt, deadline, t0, BUDGET, iu0, iu1, r_init=1.0e-3, max_steps=25)
        except Exception:
            continue
        if vp < overall_val:
            overall_val = vp
            overall_best = Xp.copy()
            # early exit if clearly improved a lot? continue to use time
        if time.time() - t0 > BUDGET - 2.0:
            break

    # accept only exact improvement, else fallback
    if overall_val < best_val:
        return _norm(overall_best)
    else:
        return _norm(x_best)

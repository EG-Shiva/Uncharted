import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import expm


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def _smooth_max_cross(Cflat, alpha=200.0):
    m = float(np.max(Cflat))
    # logsumexp / alpha
    e = np.exp(alpha * (Cflat - m))
    s = float(np.sum(e))
    return m + np.log(s + 1e-300) / alpha


def _rand_unit_antisym(rng, d=9):
    U = rng.standard_normal((d, d))
    A = U - U.T
    n = float(np.linalg.norm(A))
    if n < 1e-12:
        A[0, 1] = 1.0
        A[1, 0] = -1.0
        n = float(np.linalg.norm(A))
    return A / n


def _spherical_kmeans(X, k, rng, iters=15):
    n, d = X.shape
    # init centers as random points
    idx = rng.choice(n, k, replace=False)
    C = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        D = X @ C.T
        labels = np.argmax(D, axis=1)
        newC = np.zeros_like(C)
        for j in range(k):
            m = labels == j
            if np.any(m):
                v = X[m].sum(axis=0)
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    v = rng.standard_normal(d)
                    nv = np.linalg.norm(v)
                newC[j] = v / nv
            else:
                v = rng.standard_normal(d)
                newC[j] = v / np.linalg.norm(v)
        # check convergence
        if np.allclose(newC, C):
            C = newC
            break
        C = newC
    return labels


def _get_partition(X, kind, rng):
    n, d = X.shape
    if kind == 0:
        # sign of first principal component
        try:
            _, _, Vt = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
            v1 = Vt[0]
        except Exception:
            v1 = rng.standard_normal(d)
            v1 /= np.linalg.norm(v1)
        proj = X @ v1
        med = np.median(proj)
        lab = (proj > med).astype(int)
        # ensure both non-empty and balanced; if degenerate, random split
        if lab.sum() == 0 or lab.sum() == n:
            lab = (proj > 0).astype(int)
        if lab.sum() == 0 or lab.sum() == n:
            perm = rng.permutation(n)
            lab = np.zeros(n, dtype=int)
            lab[perm[:n // 2]] = 1
        return lab
    elif kind == 1:
        # spherical k-means k=2
        lab = _spherical_kmeans(X, 2, rng, iters=15)
        if lab.min() == lab.max():
            perm = rng.permutation(n)
            lab = np.zeros(n, dtype=int)
            lab[perm[:n // 2]] = 1
        return lab
    elif kind == 2:
        # spherical k-means k=3, rotate one cluster vs rest
        labk = _spherical_kmeans(X, 3, rng, iters=15)
        # pick random cluster to be part 1
        c = int(rng.integers(0, 3))
        lab = (labk == c).astype(int)
        if lab.sum() == 0 or lab.sum() == n:
            lab = (labk == 0).astype(int)
        # ensure rotating part is smaller side for efficiency (doesn't matter)
        return lab
    elif kind == 3:
        # spherical k-means k=4
        labk = _spherical_kmeans(X, 4, rng, iters=15)
        c = int(rng.integers(0, 4))
        lab = (labk == c).astype(int)
        if lab.sum() < 10 or lab.sum() > n - 10:
            # fallback to k=2 style: merge
            lab = (labk % 2).astype(int)
        return lab
    else:
        # random hyperplane (diverse halves)
        r = rng.standard_normal(d)
        r /= np.linalg.norm(r)
        proj = X @ r
        med = np.median(proj)
        lab = (proj > med).astype(int)
        return lab


def _fast_settle(X, steps, lr0, lr1, m0, m1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X = _norm_rows(X)
    return X


def _lp_polish(x, seconds, deadline):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    start = time.time()
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        if len(I) == 0:
            break
        # cap constraints for speed
        if len(I) > 6000:
            idx = np.argsort(-c[sel])[:6000]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    SURG_END = t0 + 120.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm_rows(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best = _norm_rows(best)
        best_val = _max_cos(best)
    incumbent = best.copy()
    cur_val = best_val

    surgery = 0
    # main rigid-rotation surgery loop
    while time.time() < SURG_END:
        surgery += 1
        # time guard for at least one more surgery + settle
        if HARD - time.time() < 8:
            break
        kind = surgery % 5  # 0..4 cycle through partitions
        # extra randomization every few
        if rng.random() < 0.15:
            kind = int(rng.integers(0, 5))
        try:
            lab = _get_partition(incumbent, kind, rng)
        except Exception:
            continue
        idx1 = np.where(lab == 1)[0]
        idx0 = np.where(lab == 0)[0]
        if len(idx1) < 8 or len(idx0) < 8:
            continue
        # rotating part = smaller or idx1; static = other
        # rotate idx1, keep idx0 fixed
        Xrot = incumbent[idx1]  # (n1,d)
        Xfix = incumbent[idx0]  # (n0,d)
        n1 = len(idx1)
        # initial theta and A
        theta = float(rng.uniform(0.05, 0.45))
        # occasionally try larger angle
        if rng.random() < 0.1:
            theta = float(rng.uniform(0.30, 0.45))
        A_cur = _rand_unit_antisym(rng, d)
        try:
            R_cur = expm(theta * A_cur)
        except Exception:
            continue
        # initial cross smooth
        try:
            X1R = Xrot @ R_cur.T if False else Xfix @ R_cur.T  # placeholder
        except Exception:
            pass
        # Decide which side to rotate: rotate idx1 side.
        # cross = Xfix @ R? No: if we rotate idx1: cross = Xfix @ R_idx1 @ Xrot.T
        # compute as (Xfix) @ R? Actually Xrot_rot = Xrot @ R.T, cross = Xfix @ Xrot_rot.T = Xfix @ R @ Xrot.T
        def cross_smooth_for_R(R):
            C = (Xfix @ R) @ Xrot.T
            return _smooth_max_cross(C.ravel(), alpha=200.0), float(np.max(C))

        try:
            cur_smooth, cur_true_cross = cross_smooth_for_R(R_cur)
        except Exception:
            continue
        init_smooth = cur_smooth
        best_smooth = cur_smooth
        best_A = A_cur.copy()
        best_R = R_cur.copy()
        # derivative-free search over A
        n_prop = int(rng.integers(300, 801))
        sigma = float(rng.uniform(0.05, 0.2))
        # cap props if time low
        remain = SURG_END - time.time()
        if remain < 10:
            n_prop = min(n_prop, 300)
        for _ in range(n_prop):
            N = _rand_unit_antisym(rng, d)
            A_prop_raw = A_cur + sigma * N
            fnorm = float(np.linalg.norm(A_prop_raw))
            if fnorm < 1e-12:
                continue
            A_prop = A_prop_raw / fnorm
            try:
                R_prop = expm(theta * A_prop)
            except Exception:
                continue
            try:
                s_prop, _ = cross_smooth_for_R(R_prop)
            except Exception:
                continue
            if s_prop < cur_smooth:
                cur_smooth = s_prop
                A_cur = A_prop
                R_cur = R_prop
                if s_prop < best_smooth:
                    best_smooth = s_prop
                    best_A = A_prop.copy()
                    best_R = R_prop.copy()
            # early time exit
            if time.time() > SURG_END:
                break
        if best_smooth >= init_smooth - 1e-12:
            continue  # no cross improvement, skip settle
        # apply best rotation
        try:
            Xrot_new = Xrot @ best_R.T
        except Exception:
            continue
        Y = incumbent.copy()
        Y[idx1] = Xrot_new
        Y = _norm_rows(Y)
        # fast repulsion settle 300-800 steps
        steps = int(rng.integers(300, 801))
        # randomize settle hypers a bit
        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0003, 0.0012))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        try:
            Y = _fast_settle(Y, steps, lr0, lr1, m0, m1)
        except Exception:
            continue
        v = _max_cos(Y)
        if v < cur_val - 1e-12:
            incumbent = Y
            cur_val = v
            best = Y.copy()
            best_val = v
        if time.time() > SURG_END:
            break

    # final LP polish capped at 30s with hard deadline 170s
    remain_hard = HARD - time.time()
    if remain_hard > 5:
        lp_time = min(30.0, remain_hard - 2.0)
        if lp_time > 3:
            try:
                incumbent = _lp_polish(incumbent, lp_time, HARD)
            except Exception:
                pass
            incumbent = _norm_rows(incumbent)
    return incumbent

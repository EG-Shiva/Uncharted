import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = iu_c.max()
        sel = iu_c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns: I-block, J-block, t
        base = np.arange(d)
        cols_I = (I[:, None] * d + base).ravel()
        cols_J = (J[:, None] * d + base).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = cols_I[0::d] if False else 0  # placeholder, rebuilt below
        # build cols vectorized
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + base
        cols_mat[:, d:2 * d] = J[:, None] * d + base
        cols_mat[:, 2 * d] = n * d
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_opt(X, steps, k0=80.0, k1=420.0, lr0=0.02, lr1=0.002):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights, max is 1
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / nrm) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _best_void(X_others, rng, n_cand=4000):
    n, d = X_others.shape
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # max dot to others
    S = C @ X_others.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return C[i]


def solve():
    rng = np.random.default_rng(1)
    t0 = time.time()
    TIME_BUDGET = 172.0
    t_dead = t0 + TIME_BUDGET

    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    # ensure shape (181,9)
    if xb.shape != (181, 9):
        # try to reshape/crop deterministically
        flat = xb.ravel()
        need = 181 * 9
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(181, 9)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()

    # quick stats to know rattlers etc. (not strictly needed)
    n, d = X_best.shape

    # main exploration: fixed max iters, time-guarded
    MAX_IT = 70
    for it in range(MAX_IT):
        if time.time() > t_dead - 22.0:
            break
        X = X_best.copy()
        mode = it % 4
        if mode == 0:
            sigma = 0.015 if it < 30 else 0.008
            X = X + sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 700
        elif mode == 1:
            # relocate worst point into largest void
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            victim = int(np.argmax(per_max))
            # tie-break deterministically by overloaded count
            mask = np.zeros_like(X)
            others_idx = [i for i in range(n) if i != victim]
            X_others = X[others_idx]
            cand = _best_void(X_others, rng, n_cand=4000)
            X[victim] = cand
            # tiny shake of others to break symmetry
            X = X + 0.004 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # keep victim close to void optimum: re-assign after shake
            # recompute: keep as is (shake already small)
            steps = 800
        elif mode == 2:
            # cluster shake around most crowded point
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = G.max()
            cnt = (G > gmax - 0.02).sum(axis=1)
            center = int(np.argmax(cnt))
            order = np.argsort(-G[center])
            k_near = 12
            neigh = order[:k_near]
            X[neigh] += 0.06 * rng.standard_normal((k_near, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 700
        else:
            sigma = 0.035
            X = X + sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 900

        # annealed soft-max descent
        X = _smooth_opt(X, steps=steps)

        v = _max_cos(X)
        # promising? then LP snap
        if v < best_val + 0.012:
            # give 1.2s for snap
            if time.time() > t_dead - 22.0:
                # no time for LP, just compare
                if v < best_val:
                    X_best = X
                    best_val = v
                continue
            X2 = _lp_polish(X, seconds=1.2, t_deadline=t_dead - 20.0)
            v2 = _max_cos(X2)
            if v2 < best_val:
                X_best = X2
                best_val = v2
            elif v < best_val:
                X_best = X
                best_val = v
        else:
            if v < best_val:
                X_best = X
                best_val = v

    # final polish of best with remaining time
    remain = t_dead - time.time() - 1.0
    if remain > 2.0:
        X_best = _lp_polish(X_best, seconds=min(18.0, remain), t_deadline=t_dead - 0.5)
        # one more short smooth+LP cycle to squeeze
        if time.time() < t_dead - 4.0:
            X_best = _smooth_opt(X_best, steps=300, k0=300.0, k1=600.0, lr0=0.005, lr1=0.0005)
            X_best = _lp_polish(X_best, seconds=max(1.0, t_dead - time.time() - 0.5), t_deadline=t_dead - 0.2)

    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(X_best)):
        X_best = xb
    return X_best

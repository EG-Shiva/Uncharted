import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _descend(X, steps, k0, k1, lr0, lr1, mu=0.85, rng=None):
    n, d = X.shape
    vel = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # softmax weights
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        Gt = Gt / (nrm + 1e-30)
        vel = mu * vel + (1.0 - mu) * Gt if s > 0 else Gt
        # bias correction-ish
        X = X - lr * vel
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if not np.all(np.isfinite(X)):
            break
    return X

def _void_candidate(X_keep, rng, n_cand=2000):
    d = X_keep.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ X_keep.T
    m = S.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it > 25:
            break
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            cs = c[sel]
            idx = np.argpartition(-cs, 1500)[:1500]
            I = I[idx]; J = J[idx]
            cb = cs[idx]
        else:
            cb = c[sel]
            # also cap extreme
        nv = n * d + 1
        cols_mat = np.empty((p if p <= 1500 else 1500, 2 * d + 1), dtype=np.int64)
        pp = len(I)
        cols_mat = np.empty((pp, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(pp), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((pp, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(pp, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    TIME_BUDGET = 172.0
    t_dead = t0 + TIME_BUDGET
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (181, 9):
        flat = xb.ravel()
        need = 181 * 9
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(181, 9)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()
    n, d = X_best.shape

    it = 0
    # exploration without any LP
    while True:
        if time.time() > t_dead - 26.0:
            break
        if it > 400:
            break
        X = X_best.copy()
        mode = it % 6
        if mode == 0:
            sigma = 0.008 + 0.004 * rng.random()
            X = X + sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 900
            k0, k1 = 40.0, 280.0
            lr0, lr1 = 0.03, 0.002
            mu = 0.8
        elif mode == 1:
            # single worst-point void reinsertion
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            victim = int(np.argmax(per))
            keep = [i for i in range(n) if i != victim]
            cand = _void_candidate(X[keep], rng, n_cand=3000)
            X[victim] = cand
            X = X + 0.003 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 1100
            k0, k1 = 50.0, 350.0
            lr0, lr1 = 0.025, 0.0015
            mu = 0.85
        elif mode == 2:
            # multi-delete: rattlers/loose + overloaded
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-7).sum(axis=1)
            # victims: 2 smallest cnt + 6 largest cnt (randomized)
            order = np.argsort(cnt)
            kdel = 8
            # pick 2 loosest + 6 tightest with jitter
            loos = order[:4]
            tight = order[-8:]
            sel_lo = rng.choice(loos, size=2, replace=False)
            sel_ti = rng.choice(tight, size=6, replace=False)
            victims = np.concatenate([sel_lo, sel_ti])
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            K = X[keep_mask]
            # relax keep briefly
            K = _descend(K, steps=120, k0=60.0, k1=150.0, lr0=0.015, lr1=0.004, mu=0.7)
            # reinsert one by one into holes
            D = np.empty((kdel, d))
            cur = K
            for j in range(kdel):
                cand = _void_candidate(cur, rng, n_cand=2500)
                # small random alternative among top holes occasionally
                D[j] = cand
                cur = np.vstack([cur, cand[None, :]])
            X[keep_mask] = K
            X[~keep_mask] = D[rng.permutation(kdel)]
            X = X + 0.002 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 1200
            k0, k1 = 40.0, 300.0
            lr0, lr1 = 0.03, 0.002
            mu = 0.85
        elif mode == 3:
            # neighborhood shake of most crowded hub
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.02).sum(axis=1)
            center = int(np.argmax(cnt))
            order = np.argsort(-G[center])
            kne = 10 + int(rng.integers(0, 6))
            neigh = order[:kne]
            X[neigh] += (0.05 + 0.03 * rng.random()) * rng.standard_normal((kne, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 1000
            k0, k1 = 60.0, 400.0
            lr0, lr1 = 0.02, 0.001
            mu = 0.9
        elif mode == 4:
            sigma = 0.03 + 0.02 * rng.random()
            X = X + sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 1200
            k0, k1 = 30.0, 250.0
            lr0, lr1 = 0.04, 0.003
            mu = 0.8
        else:
            # fresh random start (rare basin jump)
            if it % 24 == 23:
                X = rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                steps = 1800
                k0, k1 = 25.0, 300.0
                lr0, lr1 = 0.05, 0.002
                mu = 0.8
            else:
                sigma = 0.012
                X = X + sigma * rng.standard_normal(X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                steps = 900
                k0, k1 = 80.0, 450.0
                lr0, lr1 = 0.015, 0.001
                mu = 0.88
        X = _descend(X, steps=steps, k0=k0, k1=k1, lr0=lr0, lr1=lr1, mu=mu)
        v = _max_cos(X)
        if v < best_val:
            X_best = X
            best_val = v
        it += 1

    remain = t_dead - time.time() - 0.8
    if remain > 3.0:
        # short pre-polish smooth then LP snap
        X_best = _descend(X_best, steps=250, k0=250.0, k1=600.0, lr0=0.006, lr1=0.0006, mu=0.9)
        X_best = _lp_polish(X_best, budget=min(20.0, remain - 0.5), t_dead=t_dead - 0.3)
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(X_best)):
        X_best = xb
    return X_best

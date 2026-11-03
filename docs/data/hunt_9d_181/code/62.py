import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(X, steps, k0=80.0, k1=350.0, lr0=0.02, lr1=0.002):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = float(np.linalg.norm(Gt, axis=1).max())
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / nrm) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, budget):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if radius < 1e-13 or it >= 12:
            break
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = c > thr
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1200:
            idx = np.argpartition(-csel, 1200)[:1200]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = 1200
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _deepest_hole(K, rng, n_cand=20000):
    d = K.shape[1]
    # batch to bound memory/time; 2 x 10000
    best_pt = None
    best_v = 2.0
    for _ in range(n_cand // 10000):
        C = rng.standard_normal((10000, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ K.T
        mx = S.max(axis=1)
        i = int(np.argmin(mx))
        if float(mx[i]) < best_v:
            best_v = float(mx[i])
            best_pt = C[i]
    return best_pt

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (181, 9):
        flat = xb.ravel()
        need = 181 * 9
        rng0 = np.random.default_rng(0)
        if flat.size < need:
            flat = np.concatenate([flat, rng0.standard_normal((need - flat.size,))])
        xb = flat[:need].reshape(181, 9)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()
    n, d = 181, 9
    del_sizes = [18, 22, 15, 25, 20, 16]
    n_restart = 0
    # time for final polish reserve
    while time.time() < T_END - 18.0 and n_restart < 12:
        del_size = del_sizes[n_restart % len(del_sizes)]
        rng = np.random.default_rng(1000 + n_restart * 77)
        n_restart += 1
        # contact analysis for hub-biased deletion
        try:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt_loose = (G > m - 0.01).sum(axis=1)
            cnt_tight = (G > m - 1e-7).sum(axis=1)
        except Exception:
            continue
        order_hub = np.argsort(-cnt_loose, kind="stable")
        n_hub = del_size // 2
        hub_set = list(order_hub[:n_hub])
        # add rattlers (0 tight contacts) if any not already included
        ratt = [int(i) for i in range(n) if cnt_tight[i] == 0]
        for r in ratt:
            if len(hub_set) >= n_hub + 2:
                break
            if r not in hub_set:
                hub_set.append(r)
        # fill remainder randomly
        remaining = [i for i in range(n) if i not in hub_set]
        k_rand = del_size - len(hub_set)
        if k_rand > 0:
            perm = rng.permutation(len(remaining))
            for k in range(k_rand):
                hub_set.append(remaining[int(perm[k])])
        del_idx = np.array(sorted(hub_set)[:del_size], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        K = X_best[keep_mask].copy()
        # hard optimise reduced code well below 0.4624
        # adapt steps to remaining time
        time_left = T_END - time.time()
        if time_left < 25.0:
            break
        K = _smooth(K, steps=500, k0=60.0, k1=350.0, lr0=0.02, lr1=0.002)
        # regrow one at a time
        ok = True
        for t in range(del_size):
            if time.time() > T_END - 18.0:
                ok = False
                break
            pt = _deepest_hole(K, rng, n_cand=20000)
            K = np.vstack([K, pt])
            # 150 global smoothing steps (use 120 for larger N to save time, 150 otherwise)
            steps = 150 if K.shape[0] < 175 else 120
            # anneal hardness with growth
            K = _smooth(K, steps=steps, k0=80.0, k1=320.0, lr0=0.015, lr1=0.002)
            if (t + 1) % 4 == 0:
                if time.time() > T_END - 19.0:
                    ok = False
                    break
                K = _lp_polish(K, budget=1.0)
        if not ok:
            # evaluate partial? need exactly 181 points, skip
            continue
        # ensure 181 points
        if K.shape[0] != 181:
            continue
        v = _max_cos(K)
        # LP snap if promising
        if v < best_val + 0.004:
            if time.time() < T_END - 19.0:
                K2 = _lp_polish(K, budget=1.5)
                v2 = _max_cos(K2)
                if v2 < v:
                    K, v = K2, v2
        if v < best_val:
            X_best = K / np.linalg.norm(K, axis=1, keepdims=True)
            best_val = float(v)
    # final polish of incumbent with remaining time
    rem = T_END - time.time() - 0.5
    if rem > 2.0:
        X_best = _lp_polish(X_best, budget=min(12.0, rem))
        if time.time() < T_END - 3.0:
            X_best = _smooth(X_best, steps=250, k0=250.0, k1=550.0, lr0=0.005, lr1=0.0005)
            X_best = _lp_polish(X_best, budget=max(1.0, T_END - time.time() - 0.3))
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(X_best)):
        X_best = xb
    return X_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, init_radius=0.005):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        now = time.time()
        if now - t0 >= budget:
            break
        if budget - (now - t0) < 0.12:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt_sel = int(sel.sum())
        if cnt_sel == 0:
            radius *= 0.5
            continue
        if cnt_sel > 1800:
            # keep worst 1800
            idx = np.argpartition(-c, 1800)[:1800]
            I = iu0[0][idx]; J = iu0[1][idx]
            csel = c[idx]
        else:
            I = iu0[0][sel]; J = iu0[1][sel]
            csel = c[sel]
        p = len(I)
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2*d] = J[:, None] * d + ar
        cols_mat[:, 2*d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2*d] = x[I]
        vals_mat[:, 2*d] = -1.0
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(X, steps, k0, k1, lr0, lr1, free=None, cap=0.03):
    n, d = X.shape
    if free is None:
        free = np.ones(n, dtype=bool)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(k * (G - gmax), -50.0, 0.0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        Gt[~free] = 0.0
        nrm = np.linalg.norm(Gt[free], axis=1).max() if free.any() else 0.0
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        step = (lr / nrm) * Gt
        # cap max row step
        mx = np.linalg.norm(step[free], axis=1).max() if free.any() else 0.0
        if mx > cap and mx > 0:
            step *= (cap / mx)
        X[free] = X[free] - step[free]
        X[free] /= np.linalg.norm(X[free], axis=1, keepdims=True)
    return X

def _deepest_hole(Xkeep, rng, n_cand=3000, exclude=None):
    d = Xkeep.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xkeep.T
    mx = S.max(axis=1)
    i = int(np.argmin(mx))
    return C[i]

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
    except Exception:
        vr = 1e9
    xb = np.asarray(xb, dtype=float)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    vb = _max_cos(xb)
    if vr < vb:
        xb, vb = xr, vr
    n, d = xb.shape
    x_best = xb.copy()
    best_val = vb

    ks = [18, 14, 22, 16]
    sigmas = [0.07, 0.10, 0.05]
    it = 0
    while time.time() - t0 < LIMIT - 26.0:
        it += 1
        remain = LIMIT - 26.0 - (time.time() - t0)
        if remain < 3.5:
            break
        y = x_best.copy()
        if it % 4 == 0:
            # moderate global shake for diversity
            sigma = sigmas[(it // 4 - 1) % len(sigmas)]
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=110, k0=30.0, k1=220.0, lr0=0.02, lr1=0.003, free=None)
            y = _anneal(y, steps=60, k0=220.0, k1=350.0, lr0=0.003, lr1=0.001, free=None)
        else:
            k = ks[(it - 1) % len(ks)]
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-7).sum(axis=1)
            nh = k // 3
            nl = k // 3
            hubs = np.argsort(-cnt)[:nh]
            lows = np.argsort(cnt)[:nl]
            center = int(rng.integers(0, n))
            dots = y[center] @ y.T
            nn = np.argsort(-dots)[: (k - nh - nl + 2)]
            victims = list(dict.fromkeys(list(map(int, hubs)) + list(map(int, lows)) + list(map(int, nn))))
            # trim / pad to k
            if len(victims) > k:
                victims = victims[:k]
            while len(victims) < k:
                r = int(rng.integers(0, n))
                if r not in victims:
                    victims.append(r)
            victims = np.array(victims, dtype=int)
            rng.shuffle(victims)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            free_keep = np.zeros(n, dtype=bool)
            # relax keepers briefly with keepers-only anneal
            K = y[keep_mask].copy()
            K = _anneal(K, steps=50, k0=40.0, k1=160.0, lr0=0.015, lr1=0.004, free=None)
            y[keep_mask] = K
            # sequential deepest-hole reinsertion
            Xkeep = y[keep_mask].copy()
            for v in victims:
                h = _deepest_hole(Xkeep, rng, n_cand=3000)
                h = h + 0.008 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[v] = h
                Xkeep = np.vstack([Xkeep, h.reshape(1, -1)])
            # stage 1: move only victims
            free_v = np.zeros(n, dtype=bool)
            free_v[victims] = True
            y = _anneal(y, steps=70, k0=50.0, k1=200.0, lr0=0.02, lr1=0.005, free=free_v)
            # stage 2: joint all-active anneal (balances degrees)
            y = _anneal(y, steps=110, k0=40.0, k1=280.0, lr0=0.015, lr1=0.002, free=None)
        # short LP evaluation
        rem2 = LIMIT - 26.0 - (time.time() - t0)
        if rem2 < 1.5:
            break
        budget = min(3.0, rem2)
        y = _lp_polish(y, budget=budget, init_radius=0.006)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
    # final polish
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        x_best = _lp_polish(x_best, budget=rem, init_radius=0.003)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

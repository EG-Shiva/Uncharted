import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(X, steps, k0, k1, lr0, lr1):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # exp weights, clipped
        W = np.exp(np.clip(k * (G - gmax), -50.0, 0.0))
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

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it >= 25:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
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
            p = len(I)
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
    rng = np.random.default_rng(42)
    t0 = time.time()
    T_END = t0 + 168.0
    # load seed
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

    KDEL_SEQ = [6, 8, 10, 5, 7, 9, 12, 4]
    MAX_IT = 150
    for it in range(MAX_IT):
        # reserve 30s for final polish
        if time.time() > T_END - 30.0:
            break
        kdel = KDEL_SEQ[it % len(KDEL_SEQ)]
        X = X_best.copy()
        # contact stats on current best copy (fast)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        tol = 1e-7
        cnt = (G > m - tol).sum(axis=1)
        # victims: rattlers first then highest contacts
        order_hi = np.argsort(-cnt, kind='stable')
        # rattlers = cnt==0
        victims = []
        for idx in range(n):
            if cnt[idx] == 0:
                victims.append(idx)
        for idx in order_hi:
            if idx not in victims:
                victims.append(int(idx))
            if len(victims) >= kdel:
                break
        victims = np.array(victims[:kdel], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        K = X[keep_mask]  # (n-kdel, d)
        # brief relax of keeper to open holes (collective move)
        K = _smooth(K, steps=80, k0=40.0, k1=150.0, lr0=0.015, lr1=0.003)
        # greedy deepest-hole reinsertion
        new_pts = []
        # vary candidate count / randomness per iteration
        n_cand = 2500 if (it % 2 == 0) else 3500
        for j in range(kdel):
            C = rng.standard_normal((n_cand, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ K.T
            mx = S.max(axis=1)
            if len(new_pts) > 0:
                D = np.stack(new_pts, axis=0)
                S2 = C @ D.T
                mx2 = S2.max(axis=1)
                mx = np.maximum(mx, mx2)
            # diversity: sometimes pick among top few
            if rng.random() < 0.4:
                k_top = 6
                idxp = np.argpartition(mx, k_top - 1)[:k_top]
                # pick random among top
                pick = idxp[rng.integers(0, k_top)]
            else:
                pick = int(np.argmin(mx))
            pt = C[pick]
            new_pts.append(pt)
            # append to K for next insertion scoring
            K = np.concatenate([K, pt[None, :]], axis=0)
        # K now has n points (keepers relaxed + reinserted); this is full config
        X = K.copy()
        # small global shake to break symmetry
        shake = 0.004 if (it % 3 != 0) else 0.008
        X = X + shake * rng.standard_normal(X.shape)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # anneal: two stages low-k spread then high-k max-min
        if it % 3 == 0:
            X = _smooth(X, steps=450, k0=25.0, k1=320.0, lr0=0.02, lr1=0.002)
        elif it % 3 == 1:
            X = _smooth(X, steps=500, k0=60.0, k1=450.0, lr0=0.018, lr1=0.0015)
        else:
            X = _smooth(X, steps=350, k0=15.0, k1=200.0, lr0=0.025, lr1=0.003)
            X = _smooth(X, steps=150, k0=250.0, k1=550.0, lr0=0.006, lr1=0.0008)
        v = _max_cos(X)
        if v < best_val:
            best_val = v
            X_best = X.copy()
        # also try mild shake-only basin (no surgery) every 5th iter as alternative
        if it % 5 == 4:
            if time.time() > T_END - 30.0:
                break
            Y = X_best.copy()
            Y = Y + 0.012 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, steps=300, k0=80.0, k1=400.0, lr0=0.015, lr1=0.0015)
            vy = _max_cos(Y)
            if vy < best_val:
                best_val = vy
                X_best = Y.copy()

    # final polish: short smooth then capped LP
    remain = T_END - time.time() - 0.5
    if remain > 5.0:
        # squeeze max with high-k smooth
        X_best = _smooth(X_best, steps=250, k0=300.0, k1=600.0, lr0=0.005, lr1=0.0006)
        v = _max_cos(X_best)
        if v < best_val:
            best_val = v
        lp_budget = min(22.0, remain - 2.0)
        if lp_budget > 3.0:
            Xp = _lp_polish(X_best, budget=lp_budget, t_dead=T_END - 0.3)
            vp = _max_cos(Xp)
            if vp < best_val:
                X_best = Xp
                best_val = vp
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(X_best)):
        X_best = xb
    return X_best

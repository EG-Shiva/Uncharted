import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contact_counts(x, tol=2e-3):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = np.sum(g > m - tol, axis=1)
    return cnt, m

def lp_polish(x, seconds, radius_init=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if np.sum(sel) > 2400:
            k = 2400
            idx = np.argpartition(-c, k)[:k]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def target_quench(X_init, T, steps, lr0, lr1):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        p = 2.0 + 4.0 * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = G - T
        W = np.zeros_like(G)
        pos = D > 0
        if np.any(pos):
            W[pos] = p * (D[pos] ** (p - 1.0))
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def far_hole(fixed, rng, n_try=2500):
    n, d = fixed.shape
    Z = rng.standard_normal((n_try, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ fixed.T
    m = S.max(axis=1)
    return Z[np.argmin(m)]

def solve():
    t0 = time.time()
    LIMIT = 170.0
    rng = np.random.default_rng(1)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if time.time() - t0 < LIMIT - 30:
        xb = lp_polish(xb, seconds=2.5)
    best_val = max_cos(xb)
    X_best = xb.copy()
    it = 0
    while time.time() - t0 < LIMIT - 8:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        if time_left < 9:
            break
        # target gap
        r = rng.random()
        if it % 11 == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(4e-3))))
        elif it % 4 == 0:
            delta = float(np.exp(rng.uniform(np.log(2e-4), np.log(9e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(3e-5), np.log(3e-4))))
        T = best_val - delta
        cnt, cur_m = contact_counts(X_best, tol=2e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        mode = it % 6
        X = X_best.copy()
        # perturbation: hole re-insert + kicks
        if mode == 0:
            nv = int(rng.integers(3, 7))
            victims = order_low[:nv].copy()
            keep = np.ones(n, dtype=bool); keep[victims] = False
            fixed = X[keep]
            for v in victims:
                nh = far_hole(fixed, rng, n_try=2000)
                X[v] = nh
                fixed = np.vstack([fixed, nh[None, :]])
            sig = 0.015
            X += sig * rng.standard_normal((n, d))
        elif mode == 1:
            nv = int(rng.integers(2, 5))
            victims = np.concatenate([order_low[:2], rng.choice(n, size=nv, replace=False)])
            victims = np.unique(victims)[:nv+2]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            fixed = X[keep]
            for v in victims:
                X[v] = far_hole(fixed, rng, n_try=2500)
                fixed = np.vstack([fixed, X[v][None, :]])
            sig = 0.008
            X += sig * rng.standard_normal((n, d))
        elif mode == 2:
            # kick jammed hubs
            nh = int(rng.integers(4, 10))
            victims = order_high[:nh]
            X[victims] += 0.06 * rng.standard_normal((nh, d))
            X += 0.008 * rng.standard_normal((n, d))
        elif mode == 3:
            X += 0.02 * rng.standard_normal((n, d))
        elif mode == 4:
            nv = int(rng.integers(1, 4))
            victims = rng.choice(n, size=nv, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            fixed = X[keep]
            for v in victims:
                X[v] = far_hole(fixed, rng, n_try=2500)
                fixed = np.vstack([fixed, X[v][None, :]])
            X += 0.012 * rng.standard_normal((n, d))
        else:
            X += 0.035 * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # quench length adapts to time left
        if time_left > 60:
            steps = 450
        elif time_left > 25:
            steps = 350
        else:
            steps = 250
        lr0 = 0.025 if (mode % 2 == 0) else 0.018
        Xq = target_quench(X, T, steps, lr0, 0.002)
        vq = max_cos(Xq)
        # only polish if promising to save time
        thresh = best_val + (0.002 if it % 3 == 0 else 0.0006)
        if vq < thresh:
            budget = min(2.8, max(0.8, time_left - 9.0))
            # smaller budget if far
            if vq > best_val:
                budget = min(budget, 1.4)
            Xp = lp_polish(Xq, seconds=budget)
            vp = max_cos(Xp)
            if vp < best_val - 1e-12:
                X_best = Xp.copy()
                best_val = vp
                continue
            # also accept quench alone if it beats best (polish failed to improve but quench did)
            if vq < best_val - 1e-12:
                # quick extra polish next loop will jam it; store quenched
                X_best = lp_polish(Xq, seconds=min(1.0, max(0.5, LIMIT-(time.time()-t0)-7.0)))
                best_val = max_cos(X_best)
        # opportunistic: if quench alone is much better than gap, keep as new base without polish
        elif vq < best_val - 5e-4:
            X_best = Xq.copy()
            best_val = vq
    # final jam
    if time.time() - t0 < LIMIT - 2:
        X_best = lp_polish(X_best, seconds=max(0.5, LIMIT - (time.time() - t0) - 1.0))
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

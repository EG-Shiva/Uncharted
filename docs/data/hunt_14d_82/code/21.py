import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def worst_pair(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    k = int(np.argmax(g))
    n = x.shape[0]
    return k // n, k % n, float(g.max())

def far_hole_vec(fixed, rng, n_try=2000):
    d = fixed.shape[1]
    Z = rng.standard_normal((n_try, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ fixed.T
    m = S.max(axis=1)
    return Z[int(np.argmin(m))]

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
        # power-law weights, symmetrized
        W = np.zeros_like(G)
        pos = D > 0
        if np.any(pos):
            W[pos] = p * (D[pos] ** (p - 1.0))
        # gradient on sphere: Gr = W @ X projected
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        X -= lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

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
        if np.sum(sel) > 2200:
            k = 2200
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

def solve():
    t0 = time.time()
    LIMIT = 170.0
    rng = np.random.default_rng(7)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        xb = rng.standard_normal((n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # quick initial jam only if time allows (very short)
    if time.time() - t0 < LIMIT - 40:
        xb = lp_polish(xb, seconds=2.0)
    best_val = max_cos(xb)
    X_best = xb.copy()
    it = 0
    while time.time() - t0 < LIMIT - 10:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        if time_left < 11:
            break
        # target gap log-uniform
        r = rng.random()
        if it % 12 == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(3.5e-3))))
        elif it % 4 == 0:
            delta = float(np.exp(rng.uniform(np.log(2e-4), np.log(8e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(4e-5), np.log(3e-4))))
        T = best_val - delta
        # worst-point scores for victim choice
        G = X_best @ X_best.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        order_worst = np.argsort(-mx, kind="stable")
        order_best = order_worst[::-1]
        X = X_best.copy()
        mode = it % 6
        if mode == 0:
            # split the single worst pair + small noise (exploit isolated bottleneck)
            a, b, _ = worst_pair(X)
            eps = 0.004 + 0.004 * rng.random()
            Xa = X[a] - eps * X[b]
            Xb = X[b] - eps * X[a]
            Xa /= np.linalg.norm(Xa); Xb /= np.linalg.norm(Xb)
            X[a] = Xa; X[b] = Xb
            # also split next worst pair
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            # find second worst disjoint pair
            k2 = np.argsort(-G2.ravel())[:6]
            for kk in k2:
                i2, j2 = kk // n, kk % n
                if len({i2, j2, a, b}) == 4:
                    e2 = 0.003
                    Xi = X[i2] - e2 * X[j2]
                    Xj = X[j2] - e2 * X[i2]
                    X[i2] = Xi / np.linalg.norm(Xi)
                    X[j2] = Xj / np.linalg.norm(Xj)
                    break
            X += 0.008 * rng.standard_normal((n, d))
        elif mode == 1:
            nv = int(rng.integers(3, 7))
            victims = order_best[-nv:].copy()  # loose points -> far holes
            # include one end of worst pair sometimes
            if rng.random() < 0.5:
                a, b, _ = worst_pair(X)
                victims[0] = a if rng.random() < 0.5 else b
            keep = np.ones(n, dtype=bool); keep[victims] = False
            fixed = X[keep]
            for v in victims:
                nh = far_hole_vec(fixed, rng, n_try=2000)
                X[v] = nh
                fixed = np.vstack([fixed, nh[None, :]])
            X += 0.012 * rng.standard_normal((n, d))
        elif mode == 2:
            nv = int(rng.integers(2, 5))
            victims = order_worst[:nv].copy()
            keep = np.ones(n, dtype=bool); keep[victims] = False
            fixed = X[keep]
            for v in victims:
                X[v] = far_hole_vec(fixed, rng, n_try=2500)
                fixed = np.vstack([fixed, X[v][None, :]])
            X += 0.010 * rng.standard_normal((n, d))
        elif mode == 3:
            X += 0.022 * rng.standard_normal((n, d))
        elif mode == 4:
            # jammed neighbourhood kick: kick neighbors of worst pair
            a, b, _ = worst_pair(X)
            Ga = (X @ X[a])
            Ga[a] = -2
            neigh = np.argsort(-Ga)[:8]
            X[neigh] += 0.05 * rng.standard_normal((len(neigh), d))
            X += 0.008 * rng.standard_normal((n, d))
        else:
            X += 0.035 * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if time_left > 60:
            steps = 420
        elif time_left > 28:
            steps = 320
        else:
            steps = 220
        lr0 = 0.022 if (mode % 2 == 0) else 0.016
        Xq = target_quench(X, T, steps, lr0, 0.002)
        vq = max_cos(Xq)
        # polish only if promising
        thresh = best_val + (0.0015 if it % 4 == 0 else 0.0005)
        if vq < thresh:
            budget = min(2.2, max(0.7, time_left - 10.5))
            if vq > best_val:
                budget = min(budget, 1.1)
            Xp = lp_polish(Xq, seconds=budget)
            vp = max_cos(Xp)
            if vp < best_val - 1e-12:
                X_best = Xp.copy()
                best_val = vp
                continue
            if vq < best_val - 1e-12:
                # quench won but polish didn't; keep quenched with light jam
                tl = LIMIT - (time.time() - t0)
                if tl > 12:
                    X_best = lp_polish(Xq, seconds=min(0.9, tl - 10.5))
                    best_val = max_cos(X_best)
                else:
                    X_best = Xq.copy()
                    best_val = vq
        elif vq < best_val - 4e-4:
            X_best = Xq.copy()
            best_val = vq
    # final jam
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 0.6:
        X_best = lp_polish(X_best, seconds=min(rem, 6.0))
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

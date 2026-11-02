import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

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

def minimax_quench(X_init, steps, a0, a1, lr0, lr1):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    la0, la1 = np.log(a0), np.log(a1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0:
            continue
        W = E / S
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

def subgrad_balance(X_init, steps=800, lr0=0.02, tol=8e-4):
    X = X_init.copy()
    n, d = X.shape
    best = max_cos(X)
    Xb = X.copy()
    lr = lr0
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - tol
        I, J = iu[0][sel], iu[1][sel]
        if len(I) == 0:
            break
        A = np.zeros((n, n))
        np.add.at(A, (I, J), 1.0)
        np.add.at(A, (J, I), 1.0)
        Gr = A @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Gt = Gt / nrm
        Y = X - lr * Gt
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = max_cos(Y)
        if v < best:
            best = v
            Xb = Y.copy()
            X = Y
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.6
            if lr < 1e-7:
                lr = 1e-7
            # still move sometimes to rebalance
            if s % 5 == 0:
                X = Y
        if s % 200 == 199:
            tol = max(tol * 0.5, 1e-5)
    return Xb

def lp_polish_fast(x, seconds, max_pairs=850, radius_init=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    niter = 0
    while time.time() - start < seconds and radius > 1e-13 and niter < 25:
        niter += 1
        if time.time() - start > seconds - 0.15:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thresh = max(12.0 * radius, 4e-6)
        sel = c > m - thresh
        if np.sum(sel) > max_pairs:
            k = max_pairs
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
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def far_hole(fixed, rng, n_try=2000):
    d = fixed.shape[1]
    Z = rng.standard_normal((n_try, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ fixed.T
    m = S.max(axis=1)
    return Z[np.argmin(m)]

def greedy_build(rng, n, d, n_try=4000):
    Z = rng.standard_normal((n_try, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    picked = [int(rng.integers(n_try))]
    sel = [Z[picked[0]]]
    for _ in range(1, n):
        S = Z @ np.stack(sel).T
        m = S.max(axis=1)
        m[picked] = 1e9
        j = int(np.argmin(m))
        picked.append(j)
        sel.append(Z[j])
    return np.stack(sel)

def solve():
    t0 = time.time()
    LIMIT = 170.0
    rng = np.random.default_rng(12345)
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
    best_val = max_cos(xb)
    X_best = xb.copy()
    # Phase 1: jam incumbent with smooth-max (fast, no LP)
    if time.time() - t0 < LIMIT - 40:
        try:
            Xj = minimax_quench(X_best, steps=500, a0=60.0, a1=900.0, lr0=0.02, lr1=0.002)
            vj = max_cos(Xj)
            if vj < best_val:
                best_val = vj
                X_best = Xj.copy()
            Xj2 = subgrad_balance(X_best, steps=400, lr0=0.01, tol=5e-4)
            vj2 = max_cos(Xj2)
            if vj2 < best_val:
                best_val = vj2
                X_best = Xj2.copy()
        except Exception:
            pass
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < LIMIT - 14:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        if time_left < 10:
            break
        # target gap: mostly moderate-large to force basin change
        r = rng.random()
        if it % 23 == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(2.5e-3))))
        elif it % 5 == 0:
            delta = float(np.exp(rng.uniform(np.log(1.5e-4), np.log(8e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(2e-5), np.log(2.5e-4))))
        T = best_val - delta
        X = X_best.copy()
        mode = it % 8
        if mode == 7 and it > 5:
            # fresh greedy basin (bold jump)
            try:
                X = greedy_build(rng, n, d, n_try=3500)
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
                X += 0.01 * rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            except Exception:
                X = X_best + 0.15 * rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # find busy points from top tight pairs
            try:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                c = G[iu0]
                K = min(250, len(c))
                idx = np.argpartition(-c, K)[:K]
                freq = np.zeros(n, dtype=int)
                for k in idx:
                    a = iu0[0][k]; b = iu0[1][k]
                    freq[a] += 1; freq[b] += 1
                order_busy = np.argsort(-freq, kind="stable")
                order_rand = rng.permutation(n)
            except Exception:
                order_busy = rng.permutation(n)
                order_rand = order_busy
            if mode == 0:
                krep = int(rng.integers(10, 19))
                victims = np.unique(np.concatenate([order_busy[:krep//2], rng.choice(n, size=krep, replace=False)]))[:krep]
                keep = np.ones(n, dtype=bool); keep[victims] = False
                fixed = X[keep]
                for v in victims:
                    nh = far_hole(fixed, rng, n_try=1800)
                    X[v] = nh
                    fixed = np.vstack([fixed, nh[None, :]])
                X += 0.02 * rng.standard_normal((n, d))
            elif mode == 1:
                krep = int(rng.integers(6, 12))
                victims = np.unique(np.concatenate([order_busy[:4], rng.choice(n, size=krep, replace=False)]))[:krep+4]
                keep = np.ones(n, dtype=bool); keep[victims] = False
                fixed = X[keep]
                for v in victims:
                    nh = far_hole(fixed, rng, n_try=2200)
                    X[v] = nh
                    fixed = np.vstack([fixed, nh[None, :]])
                X += 0.035 * rng.standard_normal((n, d))
            elif mode == 2:
                # global shake large
                sig = float(rng.uniform(0.08, 0.22))
                X += sig * rng.standard_normal((n, d))
            elif mode == 3:
                # rotate a random block by random orthogonal twist
                nb = int(rng.integers(15, 35))
                blk = rng.choice(n, size=nb, replace=False)
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                ang = float(rng.uniform(0.2, 0.9))
                # interpolate between I and Q
                R = np.eye(d) * np.cos(ang) + Q * np.sin(ang)
                # re-orthogonalize
                R, _ = np.linalg.qr(R)
                if float(np.linalg.det(R)) < 0:
                    R[:, 0] = -R[:, 0]
                X[blk] = X[blk] @ R.T
                X += 0.02 * rng.standard_normal((n, d))
            elif mode == 4:
                krep = int(rng.integers(12, 22))
                victims = rng.choice(n, size=krep, replace=False)
                keep = np.ones(n, dtype=bool); keep[victims] = False
                fixed = X[keep]
                for v in victims:
                    nh = far_hole(fixed, rng, n_try=1500)
                    X[v] = nh
                    fixed = np.vstack([fixed, nh[None, :]])
                X += 0.05 * rng.standard_normal((n, d))
            elif mode == 5:
                sig = float(rng.uniform(0.03, 0.09))
                # kick busy hubs hard + global jitter
                nb = int(rng.integers(5, 12))
                victims = order_busy[:nb]
                X[victims] += 0.25 * rng.standard_normal((nb, d))
                X += sig * rng.standard_normal((n, d))
            else:
                sig = float(rng.uniform(0.04, 0.12))
                X += sig * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        if time_left > 60:
            steps = 420
        elif time_left > 28:
            steps = 320
        else:
            steps = 220
        lr0 = 0.025 if (it % 2 == 0) else 0.016
        try:
            Xq = target_quench(X, T, steps, lr0, 0.002)
        except Exception:
            continue
        vq = max_cos(Xq)
        # refine promising quenches with smooth-max balancing
        if vq < best_val + 0.0012:
            try:
                Xr = minimax_quench(Xq, steps=260 if time_left > 30 else 150, a0=80.0, a1=1200.0, lr0=0.012, lr1=0.002)
                vr = max_cos(Xr)
            except Exception:
                vr = vq; Xr = Xq
            if vr < best_val - 1e-12:
                # extra squeeze with subgradient
                try:
                    Xs = subgrad_balance(Xr, steps=250, lr0=0.008, tol=3e-4)
                    vs = max_cos(Xs)
                    if vs < vr:
                        vr, Xr = vs, Xs
                except Exception:
                    pass
                X_best = Xr.copy()
                best_val = vr
                continue
            if vq < best_val - 5e-4:
                X_best = Xq.copy()
                best_val = vq
        elif vq < best_val - 5e-4:
            X_best = Xq.copy()
            best_val = vq
    # final jam
    try:
        if time.time() - t0 < LIMIT - 3:
            Xf = minimax_quench(X_best, steps=600, a0=100.0, a1=2500.0, lr0=0.012, lr1=0.001)
            vf = max_cos(Xf)
            if vf < best_val:
                X_best, best_val = Xf, vf
    except Exception:
        pass
    try:
        if time.time() - t0 < LIMIT - 3:
            Xs = subgrad_balance(X_best, steps=500, lr0=0.008, tol=4e-4)
            vs = max_cos(Xs)
            if vs < best_val:
                X_best, best_val = Xs, vs
    except Exception:
        pass
    try:
        rem = LIMIT - (time.time() - t0) - 1.0
        if rem > 1.2:
            Xp = lp_polish_fast(X_best, seconds=min(rem, 7.0))
            vp = max_cos(Xp)
            if vp < best_val:
                X_best, best_val = Xp, vp
    except Exception:
        pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

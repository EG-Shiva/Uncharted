import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, radius_init=1.5e-3, cap=1500):
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
        sel = c > m - max(25.0 * radius, 2e-9)
        cnt_sel = int(np.sum(sel))
        if cnt_sel == 0:
            break
        if cnt_sel > cap:
            # keep largest cap
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def target_quench(X_init, T, steps, lr0, lr1):
    X = X_init.copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1p, b2p = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        p = 2.0 + 5.0 * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = G - T
        # gradient of sum max(0,D)^p
        W = np.zeros_like(G)
        pos = D > 0
        if np.any(pos):
            W[pos] = p * (D[pos] ** (p - 1.0))
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1
        b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        X = X - lr * (mh / (np.sqrt(vh) + eps))
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def contact_counts(x, tol=2e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tol, axis=1)
    return cnt, m

def solve():
    t0 = time.time()
    LIMIT = 168.0
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
        xb = rng.standard_normal((n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if time.time() - t0 < LIMIT - 30:
        xb = lp_polish(xb, seconds=3.0)
    best_val = max_cos(xb)
    X_best = xb.copy()
    it = 0
    while time.time() - t0 < LIMIT - 8:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        if time_left < 9:
            break
        if it % 8 == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(3.5e-3))))
        elif it % 3 == 0:
            delta = float(np.exp(rng.uniform(np.log(1.5e-4), np.log(7e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(2e-5), np.log(2.5e-4))))
        T = best_val - delta
        cnt, cur_m = contact_counts(X_best, tol=2e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        mode = it % 6
        X = X_best.copy()
        if mode == 0:
            n_rep = int(rng.integers(3, 7))
            victims = order_low[:n_rep].copy()
        elif mode == 1:
            n_rep = int(rng.integers(2, 5))
            victims = rng.choice(n, size=n_rep, replace=False)
        elif mode == 2:
            n_rep = int(rng.integers(2, 5))
            victims = order_high[:n_rep].copy()
        elif mode == 3:
            r = np.where(cnt <= 1)[0]
            if len(r) == 0:
                r = order_low[:4]
            n_rep = min(len(r), int(rng.integers(2, 5)))
            victims = rng.choice(r, size=n_rep, replace=False) if len(r) >= n_rep else r.copy()
        elif mode == 4:
            n_rep = int(rng.integers(4, 9))
            victims = rng.choice(n, size=n_rep, replace=False)
        else:
            victims = np.array([], dtype=int)
        sigma = float(10.0 ** rng.uniform(-2.3, -0.8))
        for v in victims:
            if rng.random() < 0.85:
                nc = 2000
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                # max dot to others
                dots = cand @ X.T
                # exclude self column approximately by setting col v to -inf via recompute
                # dots includes X[v]; replace with -1
                dots[:, int(v)] = -1.0
                worst = dots.max(axis=1)
                if rng.random() < 0.12:
                    k = int(rng.integers(1, 5))
                    idx = np.argpartition(worst, k)[:k][-1]
                else:
                    idx = int(np.argmin(worst))
                X[int(v)] = cand[idx]
            else:
                z = rng.standard_normal(d)
                z /= np.linalg.norm(z)
                X[int(v)] = z
        if len(victims) > 0:
            X[victims] += rng.standard_normal((len(victims), d)) * (sigma * 0.5)
            X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
        if mode == 5:
            gsig = float(10.0 ** rng.uniform(-2.1, -1.0))
            X += rng.standard_normal((n, d)) * gsig
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            X += rng.standard_normal((n, d)) * (sigma * 0.2)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        if time_left > 50:
            steps = 500
        elif time_left > 25:
            steps = 350
        else:
            steps = 220
        lr0 = float(0.012 + rng.random() * 0.015)
        lr1 = lr0 * 0.12
        X2 = target_quench(X, T, steps=steps, lr0=lr0, lr1=lr1)
        qv = max_cos(X2)
        if qv > best_val + 0.04 and time_left < 60:
            continue
        if time.time() - t0 >= LIMIT - 8:
            break
        polish_sec = 1.1 if time_left > 30 else 0.7
        if qv < best_val:
            polish_sec = min(polish_sec + 0.7, 2.2)
        Xp = lp_polish(X2, seconds=polish_sec)
        pv = max_cos(Xp)
        if pv < best_val - 1e-12:
            X_best = Xp
            best_val = pv
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 1.2:
        X_best = lp_polish(X_best, seconds=min(rem, 6.0), radius_init=8e-4)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

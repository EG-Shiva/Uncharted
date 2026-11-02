import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


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
        if np.sum(sel) > 2600:
            idx = np.argpartition(-c, 2600)[:2600]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
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
        p = 2.0 + 4.0 * frac  # 2 -> 6 sharpening
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
        b1p *= b1
        b2p *= b2
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


def contact_counts(x, tol=2e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tol, axis=1)
    return cnt, m


def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    if xb.shape != (n, d):
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # initial jam
    if time.time() - t0 < LIMIT - 30:
        xb = lp_polish(xb, seconds=2.5)
    best_val = max_cos(xb)
    X_best = xb.copy()

    it = 0
    while time.time() - t0 < LIMIT - 7:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        # target gap: log-uniform, sometimes tiny (polish step), sometimes big (basin jump)
        if it % 9 == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(4e-3))))
        elif it % 3 == 0:
            delta = float(np.exp(rng.uniform(np.log(2e-4), np.log(8e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(3e-5), np.log(3e-4))))
        T = best_val - delta

        cnt, cur_m = contact_counts(X_best, tol=2e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        mode = it % 7
        X = X_best.copy()

        # choose victims
        if mode == 0:
            n_rep = int(rng.integers(4, 8))
            victims = order_low[:n_rep].copy()
        elif mode == 1:
            n_rep = int(rng.integers(2, 5))
            victims = np.unique(np.concatenate(
                [order_low[:2], rng.choice(n, size=n_rep, replace=False)]))[:n_rep + 2].copy()
        elif mode == 2:
            # attack hubs
            n_rep = int(rng.integers(2, 5))
            victims = order_high[:n_rep].copy()
        elif mode == 3:
            n_rep = int(rng.integers(1, 4))
            victims = rng.choice(n, size=n_rep, replace=False)
        elif mode == 4:
            n_rep = int(rng.integers(5, 10))
            victims = rng.choice(n, size=n_rep, replace=False)
        elif mode == 5:
            # pure rattler refill: points with <=1 contact
            r = np.where(cnt <= 1)[0]
            if len(r) == 0:
                r = order_low[:4]
            n_rep = min(len(r), int(rng.integers(2, 6)))
            victims = rng.choice(r, size=n_rep, replace=False) if len(r) >= n_rep else r.copy()
        else:
            n_rep = 0
            victims = np.array([], dtype=int)

        sigma = float(10.0 ** rng.uniform(-2.4, -0.7))
        # hole-filling for victims: best of random candidates
        for v in victims:
            if rng.random() < 0.8:
                nc = 3000
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                others = np.delete(X, v, axis=0)
                dots = cand @ others.T
                worst = dots.max(axis=1)
                # with small prob pick 2nd/3rd best for diversity
                if rng.random() < 0.15:
                    k = int(rng.integers(1, 6))
                    idx = np.argpartition(worst, k)[:k][-1]
                else:
                    idx = int(np.argmin(worst))
                X[v] = cand[idx]
            else:
                z = rng.standard_normal(d)
                z /= np.linalg.norm(z)
                X[v] = z
        if len(victims) > 0 and sigma > 0:
            X[victims] += rng.standard_normal((len(victims), d)) * (sigma * 0.6)
            X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
        # global jitter
        if mode == 6:
            gsig = float(10.0 ** rng.uniform(-2.0, -0.9))
            X += rng.standard_normal((n, d)) * gsig
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            X += rng.standard_normal((n, d)) * (sigma * 0.22)
            X /= np.linalg.norm(X, axis=1, keepdims=True)

        steps = 650 if time_left > 45 else (400 if time_left > 20 else 220)
        lr0 = float(0.015 + rng.random() * 0.015)
        lr1 = lr0 * 0.12
        X2 = target_quench(X, T, steps=steps, lr0=lr0, lr1=lr1)
        qv = max_cos(X2)
        # skip expensive LP if far worse
        if qv > best_val + 0.035 and time.time() - t0 < LIMIT - 25:
            continue
        polish_sec = 1.6 if time_left > 30 else 0.9
        # occasionally give more polish when promising (below target)
        if qv < best_val:
            polish_sec = min(polish_sec + 0.8, 2.6)
        Xp = lp_polish(X2, seconds=polish_sec)
        pv = max_cos(Xp)
        if pv < best_val - 1e-12:
            X_best = Xp
            best_val = pv
        if time.time() - t0 >= LIMIT - 7:
            break

    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 1.5:
        X_best = lp_polish(X_best, seconds=min(rem, 7.0), radius_init=1e-3)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

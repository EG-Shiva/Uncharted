import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contact_counts(x, tol=1.5e-3):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = np.sum(g > m - tol, axis=1)
    return cnt, m

def lp_polish(x, deadline, radius_init=2e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while radius > 1e-14:
        if time.time() >= deadline:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > cap:
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def quench_lse(X_init, steps, lr0, lr1, b0, b1):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    p1, p2 = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    e0, e1 = np.log(b0), np.log(b1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        # avoid overflow: exp(beta*(G-gm)), clip
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        # symmetrize (already symmetric) and weight
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        p1 *= b_1
        p2 *= b_2
        m1 = b_1 * m1 + (1.0 - b_1) * Gt
        m2 = b_2 * m2 + (1.0 - b_2) * (Gt * Gt)
        mh = m1 / (1.0 - p1)
        vh = m2 / (1.0 - p2)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def far_hole_pick(fixed, rng, n_try=2000):
    d = fixed.shape[1]
    Z = rng.standard_normal((n_try, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ fixed.T
    m = S.max(axis=1)
    return Z[int(np.argmin(m))]

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(42)
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
    # quick initial jam
    if time.time() - t0 < LIMIT - 30:
        xb = lp_polish(xb, deadline=time.time() + 2.2, radius_init=2e-3)
    best_val = max_cos(xb)
    X_best = xb.copy()

    it = 0
    iu_all = np.triu_indices(n, 1)
    while time.time() - t0 < LIMIT - 8:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        if time_left < 9:
            break
        cnt, cur_m = contact_counts(X_best, tol=1.5e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        mode = it % 6
        X = X_best.copy()

        if mode == 0:
            nv = int(rng.integers(3, 7))
            victims = order_low[:nv].copy()
            for v in victims:
                others = np.delete(X, v, axis=0)
                if rng.random() < 0.9:
                    X[v] = far_hole_pick(others, rng, n_try=2000)
                else:
                    z = rng.standard_normal(d); z /= np.linalg.norm(z); X[v] = z
            sig = float(10.0 ** rng.uniform(-2.2, -1.0))
            X[victims] += rng.standard_normal((len(victims), d)) * (sig * 0.4)
            X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
            X += rng.standard_normal((n, d)) * (sig * 0.18)
        elif mode == 1:
            # worst-pair attack: find top pairs, kick one endpoint
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # get worst ~6 pairs
            flat = G[iu_all]
            k = 6
            idx = np.argpartition(-flat, k)[:k]
            victims = np.unique(np.concatenate([iu_all[0][idx], iu_all[1][idx]]))
            # keep only few
            if len(victims) > 5:
                victims = rng.choice(victims, size=5, replace=False)
            sig = float(10.0 ** rng.uniform(-1.8, -0.7))
            X[victims] += rng.standard_normal((len(victims), d)) * sig
            X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
            X += rng.standard_normal((n, d)) * (sig * 0.15)
        elif mode == 2:
            nv = int(rng.integers(2, 5))
            victims = order_high[:nv].copy()
            for v in victims:
                others = np.delete(X, v, axis=0)
                X[v] = far_hole_pick(others, rng, n_try=2000)
            sig = float(10.0 ** rng.uniform(-2.0, -0.9))
            X += rng.standard_normal((n, d)) * sig
        elif mode == 3:
            nv = int(rng.integers(1, 4))
            victims = rng.choice(n, size=nv, replace=False)
            for v in victims:
                others = np.delete(X, v, axis=0)
                X[v] = far_hole_pick(others, rng, n_try=1500)
            sig = float(10.0 ** rng.uniform(-2.4, -1.2))
            X += rng.standard_normal((n, d)) * sig
        elif mode == 4:
            nv = int(rng.integers(5, 9))
            victims = rng.choice(n, size=nv, replace=False)
            for v in victims:
                if rng.random() < 0.7:
                    others = np.delete(X, v, axis=0)
                    X[v] = far_hole_pick(others, rng, n_try=1200)
                else:
                    z = rng.standard_normal(d); z /= np.linalg.norm(z); X[v] = z
            sig = float(10.0 ** rng.uniform(-1.6, -0.7))
            X += rng.standard_normal((n, d)) * (sig * 0.5)
        else:
            sig = float(10.0 ** rng.uniform(-2.0, -0.9))
            X += rng.standard_normal((n, d)) * sig
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        if time_left > 50:
            steps = 550
        elif time_left > 25:
            steps = 380
        else:
            steps = 240
        lr0 = float(0.012 + rng.random() * 0.020)
        lr1 = lr0 * float(0.08 + rng.random() * 0.08)
        b0 = float(10.0 ** rng.uniform(1.4, 2.0))   # ~25-100
        b1 = float(10.0 ** rng.uniform(2.4, 3.4))   # ~250-2500
        # occasional sharp finish
        if rng.random() < 0.2:
            b1 = float(10.0 ** rng.uniform(3.0, 3.7))

        X2 = quench_lse(X, steps=steps, lr0=lr0, lr1=lr1, b0=b0, b1=b1)
        qv = max_cos(X2)
        if qv > best_val + 0.03 and time.time() - t0 < LIMIT - 25:
            continue
        # selective LP
        if qv < best_val + 0.004 or rng.random() < 0.12:
            if time_left > 30:
                sec = 1.4 if qv >= best_val else 2.2
            else:
                sec = 0.8 if qv >= best_val else 1.4
            deadline = min(time.time() + sec, t0 + LIMIT - 7)
            Xp = lp_polish(X2, deadline=deadline, radius_init=2e-3)
            pv = max_cos(Xp)
            if pv < best_val - 1e-12:
                X_best = Xp
                best_val = pv
        else:
            if qv < best_val - 1e-12:
                X_best = X2
                best_val = qv
        if time.time() - t0 >= LIMIT - 7:
            break

    rem = (t0 + LIMIT - 1.0) - time.time()
    if rem > 1.2:
        X_best = lp_polish(X_best, deadline=time.time() + min(rem, 6.0), radius_init=1e-3)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

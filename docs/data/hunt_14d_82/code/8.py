import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_info(x, tol=2e-3):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tol, axis=1)
    per = G.max(axis=1)
    return cnt, per, float(m), G


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-15:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = max(20.0 * radius, 5e-10)
        sel = c > m - thr
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > 3000:
            k = 3000
            idx = np.argpartition(-c, k)[:k]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
            ns = int(np.sum(sel))
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p = ns
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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


def anneal_quench(x, steps=500, beta0=10.0, beta1=120.0,
                  lr0=0.03, lr1=0.003, mu=0.75, mask=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        scale = float(W.sum() / n)
        if scale < 1e-12:
            scale = 1.0
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        grad = grad / scale
        rn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 1.0 / rn)
        grad = grad * clip
        vel = mu * vel - lr * grad
        vdot = np.sum(vel * x, axis=1, keepdims=True)
        vel = vel - vdot * x
        if mask is not None:
            step = np.zeros_like(x)
            step[mask] = vel[mask]
        else:
            step = vel
        x = x + step
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def greedy_refill(X, victims, rng, n_cand=3000):
    n, d = X.shape
    fixed_idx = np.ones(n, dtype=bool)
    fixed_idx[victims] = False
    for v in victims:
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = X[fixed_idx]
        dots = cand @ others.T
        s = dots.max(axis=1)
        b = int(np.argmin(s))
        X[v] = cand[b]
        fixed_idx[v] = True
    return X


def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
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
        m0 = min(xb.shape[0], n)
        d0 = min(xb.shape[1], d)
        try:
            z[:m0, :d0] = xb[:m0, :d0]
            z /= np.linalg.norm(z, axis=1, keepdims=True)
        except Exception:
            pass
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    if time.time() - t0 < LIMIT - 20:
        xb = lp_polish(xb, seconds=2.0, radius_init=1e-3)
        best_val = max_cos(xb)
    X_best = xb.copy()

    it = 0
    while time.time() - t0 < LIMIT - 6:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        polish_sec = 2.0 if time_left > 35 else 1.0
        cnt, per, cur_m, G = contact_info(X_best, tol=2e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]

        r = rng.random()
        X = X_best.copy()
        # choose avalanche
        if r < 0.55:
            # neighbourhood avalanche around overloaded/random centre
            if rng.random() < 0.65:
                centre = int(order_high[int(rng.integers(0, min(12, n)))])
            else:
                centre = int(rng.integers(0, n))
            # k nearest by dot
            sims = G[centre].copy()
            sims[centre] = -2.0
            k = int(rng.integers(4, 13))
            if rng.random() < 0.12:
                k = int(rng.integers(13, 22))
            victims = np.argsort(-sims)[:k]
            # include centre sometimes
            if rng.random() < 0.7 and centre not in victims:
                victims = np.concatenate([[centre], victims])[:k + 1]
            victims = np.unique(victims.astype(int))
            X = greedy_refill(X, victims, rng, n_cand=3000)
            sigma = float(10.0 ** rng.uniform(-2.4, -0.8))
            X[victims] += rng.standard_normal((len(victims), d)) * (sigma * 0.6)
            X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
            # hole-only quench then global
            b0 = float(8 + rng.random() * 8)
            b1 = float(90 + rng.random() * 80)
            hole_mask = np.zeros(n, dtype=bool)
            hole_mask[victims] = True
            X = anneal_quench(X, steps=250, beta0=b0, beta1=b1,
                              lr0=0.03, lr1=0.004, mu=0.75, mask=hole_mask)
            X = anneal_quench(X, steps=350 if time_left > 40 else 200,
                              beta0=b0, beta1=b1 * 1.2,
                              lr0=0.03, lr1=0.003, mu=0.75, mask=None)
        elif r < 0.75:
            # rattler reinsertion: attack loosest
            n_rep = int(rng.integers(3, 9))
            victims = order_low[:n_rep]
            X = greedy_refill(X, victims, rng, n_cand=4000)
            sigma = float(10.0 ** rng.uniform(-2.2, -1.0))
            X += rng.standard_normal((n, d)) * (sigma * 0.3)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            b0 = float(8 + rng.random() * 6)
            b1 = float(80 + rng.random() * 60)
            X = anneal_quench(X, steps=400 if time_left > 40 else 220,
                              beta0=b0, beta1=b1,
                              lr0=0.03, lr1=0.003, mu=0.75)
        else:
            # global kick with anneal (basin hop)
            sigma = float(10.0 ** rng.uniform(-1.8, -0.5))
            X += rng.standard_normal((n, d)) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            b0 = float(6 + rng.random() * 8)
            b1 = float(70 + rng.random() * 90)
            X = anneal_quench(X, steps=450 if time_left > 40 else 220,
                              beta0=b0, beta1=b1,
                              lr0=0.035, lr1=0.003, mu=0.78)

        qv = max_cos(X)
        if qv > best_val + 0.04 and (time.time() - t0) < LIMIT - 20:
            continue
        Xp = lp_polish(X, seconds=polish_sec, radius_init=2e-3)
        pv = max_cos(Xp)
        if pv < best_val - 1e-12:
            X_best = Xp
            best_val = pv
        if time.time() - t0 >= LIMIT - 6:
            break

    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 1.2:
        X_best = lp_polish(X_best, seconds=min(rem, 7.0), radius_init=7e-4)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

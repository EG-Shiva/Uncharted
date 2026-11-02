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
    m = float(np.max(g))
    cnt = np.sum(g > m - tol, axis=1)
    return cnt, m

def lp_polish(x, seconds, radius_init=2e-3, max_solves=60):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() - start < seconds and radius > 1e-14 and solves < max_solves:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > 2000:
            idx = np.argpartition(-c, 2000)[:2000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
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

def soft_quench(x, steps, beta0, beta1, lr0, lr1, mu=0.75):
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
        grad -= dot * x
        grad /= scale
        rn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad = grad * np.minimum(1.0, 1.0 / rn)
        vel = mu * vel - lr * grad
        vdot = np.sum(vel * x, axis=1, keepdims=True)
        vel -= vdot * x
        x = x + vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def target_quench(X_init, T, steps, lr0, lr1):
    X = X_init.copy()
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

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(0)
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
    if time.time() - t0 < LIMIT - 30:
        xb = lp_polish(xb, seconds=2.0)
    best_val = max_cos(xb)
    X_best = xb.copy()
    it = 0
    while time.time() - t0 < LIMIT - 7:
        it += 1
        time_left = LIMIT - (time.time() - t0)
        if time_left < 7:
            break
        # target gap
        r = rng.random()
        if it % 9 == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(4e-3))))
        elif it % 3 == 0:
            delta = float(np.exp(rng.uniform(np.log(2e-4), np.log(8e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(3e-5), np.log(3e-4))))
        # occasional large jump
        if r < 0.08:
            delta = float(np.exp(rng.uniform(np.log(2e-3), np.log(6e-3))))
        T = best_val - delta
        cnt, cur_m = contact_counts(X_best, tol=2e-3)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        mode = it % 7
        X = X_best.copy()
        log_sigma = rng.uniform(-2.6, -0.6)
        sigma = 10.0 ** log_sigma
        if mode == 0:
            n_rep = int(rng.integers(4, 8))
            victims = order_low[:n_rep].copy()
            use_greedy = True
        elif mode == 1:
            n_rep = int(rng.integers(2, 5))
            victims = np.unique(np.concatenate([order_low[:2], rng.choice(n, size=n_rep, replace=False)]))[:n_rep+1]
            use_greedy = True
        elif mode == 2:
            n_rep = int(rng.integers(1, 4))
            victims = rng.choice(n, size=n_rep, replace=False)
            use_greedy = rng.random() < 0.7
        elif mode == 3:
            n_rep = int(rng.integers(2, 5))
            victims = order_high[:n_rep].copy()
            use_greedy = False
        elif mode == 4:
            victims = np.array([], dtype=int)
            use_greedy = False
            sigma = float(10.0 ** rng.uniform(-2.0, -0.9))
        elif mode == 5:
            n_rep = int(rng.integers(3, 7))
            victims = rng.choice(n, size=n_rep, replace=False)
            use_greedy = True
        else:
            n_rep = int(rng.integers(5, 10))
            victims = np.unique(np.concatenate([order_low[:3], rng.choice(n, size=n_rep, replace=False)]))[:n_rep+2]
            use_greedy = True
        if len(victims) > 0:
            for v in victims:
                if use_greedy:
                    cand = rng.standard_normal((1500, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    others = np.delete(X, int(v), axis=0)
                    dots = cand @ others.T
                    s = dots.max(axis=1)
                    X[int(v)] = cand[int(np.argmin(s))]
                else:
                    z = rng.standard_normal(d)
                    z /= np.linalg.norm(z)
                    X[int(v)] = z
            nz = rng.standard_normal((len(victims), d)) * (sigma * 0.7)
            X[victims] += nz
            X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True)
        if len(victims) > 0:
            X += rng.standard_normal((n, d)) * (sigma * 0.25)
        else:
            X += rng.standard_normal((n, d)) * sigma
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        steps = 350 if time_left > 40 else 200
        if time_left < 20:
            steps = 150
        b0 = float(8 + rng.random() * 6)
        b1 = float(80 + rng.random() * 60)
        # alternate quench type
        if it % 2 == 0:
            tsteps = steps // 2
            X = target_quench(X, T, steps=tsteps, lr0=0.005, lr1=0.0008)
            X = soft_quench(X, steps=tsteps, beta0=b0, beta1=b1, lr0=0.03, lr1=0.003, mu=0.75)
        else:
            X = soft_quench(X, steps=steps, beta0=b0, beta1=b1, lr0=0.03, lr1=0.003, mu=0.75)
            # short target refinement
            if rng.random() < 0.5:
                X = target_quench(X, T, steps=60, lr0=0.003, lr1=0.0008)
        qv = max_cos(X)
        if qv > best_val + 0.04 and time_left > 20:
            continue
        polish_sec = 1.8 if time_left > 30 else 0.8
        # only polish if promising or occasionally
        if qv > best_val + 0.015 and (it % 4 != 0):
            continue
        Xp = lp_polish(X, seconds=polish_sec)
        pv = max_cos(Xp)
        if pv < best_val - 1e-12:
            X_best = Xp
            best_val = pv
        if time.time() - t0 >= LIMIT - 7:
            break
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 1.2:
        X_best = lp_polish(X_best, seconds=min(rem, 6.0), radius_init=1e-3)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

import time
import numpy as np


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def quench_lse(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    X = X.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = beta0 * ((beta1 / beta0) ** frac) if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * frac
        D = X @ X.T
        np.fill_diagonal(D, -2.0)
        dm = D.max()
        # stable weights <=1
        W = np.exp(beta * (D - dm))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if not np.isfinite(S) or S == 0:
            break
        G = (W @ X) / S
        proj = (G * X).sum(axis=1, keepdims=True)
        Gt = G - proj * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1 ** (s + 1))
        vhat = v1 / (1.0 - b2 ** (s + 1))
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X


def cheap_polish(X, steps=250, lr0=0.005):
    X = X.copy()
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    best = max_cos(X)
    bestX = X.copy()
    for s in range(steps):
        D = X @ X.T
        np.fill_diagonal(D, -2.0)
        m = D.max()
        delta = 0.005 + 0.01 * (1.0 - s / max(1, steps))
        mask = D > (m - delta)
        deg = mask.sum(axis=1, keepdims=True)
        G = mask @ X
        G = G / np.maximum(deg, 1)
        proj = (G * X).sum(axis=1, keepdims=True)
        Gt = G - proj * X
        lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
        Xn = X - lr * Gt
        nrm = np.linalg.norm(Xn, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = Xn / nrm
        if (s % 10 == 9) or (s == steps - 1):
            v = max_cos(X)
            if v < best:
                best = v
                bestX = X.copy()
    return bestX


def lp_polish(x, seconds, t_start=None, t_limit=174.0):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_start is not None and (time.time() - t_start) >= t_limit:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def rand_sphere(rng, n, d):
    a = rng.normal(size=(n, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def solve():
    t0 = time.time()
    TIME_BUDGET = 174.0
    EXPLORE_END = 142.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (153, 4):
        # try to fix shape if transposed
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    X_best = xb.copy()

    rng = np.random.default_rng(20261031)
    n, d = 153, 4

    # precompute contact info helper inline
    it = 0
    # candidate pool: keep best
    while time.time() - t0 < EXPLORE_END:
        it += 1
        u = rng.random()
        # base to perturb: mostly best, sometimes original
        if rng.random() < 0.85:
            base = X_best
        else:
            base = xb
        X = base.copy()

        if u < 0.40:
            # global gaussian shake
            sigma = float(10.0 ** rng.uniform(-2.0, -0.4))
            X = X + sigma * rng.normal(size=(n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif u < 0.65:
            # subset replace with random
            mrep = int(rng.integers(1, 7))
            idx = rng.choice(n, size=mrep, replace=False)
            R = rand_sphere(rng, mrep, d)
            # small blend to keep near? full replace
            X[idx] = R
            # tiny shake rest
            sigma = float(10.0 ** rng.uniform(-2.5, -1.2))
            X += sigma * rng.normal(size=(n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif u < 0.85:
            # hole filling: move worst points to empty caps
            D = X @ X.T
            np.fill_diagonal(D, -2.0)
            mx = D.max()
            # degrees
            deg = (D > mx - 1e-4).sum(axis=1)
            # pick 1-3 points with highest max-contact (worst)
            pmax = D.max(axis=1)
            order = np.argsort(-pmax)
            mrep = int(rng.integers(1, 4))
            cand = rand_sphere(rng, 2500, d)
            for k in range(mrep):
                ii = int(order[k if k < n else 0])
                # dot of candidates to all except ii
                # compute max dot
                dots = cand @ X.T  # 2500 x 153
                # ignore column ii
                dots[:, ii] = -2.0
                md = dots.max(axis=1)
                bi = int(np.argmin(md))
                X[ii] = cand[bi]
            # small shake
            X += 0.01 * rng.normal(size=(n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # large kick of low-degree (rattler) region + global
            D = X @ X.T
            np.fill_diagonal(D, -2.0)
            mx = D.max()
            deg = (D > mx - 5e-4).sum(axis=1)
            ratt = np.where(deg == 0)[0]
            if len(ratt) > 0:
                mrep = min(len(ratt), int(rng.integers(1, 4)))
                idx = rng.choice(ratt, size=mrep, replace=False)
                X[idx] = rand_sphere(rng, mrep, d)
            else:
                idx = rng.choice(n, size=int(rng.integers(1, 4)), replace=False)
                X[idx] = rand_sphere(rng, len(idx), d)
            sigma = float(rng.uniform(0.02, 0.12))
            X += sigma * rng.normal(size=(n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)

        steps = int(rng.integers(400, 1400))
        beta0 = 20.0
        beta1 = float(rng.uniform(80.0, 260.0))
        lr0 = float(rng.uniform(0.015, 0.06))
        lr1 = lr0 * 0.1
        Xq = quench_lse(X, steps, beta0, beta1, lr0, lr1)
        # cheap contact descent
        csteps = int(rng.integers(150, 350))
        lr_c = float(rng.uniform(0.003, 0.010))
        Xq2 = cheap_polish(Xq, steps=csteps, lr0=lr_c)
        v = max_cos(Xq2)
        if v < best_val:
            best_val = v
            X_best = Xq2.copy()
            # quick extra polish if promising and time allows
            # do a short LP burst of 1.5s to get true basin value
            if time.time() - t0 < EXPLORE_END - 5:
                Xp = lp_polish(X_best, seconds=1.5, t_start=t0, t_limit=TIME_BUDGET)
                vp = max_cos(Xp)
                if vp < best_val:
                    best_val = vp
                    X_best = Xp.copy()

        # safety time check
        if time.time() - t0 >= EXPLORE_END:
            break

    remain = TIME_BUDGET - (time.time() - t0) - 1.0
    if remain > 2.0:
        remain = min(remain, 30.0)
        X_best = lp_polish(X_best, seconds=remain, t_start=t0, t_limit=TIME_BUDGET)

    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def quench_adam(Y0, steps=400, beta0=80.0, beta1=300.0, lr=0.02):
    n, d = Y0.shape
    Y = Y0.copy()
    m1 = np.zeros_like(Y)
    v1 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, steps + 1):
        frac = it / steps
        beta = beta0 * (beta1 / beta0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Grad = E @ Y
        # tangent projection
        dot = (Grad * Y).sum(axis=1, keepdims=True)
        Gt = Grad - dot * Y
        # adam
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        mhat = m1 / (1 - b1 ** it)
        vhat = v1 / (1 - b2 ** it)
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        # keep tangent
        ddot = (delta * Y).sum(axis=1, keepdims=True)
        delta = delta - ddot * Y
        # clip large steps
        dn = np.linalg.norm(delta, axis=1, keepdims=True)
        cap = 0.05
        scale = np.minimum(1.0, cap / (dn + 1e-12))
        delta = delta * scale
        Y = Y - delta
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    # quick initial snap
    if time.time() - t0 < LIMIT - 20:
        remain = LIMIT - 20 - (time.time() - t0)
        bestX = lp_polish(bestX, min(2.0, max(0.3, remain * 0.1)))
        bestV = max_cos(bestX)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < LIMIT - 12:
        it += 1
        time_left = LIMIT - 12 - (time.time() - t0)
        if time_left < 2.0:
            break
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        mcur = G.max()
        cnts = (G > mcur - 1e-9).sum(axis=1)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 3
        if mode == 0:
            # gaussian kick on low-contact points
            k = int(rng.integers(1, 4))  # 1..3
            pool = order[:16]
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            sigma = float(rng.uniform(0.05, 0.18))
            noise = rng.standard_normal((len(chosen), d))
            Xtry[chosen] += sigma * noise
            # tiny global jitter
            Xtry += 0.005 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 350
        elif mode == 1:
            # hole reinsertion: replace 1-2 worst points by best random hole
            k = 1 if (it % 2 == 0) else 2
            victims = order[:k].copy()
            # optionally randomize one victim
            if rng.random() < 0.3:
                victims[0] = rng.integers(0, n)
            for v in victims:
                cand = rng.standard_normal((1200, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xtry.T
                # ignore self column
                dots[:, v] = -2.0
                md = dots.max(axis=1)
                bi = int(np.argmin(md))
                Xtry[v] = cand[bi]
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 400
        else:
            # global shake
            sigma = float(rng.uniform(0.015, 0.05))
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 300
        # quench
        beta0 = float(rng.uniform(60, 120))
        beta1 = float(rng.uniform(220, 350))
        lr = float(rng.uniform(0.012, 0.030))
        Xq = quench_adam(Xtry, steps=steps, beta0=beta0, beta1=beta1, lr=lr)
        vq = max_cos(Xq)
        # accept / polish if promising
        thresh = bestV + 0.004
        if vq < thresh:
            budget = 1.2 if vq < bestV + 0.001 else 0.6
            budget = min(budget, time_left - 0.5)
            if budget > 0.25:
                Xp = lp_polish(Xq, budget)
                vp = max_cos(Xp)
                if vp < bestV:
                    bestX, bestV = Xp, vp
                    continue
                if vq < bestV:
                    bestX, bestV = Xq, vq
                    continue
            else:
                if vq < bestV:
                    bestX, bestV = Xq, vq
        else:
            # occasionally keep slightly worse to diversify? no, keep best basin
            pass
        # hard timeout guard
        if time.time() - t0 > LIMIT - 12:
            break
    # final squeeze with remaining time
    rem = LIMIT - (time.time() - t0)
    if rem > 0.5:
        bestX = lp_polish(bestX, min(rem - 0.2, 8.0))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

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
    start = time.time()
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < 40:
        it += 1
        Gf = X @ X.T
        c = Gf[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 2e-9)
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
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.5, seconds - (time.time()-start))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def quench(X0, steps, beta0, beta1, lr0, lr1, rng=None):
    n, d = X0.shape
    X = X0.copy()
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        gv = G[iu]
        m = gv.max()
        e = np.exp(beta * (gv - m))
        S = e.sum()
        w = e / S
        W = np.zeros((n, n))
        W[iu[0], iu[1]] = w
        W = W + W.T
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Tg = Gr - dot * X
        # normalize step by max tangent norm to keep stable
        tn = np.linalg.norm(Tg, axis=1, keepdims=True).max()
        if tn > 1e-12:
            step = lr / max(tn, 1e-9)
            # cap
            step = min(step, 0.2)
            X = X - step * Tg * (tn if False else 1.0)
            # actually use lr directly: X - lr*Tg/mean scale
            # redo correctly: use lr scaled
            # we already did; fix: undo double scaling -> simple:
            # X = X0_step - (lr/tn)*Tg
            pass
        else:
            break
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def quench_fast(X0, steps, beta0, beta1, lr):
    # simpler stable version: X -= lr * tangent
    n, d = X0.shape
    X = X0.copy()
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        G = X @ X.T
        gv = G[iu]
        m = gv.max()
        # avoid overflow: e in [0,1]
        e = np.exp(beta * (gv - m))
        S = e.sum()
        w = e / S
        W = np.zeros((n, n))
        W[iu[0], iu[1]] = w
        W += W.T
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Tg = Gr - dot * X
        X = X - lr * Tg
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def largest_void_vector(X, exclude_idx, rng, n_cand=2000):
    n, d = X.shape
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # max dot with others (excluding self)
    mask = np.ones(n, dtype=bool); mask[exclude_idx] = False
    dots = cand @ X[mask].T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    # quick snap
    if LIMIT - (time.time()-t0) > 20:
        bestX = lp_polish(bestX, 2.0)
        bestV = max_cos(bestX)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < LIMIT - 12:
        it += 1
        time_left = LIMIT - 12 - (time.time() - t0)
        if time_left < 4:
            break
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        cnts = (G > m - 1e-9).sum(axis=1)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        mode = it % 5
        if mode == 0:
            k = 1; sigma = 0.06 + 0.02*(it%3)
            chosen = order[:k]
            for idx in chosen:
                v = largest_void_vector(bestX, idx, rng, n_cand=3000)
                # mix with noise to keep diversity
                nz = rng.standard_normal(d); nz /= np.linalg.norm(nz)
                vv = v + 0.05*rng.standard_normal(d)
                vv /= np.linalg.norm(vv)
                Xtry[idx] = vv
            # small global jitter
            Xtry += 0.005 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, b0, b1, lr = 500, 400.0, 4000.0, 0.6
            pol_sec = 2.0
        elif mode == 1:
            k = 2; chosen = order[:8]
            sel = rng.choice(chosen, size=k, replace=False)
            for idx in sel:
                v = largest_void_vector(bestX, idx, rng, n_cand=2500)
                Xtry[idx] = v
            Xtry += 0.01 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, b0, b1, lr = 400, 300.0, 3000.0, 0.5
            pol_sec = 1.5
        elif mode == 2:
            k = 3; pool = order[:16]
            sel = rng.choice(pool, size=k, replace=False)
            sigma = float(rng.uniform(0.08, 0.20))
            for idx in sel:
                pert = Xtry[idx] + sigma * rng.standard_normal(d)
                pert /= np.linalg.norm(pert)
                Xtry[idx] = pert
            # also reinsert worst one into void
            v = largest_void_vector(bestX, order[0], rng, n_cand=2000)
            if rng.random() < 0.5:
                Xtry[order[0]] = v
            steps, b0, b1, lr = 400, 500.0, 5000.0, 0.4
            pol_sec = 1.5
        elif mode == 3:
            sigma = float(rng.uniform(0.03, 0.10))
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, b0, b1, lr = 600, 400.0, 4000.0, 0.5
            pol_sec = 2.0
        else:
            # targeted: move overloaded + rattler together
            overloaded = np.argsort(-cnts, kind='stable')[:2]
            sel = list(order[:2]) + list(overloaded[:1])
            for idx in sel:
                if rng.random() < 0.7:
                    v = largest_void_vector(bestX, idx, rng, n_cand=2500)
                    Xtry[idx] = v
                else:
                    pert = Xtry[idx] + 0.15 * rng.standard_normal(d)
                    Xtry[idx] = pert / np.linalg.norm(pert)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, b0, b1, lr = 450, 600.0, 6000.0, 0.35
            pol_sec = 1.5
        # fast quench (numpy only)
        Xtry = quench_fast(Xtry, steps, b0, b1, lr)
        vq = max_cos(Xtry)
        # only LP-polish if promising (saves time)
        if vq > bestV + 0.004 and it > 3:
            continue
        # adapt polish time to remaining
        pol_sec = min(pol_sec, max(0.6, time_left - 2))
        Xpol = lp_polish(Xtry, pol_sec)
        vp = max_cos(Xpol)
        if vp < bestV:
            bestX = Xpol; bestV = vp
        # also consider unpolished if much better topology for future? no, keep best polished
        if time.time() - t0 > LIMIT - 12:
            break
    # final squeeze
    if time.time() - t0 < LIMIT - 1:
        remain = max(0.5, min(6.0, LIMIT - 1 - (time.time()-t0)))
        bestX = lp_polish(bestX, remain)
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

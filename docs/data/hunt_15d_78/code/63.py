import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, radius_init=4e-3, cap=1200):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius_init
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    iu0 = iu0.astype(np.int64); iu1 = iu1.astype(np.int64)
    ar = np.arange(d)
    obj = np.zeros(n * d + 1); obj[-1] = 1.0
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = float(c.max())
        thr = m - max(15.0 * radius, 2e-9)
        sel = c > thr
        if int(np.count_nonzero(sel)) == 0:
            break
        if int(np.count_nonzero(sel)) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[sel]; J = iu1[sel]
        csv = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return X

def _contacts(X, tol=1.5e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    return C.sum(axis=1), m, G

def _hole_vec(X_keep, rng, n_cand=4000):
    d = X_keep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_keep.T
    md = dots.max(axis=1)
    return cand[int(np.argmin(md))]

def _quench(X, steps=200, beta=600.0, s0=0.02, s1=0.001):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        step = s0 * (1 - frac) + s1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-18
        Gr = (E @ X) / rs
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        nn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-18
        X = X - step * Gt / nn
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _whiten(X, eps=1e-6):
    n, d = X.shape
    C = (X.T @ X) / n
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    # target covariance I/d
    s = 1.0 / np.sqrt(w * d)
    W = (V * s[None, :]) @ V.T
    Y = X @ W
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _center(X):
    c = X.mean(axis=0, keepdims=True)
    Y = X - c
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(1)
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    # quick tighten of loaded best (it is already tight)
    if time.time() - t0 < LIMIT - 12.0:
        bestX = _lp_polish(bestX, seconds=2.0, radius_init=1.5e-3)
        bestV = _max_cos(bestX)

    it = 0
    while time.time() - t0 < LIMIT - 10.0:
        it += 1
        time_left = LIMIT - 10.0 - (time.time() - t0)
        if time_left < 3.0:
            break
        cnts, mcur, Gcur = _contacts(bestX)
        order_lo = np.argsort(cnts, kind='stable')
        order_hi = order_lo[::-1]
        Xt = bestX.copy()
        mode = it % 7
        if mode == 0:
            # whiten + center to enforce isotropy, then small noise
            Xt = _center(Xt)
            Xt = _whiten(Xt)
            Xt = Xt + 0.02 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _quench(Xt, steps=180, beta=500.0, s0=0.015, s1=0.001)
        elif mode == 1:
            # PC-strata intermix: split by PC1, pull samples toward opposite side
            Xm = Xt - Xt.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
            pc1 = (Xm @ Vt[0])
            lo = np.argsort(pc1)
            left = lo[:n // 2]; right = lo[n // 2:]
            # pick a few from each side and move toward opposite centroid
            cl = Xt[left].mean(axis=0); cr = Xt[right].mean(axis=0)
            cl /= np.linalg.norm(cl); cr /= np.linalg.norm(cr)
            k = 6
            pl = rng.choice(left, size=k, replace=False)
            pr = rng.choice(right, size=k, replace=False)
            for idx in pl:
                Xt[idx] = Xt[idx] * 0.7 + cr * 0.5 + 0.08 * rng.standard_normal(d)
                Xt[idx] /= np.linalg.norm(Xt[idx])
            for idx in pr:
                Xt[idx] = Xt[idx] * 0.7 + cl * 0.5 + 0.08 * rng.standard_normal(d)
                Xt[idx] /= np.linalg.norm(Xt[idx])
            Xt = _quench(Xt, steps=200, beta=600.0, s0=0.018, s1=0.001)
        elif mode == 2:
            # kill lowest-contact points into holes
            k = int(rng.integers(2, 5))
            victims = order_lo[:k].copy()
            if rng.random() < 0.5:
                victims = rng.choice(order_lo[:16], size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Xkeep = Xt[keep]
            for vi in victims:
                hv = _hole_vec(Xkeep, rng, n_cand=4000)
                hv = hv + 0.01 * rng.standard_normal(d)
                hv /= np.linalg.norm(hv)
                Xt[vi] = hv
                Xkeep = np.vstack([Xkeep, hv[None, :]])
            Xt = Xt + 0.012 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _quench(Xt, steps=180, beta=550.0, s0=0.015, s1=0.001)
        elif mode == 3:
            # kill hub (highest contact) into hole + shake rattler
            hub = int(order_hi[0]); low = int(order_lo[0])
            keep = np.ones(n, dtype=bool); keep[hub] = False
            hv = _hole_vec(Xt[keep], rng, n_cand=5000)
            Xt[hub] = hv
            Xt[low] = Xt[low] + 0.3 * rng.standard_normal(d)
            Xt[low] /= np.linalg.norm(Xt[low])
            Xt = _quench(Xt, steps=200, beta=700.0, s0=0.02, s1=0.001)
        elif mode == 4:
            # global shake
            sig = float(rng.uniform(0.04, 0.10))
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _quench(Xt, steps=220, beta=600.0, s0=0.02, s1=0.001)
        elif mode == 5:
            # center + moderate tail shake
            Xt = _center(Xt)
            pool = order_lo[:12]
            sig = float(rng.uniform(0.12, 0.30))
            for idx in pool:
                if rng.random() < 0.7:
                    Xt[idx] = Xt[idx] + sig * rng.standard_normal(d)
                    Xt[idx] /= np.linalg.norm(Xt[idx])
            Xt = _quench(Xt, steps=200, beta=650.0, s0=0.018, s1=0.001)
        else:
            # random orthogonal mix in a random 2-plane for a subset
            a, b = rng.standard_normal(d), rng.standard_normal(d)
            # orthonormalize
            a /= np.linalg.norm(a); b = b - (b @ a) * a; b /= np.linalg.norm(b)
            subset = rng.choice(n, size=12, replace=False)
            angs = rng.uniform(-0.4, 0.4, size=len(subset))
            for ii, idx in enumerate(subset):
                xa, xb = Xt[idx] @ a, Xt[idx] @ b
                # rotate components
                ca, sa = np.cos(angs[ii]), np.sin(angs[ii])
                # remove old plane comps, add rotated
                v = Xt[idx] - xa * a - xb * b
                na, nb = xa * ca - xb * sa, xa * sa + xb * ca
                Xt[idx] = v + na * a + nb * b
                Xt[idx] /= np.linalg.norm(Xt[idx])
            Xt = _quench(Xt, steps=180, beta=550.0, s0=0.016, s1=0.001)

        vq = _max_cos(Xt)
        if vq > bestV + 0.004:
            continue
        budget = 1.4
        if vq < bestV + 0.0005:
            budget = 2.4
        budget = min(budget, time_left - 1.0)
        if budget < 0.8:
            continue
        if time.time() - t0 > LIMIT - 10.0:
            break
        Xp = _lp_polish(Xt, seconds=budget)
        vp = _max_cos(Xp)
        if vp < bestV:
            bestV = vp
            bestX = Xp.copy()

    ftime = LIMIT - (time.time() - t0) - 1.0
    if ftime > 1.0:
        bestX = _lp_polish(bestX, seconds=min(ftime, 8.0), radius_init=1.5e-3)
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

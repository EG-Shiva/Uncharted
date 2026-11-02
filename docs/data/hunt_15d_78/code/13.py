import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, radius_init=3e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    iu0 = iu[0].astype(np.int64)
    iu1 = iu[1].astype(np.int64)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu]
        m = float(c.max())
        thr = m - max(12.0 * radius, 2e-9)
        sel = c > thr
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints if too many (keep tightest)
        if p > 1500:
            idx = np.argpartition(-c[sel], 1500)[:1500]
            I = I[idx]; J = J[idx]
            csv = c[sel][idx]
            p = 1500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return X

def _contact_counts(X, tol=1.5e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    return C.sum(axis=1), m, G

def _largest_hole_vec(X_keep, rng, n_cand=3000):
    # find unit vector minimizing max dot to X_keep
    d = X_keep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_keep.T
    md = dots.max(axis=1)
    return cand[int(np.argmin(md))]

def _quench(X, rng, steps=220, beta=600.0, s0=0.02, s1=0.001):
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        step = s0 * (1 - frac) + s1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G[iu].max())
        # weights, stable
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        # normalize rows for direction stability
        rs = E.sum(axis=1, keepdims=True) + 1e-18
        Gr = (E @ X) / rs
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        nn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-18
        direc = -Gt / nn
        # damp overloaded vs free roughly equally
        X = X + step * direc
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # occasional beta jitter via noise is handled outside
    return X

def solve():
    t0 = time.time()
    LIMIT = 168.0
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

    it = 0
    # main loop, reserve ~10s for final polish
    while time.time() - t0 < LIMIT - 10.0:
        it += 1
        time_left = LIMIT - 10.0 - (time.time() - t0)
        if time_left < 2.5:
            break
        cnts, mcur, Gcur = _contact_counts(bestX)
        order_lo = np.argsort(cnts, kind='stable')
        order_hi = order_lo[::-1]
        mode = it % 6
        Xt = bestX.copy()
        if mode == 0 or mode == 1:
            k = 1 if mode == 0 else 2
            # kill k lowest-contact points, reinsert into holes
            victims = order_lo[:k].copy()
            # optionally also kill hub every few cycles
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = Xt[keep_mask]
            for vi in victims:
                hv = _largest_hole_vec(Xkeep, rng, n_cand=3000)
                # small jitter around hole
                hv = hv + 0.02 * rng.standard_normal(d)
                hv /= np.linalg.norm(hv)
                Xt[vi] = hv
                Xkeep = np.vstack([Xkeep, hv[None, :]])
            # light global jitter
            Xt = Xt + 0.015 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _quench(Xt, rng, steps=180, beta=500.0, s0=0.015, s1=0.001)
        elif mode == 2:
            # kill overloaded hub into hole
            hub = int(order_hi[0])
            # also a low point
            low = int(order_lo[0])
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[hub] = False
            Xkeep = Xt[keep_mask]
            hv = _largest_hole_vec(Xkeep, rng, n_cand=4000)
            Xt[hub] = hv
            # jitter low point strongly
            Xt[low] = Xt[low] + 0.35 * rng.standard_normal(d)
            Xt[low] /= np.linalg.norm(Xt[low])
            Xt = _quench(Xt, rng, steps=200, beta=700.0, s0=0.02, s1=0.001)
        elif mode == 3:
            # targeted shake of loose tail (lowest 10)
            pool = order_lo[:10]
            sig = float(rng.uniform(0.15, 0.35))
            for idx in pool:
                if rng.random() < 0.6:
                    Xt[idx] = Xt[idx] + sig * rng.standard_normal(d)
                    Xt[idx] /= np.linalg.norm(Xt[idx])
            Xt = _quench(Xt, rng, steps=200, beta=650.0, s0=0.018, s1=0.001)
        elif mode == 4:
            # global moderate shake
            sig = float(rng.uniform(0.03, 0.09))
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _quench(Xt, rng, steps=220, beta=550.0, s0=0.02, s1=0.001)
        else:
            # 3-point hole reinsert
            victims = order_lo[:3].copy()
            # randomize: sometimes pick random among lowest 14
            if rng.random() < 0.5:
                victims = rng.choice(order_lo[:14], size=3, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = Xt[keep_mask]
            for vi in victims:
                hv = _largest_hole_vec(Xkeep, rng, n_cand=2500)
                Xt[vi] = hv
                Xkeep = np.vstack([Xkeep, hv[None, :]])
            Xt = _quench(Xt, rng, steps=160, beta=450.0, s0=0.012, s1=0.0008)

        # quick check before expensive LP: if smoothed max already much worse, skip long polish
        vq = _max_cos(Xt)
        # polish budget adapts: more if promising
        if vq < bestV + 0.004:
            budget = min(2.2, max(0.9, time_left - 1.0))
            # promising -> slightly larger budget
            if vq < bestV + 0.0005:
                budget = min(3.0, max(1.2, time_left - 1.0))
        else:
            continue
        if time.time() - t0 > LIMIT - 10.0:
            break
        Xp = _lp_polish(Xt, seconds=budget)
        vp = _max_cos(Xp)
        if vp < bestV:
            bestV = vp
            bestX = Xp.copy()

    # final squeeze of best
    ftime = LIMIT - (time.time() - t0) - 1.0
    if ftime > 1.0:
        bestX = _lp_polish(bestX, seconds=min(ftime, 9.0), radius_init=2e-3)
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

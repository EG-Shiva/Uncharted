import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _anneal(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Mx = float(np.max(G))
        E = np.exp(alpha * (G - Mx))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        if not np.all(np.isfinite(gt)):
            continue
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # keep step tangential
        corr = np.sum(step * X, axis=1, keepdims=True)
        step = step - corr * X
        if not np.all(np.isfinite(step)):
            continue
        X = X - step
        X = _norm(X)
    return X

def _hole(Y, rng, n_cand=4000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)].copy()

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1100]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            best = vv
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline = t0 + LIMIT
    rng = np.random.default_rng(0)
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(55, 21)))
        best_score = _max_cos(best)
    x_best = best.copy()
    n, d = 55, 21
    # exploration budget: leave ~30s for final LP
    explore_end = t0 + 128.0
    trial = 0
    # precompute nothing; loop fixed trials but time-guarded
    while time.time() < explore_end and trial < 80:
        trial += 1
        prop = x_best.copy()
        mode = trial % 6
        if mode == 0:
            sigma = 0.015 + 0.015 * rng.random()
            prop = prop + sigma * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            prop = _anneal(prop, steps=700, a0=120.0, a1=800.0, lr0=0.015, lr1=0.002)
        elif mode == 1:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 1e-3, axis=1)
            victims = np.argsort(-cnt)[:1]
            if rng.random() < 0.25:
                victims = rng.choice(n, size=1, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=4000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.008 * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            prop = _anneal(prop, steps=900, a0=80.0, a1=700.0, lr0=0.015, lr1=0.002)
        elif mode == 2:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 2
            victims = np.argsort(-cnt)[:k]
            if rng.random() < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=4000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.008 * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            prop = _anneal(prop, steps=900, a0=80.0, a1=700.0, lr0=0.015, lr1=0.002)
        elif mode == 3:
            k = int(3 + rng.integers(0, 3))  # 3-5
            victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=3000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.012 * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            prop = _anneal(prop, steps=800, a0=60.0, a1=600.0, lr0=0.02, lr1=0.003)
        elif mode == 4:
            sigma = 0.05 + 0.04 * rng.random()
            prop = prop + sigma * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            prop = _anneal(prop, steps=1000, a0=40.0, a1=600.0, lr0=0.02, lr1=0.002)
        else:
            vi = rng.integers(0, n)
            keep = np.ones(n, dtype=bool); keep[vi] = False
            Y = prop[keep]
            h = _hole(Y, rng, n_cand=6000)
            prop[vi] = h
            prop = prop + 0.01 * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            prop = _anneal(prop, steps=800, a0=100.0, a1=800.0, lr0=0.012, lr1=0.002)
        s = _max_cos(prop)
        if s < best_score:
            best_score = s
            x_best = prop.copy()
        if time.time() >= explore_end:
            break
    # final LP snap with remaining time
    remain_deadline = min(deadline - 2.0, time.time() + 32.0)
    if time.time() < remain_deadline:
        try:
            xp = _lp_polish(x_best, remain_deadline)
            sp = _max_cos(xp)
            if sp < best_score:
                x_best = xp
                best_score = sp
        except Exception:
            pass
    x_best = _norm(x_best)
    # tiny shrink guard (no-op, ensures finite)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    x_best = _norm(x_best)
    return x_best

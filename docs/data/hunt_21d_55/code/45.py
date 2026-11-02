import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_quench(X0, steps, beta0, beta1, lr0, lr1, rng, noise=0.0):
    X = X0.copy()
    n, d = X.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            continue
        Gw = W @ X / S
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        Gt = Gw - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        if noise > 0:
            nz = rng.standard_normal((n, d))
            nz -= np.sum(nz * X, axis=1, keepdims=True) * X
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            Gt = (1.0 - noise) * Gt + noise * nz
            # re-project
            Gt -= np.sum(Gt * X, axis=1, keepdims=True) * X
            nn2 = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
            Gt /= nn2
        X = X - lr * Gt
        X = _norm(X)
    return X

def _hole(Y, rng, n_cand=5000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, deadline, radius_init=1e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = (x @ x.T)[iu]
        m = G.max()
        sel = G > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.where(sel)[0]
            order = np.argsort(-G[idx])[:1100]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = G[sel]
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
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_SEARCH_END = t0 + 135.0
    T_LP_END = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
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
        best = _norm(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    x_best = best.copy()

    trial = 0
    while time.time() < T_SEARCH_END:
        trial += 1
        mode = trial % 6
        prop = x_best.copy()
        if mode == 0:
            sigma = 0.008 + 0.008 * rng.random()
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1, nz = 400, 200.0, 800.0, 0.006, 0.001, 0.02
        elif mode == 1:
            sigma = 0.03 + 0.03 * rng.random()
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1, nz = 450, 150.0, 700.0, 0.008, 0.0015, 0.03
        elif mode == 2:
            sigma = 0.10 + 0.10 * rng.random()
            prop = _norm(prop + sigma * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1, nz = 500, 80.0, 600.0, 0.01, 0.002, 0.05
        elif mode == 3:
            # single hole reinsert of worst point
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            cnt = np.sum(G > mx - 1e-3, axis=1)
            v = int(np.argmax(cnt))
            if rng.random() < 0.25:
                v = int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool); mask[v] = False
            Y = prop[mask]
            prop[v] = _hole(Y, rng, n_cand=5000)
            prop = _norm(prop + 0.005 * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1, nz = 450, 200.0, 800.0, 0.006, 0.001, 0.01
        elif mode == 4:
            # double hole reinsert
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 2
            victims = np.argsort(-cnt)[:k].copy()
            if rng.random() < 0.3:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=4000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm(prop + 0.006 * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1, nz = 500, 150.0, 750.0, 0.007, 0.001, 0.02
        else:
            # triple hole reinsert
            k = 3
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            cnt = np.sum(G > mx - 1e-3, axis=1)
            victims = np.argsort(-cnt)[:k].copy()
            if rng.random() < 0.4:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=4000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm(prop + 0.008 * rng.standard_normal(prop.shape))
            steps, b0, b1, l0, l1, nz = 550, 120.0, 700.0, 0.008, 0.0015, 0.02

        prop = _smooth_quench(prop, steps=steps, beta0=b0, beta1=b1, lr0=l0, lr1=l1, rng=rng, noise=nz)
        s = _max_cos(prop)
        if s < best_score:
            best_score = s
            x_best = prop.copy()
        # time check for remaining budget
        if time.time() >= T_SEARCH_END:
            break

    # final trust-region LP polish on the best found
    try:
        x_best = _lp_polish(x_best, T_LP_END)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

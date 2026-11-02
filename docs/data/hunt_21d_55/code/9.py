import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_descent(X0, steps, beta0, beta1, lr0, lr1, rng, noise=0.0):
    X = X0.copy()
    n, d = X.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = np.max(G)
        # avoid overflow: exp(beta*(G-m)), G-m <=0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ X
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        Gt = Gw - dot * X
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nrm
        if noise > 0:
            nz = rng.standard_normal(X.shape)
            nz -= np.sum(nz * X, axis=1, keepdims=True) * X
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            Gt = (1.0 - noise) * Gt + noise * nz
            # re-project not needed, linear combo of tangents is tangent
        X = X - lr * Gt
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-16)
    return X

def _hole_for_others(Y, rng, n_cand=8000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-16)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def _lp_polish(X, t_end, radius_init=1e-3):
    X = _norm_rows(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        Gfull = X @ X.T
        cfull = Gfull[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        # cap size for speed
        if cnt > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    rng = np.random.default_rng(42)
    t0 = time.time()
    T_LIMIT = 178.0
    # reserve last ~28s for LP polish
    SEARCH_END = t0 + 148.0
    LP_END = t0 + 176.0

    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((55, 21))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (55, 21):
        tmp = rng.standard_normal((55, 21))
        tmp = _norm_rows(tmp)
        # try to copy what fits
        try:
            r = min(xb.shape[0], 55)
            c = min(xb.shape[1], 21)
            tmp[:r, :c] = xb[:r, :c]
            tmp = _norm_rows(tmp)
        except Exception:
            pass
        xb = tmp
    xb = _norm_rows(xb)
    best_score = _max_cos(xb)
    x_best = xb.copy()
    cur = xb.copy()
    cur_score = best_score

    n, d = 55, 21
    it = 0
    # temperature for basin hopping walk
    T = 0.0008

    while time.time() < SEARCH_END:
        it += 1
        remain = SEARCH_END - time.time()
        if remain < 1.0:
            break
        mode = it % 6
        prop = cur.copy()

        if mode == 0 or mode == 1:
            # hole reinsertion of overloaded points
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 1 if mode == 0 else 2
            victims = np.argsort(-cnt)[:k]
            if rng.random() < 0.35:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = prop[keep]
            # sequentially insert holes
            for vi in victims:
                h = _hole_for_others(Y, rng, n_cand=8000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.008 * rng.standard_normal(prop.shape)
            prop = _norm_rows(prop)
            steps = 900 + int(rng.integers(0, 500))
            b0, b1 = 60.0, 900.0
            lr0, lr1 = 0.008, 0.0015
            nz = 0.02
        elif mode == 2:
            sigma = 0.02 + 0.02 * rng.random()
            prop = prop + sigma * rng.standard_normal(prop.shape)
            prop = _norm_rows(prop)
            steps = 800 + int(rng.integers(0, 400))
            b0, b1 = 150.0, 700.0
            lr0, lr1 = 0.006, 0.001
            nz = 0.02
        elif mode == 3:
            sigma = 0.06 + 0.06 * rng.random()
            prop = prop + sigma * rng.standard_normal(prop.shape)
            prop = _norm_rows(prop)
            steps = 1000 + int(rng.integers(0, 600))
            b0, b1 = 80.0, 800.0
            lr0, lr1 = 0.008, 0.0015
            nz = 0.03
        elif mode == 4:
            # subset shake
            k = int(rng.integers(4, 12))
            idx = rng.choice(n, size=k, replace=False)
            prop[idx] += 0.25 * rng.standard_normal((k, d))
            prop = _norm_rows(prop)
            steps = 1000 + int(rng.integers(0, 500))
            b0, b1 = 60.0, 600.0
            lr0, lr1 = 0.008, 0.0015
            nz = 0.02
        else:
            # single point kick to hole + global small jitter
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 1e-3, axis=1)
            vi = int(np.argmax(cnt))
            if rng.random() < 0.3:
                vi = int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool); mask[vi] = False
            h = _hole_for_others(prop[mask], rng, n_cand=12000)
            prop[vi] = h
            prop += 0.015 * rng.standard_normal(prop.shape)
            prop = _norm_rows(prop)
            steps = 900 + int(rng.integers(0, 500))
            b0, b1 = 100.0, 900.0
            lr0, lr1 = 0.007, 0.001
            nz = 0.015

        # annealed smooth descent
        try:
            prop2 = _smooth_descent(prop, steps=steps, beta0=b0, beta1=b1, lr0=lr0, lr1=lr1, rng=rng, noise=nz)
        except Exception:
            continue
        try:
            s = _max_cos(prop2)
        except Exception:
            continue

        if s < best_score:
            best_score = s
            x_best = prop2.copy()
            cur = prop2.copy()
            cur_score = s
        else:
            # Metropolis move for cur to keep walking
            # decay temperature slowly
            Tf = T * max(0.2, 1.0 - (time.time() - t0) / 160.0)
            if s < cur_score or rng.random() < np.exp((cur_score - s) / max(Tf, 1e-6)):
                cur = prop2.copy()
                cur_score = s
            # occasional restart from best
            if it % 30 == 0:
                cur = x_best.copy()
                cur_score = best_score

        # early exit if time low
        if time.time() >= SEARCH_END:
            break

    # final LP polish of best (and cur if different and promising)
    try:
        x_best = _lp_polish(x_best, LP_END)
        best_score = _max_cos(x_best)
    except Exception:
        pass
    # if cur is close, try polishing it briefly if time remains
    if time.time() < LP_END - 5:
        try:
            if cur_score < best_score + 5e-4:
                xc = _lp_polish(cur, LP_END)
                sc = _max_cos(xc)
                if sc < best_score:
                    x_best = xc
                    best_score = sc
        except Exception:
            pass

    x_best = _norm_rows(x_best)
    return x_best

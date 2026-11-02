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

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel, dtype=bool)
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
    return x, best

def _smooth(x_init, steps, lr0, b0, b1):
    x = x_init
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 * (1.0 - 0.6 * f)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # avoid overflow: beta*(G-m) <= 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ x
        dot = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        x = x - lr * Gt
        # renorm
        nr2 = np.linalg.norm(x, axis=1, keepdims=True)
        nr2 = np.maximum(nr2, 1e-12)
        x = x / nr2
    return x

def _hole(Y, rng, n_cand=4000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    LIMIT = 172.0
    # load
    xb = None
    bs = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < bs:
                    bs = s
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.standard_normal((55, 21)))
        bs = _max_cos(xb)
    n, d = 55, 21
    x_best = xb.copy()
    best = bs

    it = 0
    # main basin hopping, reserve ~18s for final polish
    while time.time() - t0 < 150.0:
        it += 1
        remain = LIMIT - (time.time() - t0)
        if remain < 22:
            break
        mode = it % 6
        x_try = x_best.copy()
        if mode == 0:
            # small global shake
            sigma = 0.012 + 0.012 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try = _norm(x_try)
            x_try = _smooth(x_try, steps=350, lr0=0.006, b0=200.0, b1=700.0)
        elif mode == 1:
            # medium shake
            sigma = 0.03 + 0.03 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try = _norm(x_try)
            x_try = _smooth(x_try, steps=450, lr0=0.007, b0=150.0, b1=800.0)
        elif mode == 2 or mode == 3:
            # hub hole-reinsertion
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            mx = float(G.max())
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 1 if mode == 2 else 2
            if rng.random() < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            else:
                victims = np.argsort(-cnt)[:k]
                # sometimes perturb victim choice
                if rng.random() < 0.4:
                    victims = victims.copy()
                    victims[0] = rng.integers(0, n)
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = x_try[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=4000)
                x_try[vi] = h
                Y = np.vstack([Y, h[None, :]])
            x_try = x_try + 0.008 * rng.standard_normal(x_try.shape)
            x_try = _norm(x_try)
            x_try = _smooth(x_try, steps=500, lr0=0.007, b0=120.0, b1=800.0)
        elif mode == 4:
            # subset shake
            k = int(rng.integers(4, 11))
            idx = rng.choice(n, size=k, replace=False)
            x_try[idx] = x_try[idx] + 0.12 * rng.standard_normal((k, d))
            x_try = _norm(x_try)
            x_try = _smooth(x_try, steps=400, lr0=0.007, b0=150.0, b1=750.0)
        else:
            # larger global shake
            sigma = 0.06 + 0.04 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try = _norm(x_try)
            x_try = _smooth(x_try, steps=500, lr0=0.008, b0=100.0, b1=700.0)

        s_sm = _max_cos(x_try)
        # only LP-polish if promising to save time
        thresh = best + (0.002 if mode in (0, 2, 3) else 0.001)
        if s_sm > thresh:
            # not promising; keep hopping but allow rare acceptance
            continue
        # short LP polish with ~1.2s budget
        deadline = min(time.time() + 1.2, t0 + 150.0)
        x_p, s_p = _lp_polish_deadline(x_try, deadline)
        if s_p < best - 1e-12:
            x_best = x_p
            best = s_p
        # opportunistic early exit if we beat record clearly? continue to improve
    # final polish of best with remaining time
    remain = LIMIT - (time.time() - t0)
    if remain > 3:
        deadline = time.time() + min(remain - 1.0, 16.0)
        x_best, best = _lp_polish_deadline(x_best, deadline)
    x_best = _norm(np.asarray(x_best, dtype=np.float64))
    return x_best

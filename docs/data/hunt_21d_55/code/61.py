import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _maxcos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _quench(x0, steps, beta0, beta1, lr0, lr1, mu, rng, noise=0.0):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1
        lr = lr0 * (lr1 / lr0) ** f if steps > 1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-30
        Gw = (W @ x) / rs
        dot = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dot * x
        # row-normalize for stable step
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / gn
        if noise > 0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            Gt = (1.0 - noise) * Gt + noise * nz
        vel = mu * vel + (1.0 - mu) * Gt
        x = x - lr * vel
        x = _norm(x)
    return x

def _hole(Y, rng, n_cand=2000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    md = (cand @ Y.T).max(axis=1)
    return cand[np.argmin(md)]

def _lp_short(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _maxcos(x)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-13 and it < 40:
        it += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 2e-9)
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:900]
            ns = np.zeros_like(sel)
            ns[idx[order]] = True
            sel = ns
        csel = c[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
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
        v = _maxcos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LIMIT = 170.0
    SEARCH_END = t0 + 150.0
    rng = np.random.default_rng(20261101)
    n, d = 55, 21
    best = None
    best_s = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _maxcos(a)
                if s < best_s:
                    best_s = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_s = _maxcos(best)
    cur_base = best.copy()

    trial = 0
    # precompute contact counts of best to guide victim choice occasionally
    while time.time() < SEARCH_END:
        trial += 1
        remain = LIMIT - (time.time() - t0)
        if remain < 12:
            break
        mode = trial % 7
        x_try = cur_base.copy()
        if mode == 0:
            # fresh random restart, cheap quench
            x_try = _norm(rng.standard_normal((n, d)))
            x_try = _quench(x_try, steps=260, beta0=25.0, beta1=500.0,
                            lr0=0.035, lr1=0.004, mu=0.7, rng=rng, noise=0.02)
        elif mode == 1:
            # mass reinsert: replace 6-12 random points by uniform random
            k = int(6 + rng.integers(0, 7))
            victims = rng.choice(n, size=k, replace=False)
            rnd = _norm(rng.standard_normal((k, d)))
            x_try[victims] = rnd
            # small jitter rest
            x_try = _norm(x_try + 0.005 * rng.standard_normal((n, d)))
            x_try = _quench(x_try, steps=240, beta0=30.0, beta1=600.0,
                            lr0=0.030, lr1=0.003, mu=0.75, rng=rng, noise=0.01)
        elif mode == 2:
            # worst-point hole reinsert (cheap 2000 cands)
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = int(2 + rng.integers(0, 3))  # 2-4
            victims = order[:k]
            # also add 2 random others to diversify
            extra = rng.choice(n, size=2, replace=False)
            victims = np.unique(np.concatenate([victims, extra]))[:k+2]
            for w in victims:
                mask = np.ones(n, dtype=bool); mask[w] = False
                h = _hole(x_try[mask], rng, n_cand=2000)
                h = h + 0.005 * rng.standard_normal(d)
                h /= np.linalg.norm(h) + 1e-12
                x_try[w] = h
            x_try = _quench(x_try, steps=240, beta0=35.0, beta1=650.0,
                            lr0=0.028, lr1=0.003, mu=0.75, rng=rng, noise=0.01)
        elif mode == 3:
            # large global shake
            sigma = 0.25 + 0.25 * rng.random()
            x_try = _norm(x_try + sigma * rng.standard_normal((n, d)))
            x_try = _quench(x_try, steps=260, beta0=20.0, beta1=550.0,
                            lr0=0.032, lr1=0.004, mu=0.7, rng=rng, noise=0.03)
        elif mode == 4:
            # subset shake 30-50%
            frac = 0.3 + 0.25 * rng.random()
            k = int(n * frac)
            sub = rng.choice(n, size=k, replace=False)
            sigma = 0.25 + 0.2 * rng.random()
            x_try[sub] += sigma * rng.standard_normal((k, d))
            x_try = _norm(x_try)
            x_try = _quench(x_try, steps=220, beta0=28.0, beta1=600.0,
                            lr0=0.030, lr1=0.003, mu=0.75, rng=rng, noise=0.02)
        elif mode == 5:
            # medium shake of best + short quench (exploit)
            sigma = 0.04 + 0.06 * rng.random()
            x_try = _norm(x_try + sigma * rng.standard_normal((n, d)))
            x_try = _quench(x_try, steps=200, beta0=60.0, beta1=800.0,
                            lr0=0.022, lr1=0.003, mu=0.8, rng=rng, noise=0.0)
        else:
            # rotate/shake: random orthogonal-ish directional push
            sigma = 0.10 + 0.10 * rng.random()
            x_try = _norm(x_try + sigma * rng.standard_normal((n, d)))
            x_try = _quench(x_try, steps=220, beta0=40.0, beta1=700.0,
                            lr0=0.025, lr1=0.003, mu=0.75, rng=rng, noise=0.015)
        v = _maxcos(x_try)
        if v < best_s - 1e-12:
            best_s = v
            best = x_try.copy()
            cur_base = x_try.copy()
        # occasionally reset base to best, occasionally keep exploring from new region
        if trial % 25 == 0:
            cur_base = best.copy()
        elif v < best_s + 0.004 and rng.random() < 0.15:
            # allow walk to nearby basin to force topology drift
            cur_base = x_try.copy()

    # final: quick gradient high-beta polish of best, then gated short LP
    try:
        if time.time() - t0 < LIMIT - 12:
            best = _quench(best, steps=180, beta0=150.0, beta1=1200.0,
                           lr0=0.012, lr1=0.002, mu=0.8, rng=rng, noise=0.0)
            best_s = _maxcos(best)
    except Exception:
        pass
    try:
        remain = LIMIT - (time.time() - t0)
        if remain > 6.0:
            dl = time.time() + min(remain - 1.0, 9.0)
            cand = _lp_short(best, dl)
            vs = _maxcos(cand)
            if vs < best_s:
                best = cand
                best_s = vs
    except Exception:
        pass
    best = _norm(np.asarray(best, dtype=np.float64))
    return best

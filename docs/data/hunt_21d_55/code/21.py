import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _nr(x):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-15)
    return x

def _quench(x0, steps, lr0, lr1, b0, b1, lam, rng, noise=0.0, mom=0.85):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # soft-max weights, overflow safe
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum() + 1e-30
        Gw = (W @ x) / S
        if lam > 0:
            Gp = np.where(G > 0, G, 0.0)
            np.fill_diagonal(Gp, 0.0)
            Gw = Gw + lam * (Gp @ x) / n
        dot = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dot * x
        nn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nn
        if noise > 0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nz /= (np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12)
            Gt = (1.0 - noise) * Gt + noise * nz
            # renormalize direction
            Gt /= (np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12)
        vel = mom * vel + (1.0 - mom) * Gt
        x = x - lr * vel
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-15)
    return x

def _hole(Y, rng, n_cand=5000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-15)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, deadline, radius_init=1e-3, cap=700):
    x = _nr(x)
    n, d = x.shape
    best = _mc(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    niter = 0
    while time.time() < deadline and radius > 1e-13 and niter < 40:
        niter += 1
        G = x @ x.T
        cfull = G[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:cap]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y = _nr(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(11)
    t0 = time.time()
    deadline = t0 + 171.0
    search_end = t0 + 150.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((55, 21))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (55, 21):
        xb = rng.standard_normal((55, 21))
    xb = _nr(xb)
    best = _mc(xb)
    x_best = xb.copy()
    n, d = 55, 21
    it = 0
    while time.time() < search_end:
        it += 1
        if deadline - time.time() < 9:
            break
        mode = it % 6
        x_try = x_best.copy()
        if mode == 0:
            sigma = 0.008 + 0.012 * rng.random()
            x_try = _nr(x_try + sigma * rng.standard_normal(x_try.shape))
            x_try = _quench(x_try, steps=280, lr0=0.006, lr1=0.002,
                            b0=200.0, b1=700.0, lam=0.004, rng=rng, noise=0.02)
        elif mode == 1:
            sigma = 0.04 + 0.06 * rng.random()
            x_try = _nr(x_try + sigma * rng.standard_normal(x_try.shape))
            x_try = _quench(x_try, steps=380, lr0=0.009, lr1=0.003,
                            b0=120.0, b1=900.0, lam=0.006, rng=rng, noise=0.05)
        elif mode == 2:
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = np.sum(G > m - 2e-4, axis=1)
            order = np.argsort(-cnt - 0.5 * G.max(axis=1))
            k = 3 + int(rng.integers(0, 4))
            worst = order[:k]
            sig = np.full(n, 0.004)
            sig[worst] = 0.12 + 0.10 * rng.random()
            x_try = _nr(x_try + sig[:, None] * rng.standard_normal(x_try.shape))
            x_try = _quench(x_try, steps=360, lr0=0.008, lr1=0.0025,
                            b0=250.0, b1=850.0, lam=0.003, rng=rng, noise=0.03)
        elif mode == 3:
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = np.sum(G > m - 2e-4, axis=1)
            order = np.argsort(-cnt)
            nr = 1 + int(rng.integers(0, 2))
            for kk in range(nr):
                w = int(order[kk]) if rng.random() < 0.85 else int(rng.integers(0, n))
                mask = np.ones(n, dtype=bool)
                mask[w] = False
                h = _hole(x_try[mask], rng, n_cand=5000)
                h = h + 0.008 * rng.standard_normal(d)
                h /= (np.linalg.norm(h) + 1e-15)
                x_try[w] = h
            x_try = _quench(x_try, steps=360, lr0=0.007, lr1=0.002,
                            b0=180.0, b1=800.0, lam=0.005, rng=rng, noise=0.02)
        elif mode == 4:
            frac = 0.20 + 0.30 * rng.random()
            k = int(n * frac)
            subset = rng.choice(n, size=k, replace=False)
            sigma = 0.05 + 0.09 * rng.random()
            x_try[subset] += sigma * rng.standard_normal((k, d))
            x_try = _nr(x_try)
            x_try = _quench(x_try, steps=380, lr0=0.008, lr1=0.0025,
                            b0=150.0, b1=900.0, lam=0.004, rng=rng, noise=0.04)
        else:
            # strong global shake, new basin attempt
            sigma = 0.18 + 0.15 * rng.random()
            x_try = _nr(x_try + sigma * rng.standard_normal(x_try.shape))
            x_try = _quench(x_try, steps=450, lr0=0.012, lr1=0.003,
                            b0=80.0, b1=700.0, lam=0.008, rng=rng, noise=0.06)
        v = _mc(x_try)
        if v < best - 1e-12:
            x_best, best = x_try, v
        # occasional restart to best is implicit (x_try starts from best)
    rem = deadline - time.time()
    if rem > 4.0:
        final_end = min(deadline - 0.8, time.time() + 19.0)
        x_best = _lp_polish(x_best, deadline=final_end, radius_init=1e-3)
    x_best = _nr(x_best)
    return x_best

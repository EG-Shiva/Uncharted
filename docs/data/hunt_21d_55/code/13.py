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

def _smooth(x0, steps, lr0, b0, b1, rng, noise=0.0):
    x = x0.copy()
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 * (1.0 - 0.6 * f)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # avoid overflow: beta*(G-m) <= 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ x
        dot = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dot * x
        nn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt /= nn
        if noise > 0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nz /= (np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12)
            Gt = (1.0 - noise) * Gt + noise * nz
        x = x - lr * Gt
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-15)
    return x

def _hole(Y, rng, n_cand=8000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-15)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, deadline, radius_init=1e-3, cap=800):
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
        if time.time() >= deadline:
            break
    return x

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    LIMIT = 172.0
    deadline = t0 + LIMIT
    search_end = t0 + 148.0
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
        xb = tmp
    xb = _nr(xb)
    best = _mc(xb)
    x_best = xb.copy()
    n, d = 55, 21
    it = 0
    # fast loop, no LP unless promising
    while time.time() < search_end:
        it += 1
        remain = deadline - time.time()
        if remain < 8:
            break
        mode = it % 5
        x_try = x_best.copy()
        if mode == 1:
            sigma = 0.015 + 0.015 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try = _nr(x_try)
            x_try = _smooth(x_try, steps=350, lr0=0.006, b0=200.0, b1=700.0, rng=rng, noise=0.02)
        elif mode == 2:
            sigma = 0.05 + 0.06 * rng.random()
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try = _nr(x_try)
            x_try = _smooth(x_try, steps=450, lr0=0.008, b0=150.0, b1=900.0, rng=rng, noise=0.05)
        elif mode == 3:
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = 4 + int(rng.integers(0, 4))
            worst = order[:k]
            sig = np.full(n, 0.004)
            sig[worst] = 0.15 + 0.12 * rng.random()
            neigh = np.argsort(-G[worst[0]])[:8]
            for j in neigh:
                if sig[j] < 0.05:
                    sig[j] = 0.05
            x_try = x_try + sig[:, None] * rng.standard_normal(x_try.shape)
            x_try = _nr(x_try)
            x_try = _smooth(x_try, steps=400, lr0=0.007, b0=250.0, b1=800.0, rng=rng, noise=0.03)
        elif mode == 4:
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            nr = 1 + int(rng.integers(0, 2))
            for kk in range(nr):
                w = int(order[kk])
                mask = np.ones(n, dtype=bool)
                mask[w] = False
                hole = _hole(x_try[mask], rng, n_cand=8000)
                hole = hole + 0.01 * rng.standard_normal(d)
                hole /= (np.linalg.norm(hole) + 1e-15)
                x_try[w] = hole
            x_try = _smooth(x_try, steps=350, lr0=0.006, b0=200.0, b1=800.0, rng=rng, noise=0.02)
        else:
            frac = 0.25 + 0.3 * rng.random()
            k = int(n * frac)
            subset = rng.choice(n, size=k, replace=False)
            sigma = 0.06 + 0.08 * rng.random()
            x_try[subset] += sigma * rng.standard_normal((k, d))
            x_try = _nr(x_try)
            x_try = _smooth(x_try, steps=400, lr0=0.007, b0=180.0, b1=850.0, rng=rng, noise=0.04)
        v_smooth = _mc(x_try)
        if v_smooth > best + 0.012:
            continue
        # only LP-polish if close to best
        if v_smooth < best + 0.002 and time.time() < search_end - 1.0:
            # give short budget
            lp_dead = min(time.time() + 1.0, search_end, deadline - 16.0)
            if lp_dead - time.time() > 0.4:
                x_pol = _lp_polish(x_try, deadline=lp_dead, radius_init=8e-4 if v_smooth > best else 1.2e-3)
                v_pol = _mc(x_pol)
                if v_pol < best - 1e-12:
                    x_best, best = x_pol, v_pol
                    continue
                elif v_smooth < best - 1e-12 and v_smooth < v_pol:
                    x_best, best = x_try, v_smooth
                    continue
                else:
                    continue
        if v_smooth < best - 1e-12:
            # accept smooth improvement without LP (LP will finish later)
            x_best, best = x_try, v_smooth
    # final polish of best with remaining time
    rem = deadline - time.time()
    if rem > 3.0:
        budget_end = deadline - 0.8
        # cap final polish to 20s to guarantee return
        final_end = min(budget_end, time.time() + 20.0)
        x_best = _lp_polish(x_best, deadline=final_end, radius_init=1e-3)
    x_best = _nr(x_best)
    return x_best

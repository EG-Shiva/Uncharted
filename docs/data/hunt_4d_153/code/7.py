import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thresh=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    tight = g > m - thresh
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return m, cnt, iu, g

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _smooth(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    if steps <= 1:
        lrb = 0.0; betb = 0.0
    else:
        lrb = np.log(lr1 / lr0) / (steps - 1)
        betb = np.log(beta1 / beta0) / (steps - 1)
    Iu0, Ju0 = iu
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _hole(rng, x_keep, n_samp=20000):
    # largest hole w.r.t. x_keep: minimise max dot
    c = rng.normal(size=(n_samp, 4))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    s = c @ x_keep.T
    m = s.max(axis=1)
    i = int(np.argmin(m))
    return c[i]

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                c = np.asarray(c, dtype=float).reshape(153, 4)
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append((c, _max_cos(c)))
            except Exception:
                pass
    if len(cands) == 0:
        xb0 = rng.normal(size=(153, 4))
        xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
        cands.append((xb0, _max_cos(xb0)))
    cands.sort(key=lambda t: t[1])
    x_best, best_val = cands[0][0].copy(), float(cands[0][1])
    n, d = 153, 4
    iu = np.triu_indices(n, 1)

    # short snap of incumbent
    if time.time() - t0 < LIMIT - 20:
        try:
            x_best, best_val = _lp_polish(x_best, 3.0)
        except Exception:
            pass

    it = 0
    while time.time() - t0 < LIMIT - 8:
        it += 1
        remain = LIMIT - (time.time() - t0)
        if remain < 10:
            break
        # analyse contacts of incumbent
        try:
            m, cnt, _, _ = _contacts(x_best, thresh=1e-7)
        except Exception:
            cnt = np.full(n, 5)
        ratt = np.where(cnt <= 1)[0]
        low = np.where(cnt <= 3)[0]
        hubs = np.where(cnt >= 7)[0]
        order = np.argsort(cnt)  # ascending
        mode = it % 7
        y = x_best.copy()
        if mode == 0:
            sigma = 0.012
            y += rng.normal(0, sigma, size=y.shape)
        elif mode == 1:
            sigma = 0.035
            y += rng.normal(0, sigma, size=y.shape)
            # reinsert up to 2 rattlers/low into holes
            k = min(2, len(low))
            if k > 0:
                victims = rng.choice(low, size=k, replace=False)
                keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
                xk = y[keep_mask]
                for v in victims:
                    y[v] = _hole(rng, xk, n_samp=15000)
                    y[v] += rng.normal(0, 0.01, size=4)
                    xk = np.vstack([xk, y[v] / np.linalg.norm(y[v])])
        elif mode == 2:
            sigma = 0.08
            y += rng.normal(0, sigma, size=y.shape)
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, 0.05, size=(len(hubs), d))
            # one random hole reinsert
            v = int(rng.integers(n))
            mask = np.ones(n, dtype=bool); mask[v] = False
            y[v] = _hole(rng, y[mask], n_samp=15000)
        elif mode == 3:
            # 3 random points to holes
            victims = rng.choice(n, size=3, replace=False)
            y += rng.normal(0, 0.02, size=y.shape)
            mask = np.ones(n, dtype=bool); mask[victims] = False
            xk = y[mask]
            for v in victims:
                y[v] = _hole(rng, xk, n_samp=15000)
                xk = np.vstack([xk, y[v] / np.linalg.norm(y[v])])
        elif mode == 4:
            sigma = 0.16
            y += rng.normal(0, sigma, size=y.shape)
        elif mode == 5:
            # targeted: worst 5 hubs get big kick, rattlers to holes
            y += rng.normal(0, 0.02, size=y.shape)
            if len(hubs) > 0:
                sel = hubs[rng.choice(len(hubs), size=min(4, len(hubs)), replace=False)]
                for v in sel:
                    if rng.random() < 0.5:
                        mask = np.ones(n, dtype=bool); mask[v] = False
                        y[v] = _hole(rng, y[mask], n_samp=15000)
                    else:
                        y[v] += rng.normal(0, 0.25, size=4)
            if len(ratt) > 0:
                for v in ratt:
                    mask = np.ones(n, dtype=bool); mask[v] = False
                    y[v] = _hole(rng, y[mask], n_samp=15000)
        else:
            sigma = 0.05
            y += rng.normal(0, sigma, size=y.shape)
            v = int(rng.integers(n))
            y[v] = rng.normal(size=4)

        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed softmax repulsion
        steps = 500
        # vary schedule a bit by mode
        if mode in (4,):
            beta0, beta1 = 60.0, 800.0
            lr0, lr1 = 0.03, 0.002
        elif mode in (2, 3):
            beta0, beta1 = 120.0, 2000.0
            lr0, lr1 = 0.02, 0.0015
        else:
            beta0, beta1 = 80.0, 1500.0
            lr0, lr1 = 0.025, 0.002
        try:
            y2 = _smooth(y, steps, beta0, beta1, lr0, lr1, iu)
        except Exception:
            continue
        v0 = _max_cos(y2)
        # only LP-polish promising candidates to save time
        if v0 > best_val + 0.015:
            continue
        # quick extra hill refine with higher beta
        try:
            y2 = _smooth(y2, 200, 800.0, 3000.0, 0.008, 0.0008, iu)
            v0 = _max_cos(y2)
        except Exception:
            pass
        if v0 > best_val + 0.008:
            continue
        lp_budget = 2.2 if v0 < best_val + 0.002 else 1.2
        if remain < 15:
            lp_budget = min(lp_budget, 1.0)
        try:
            y3, v1 = _lp_polish(y2, lp_budget)
        except Exception:
            continue
        if v1 < best_val - 1e-12:
            x_best, best_val = y3, float(v1)
            # extra polish when improved
            try:
                if time.time() - t0 < LIMIT - 12:
                    x_best, best_val = _lp_polish(x_best, 1.5)
            except Exception:
                pass
        # timeout check
        if time.time() - t0 >= LIMIT - 5:
            break

    # final polish with remaining time
    try:
        rem = LIMIT - (time.time() - t0)
        if rem > 2:
            x_best, best_val = _lp_polish(x_best, min(rem - 1, 6.0))
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        # upper tri values
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        # exp weights shifted by max
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # normalize scale a bit to keep lr stable
        # (optional) clip large grads
        v = mu * v + gt
        if mu > 0:
            vc = v / (1.0 - mu ** (t + 1))
        else:
            vc = v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x = x / nrm
    return x

def _void_reinsert(x, idx, rng, ncand=3000):
    n, d = x.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[np.asarray(idx)] = False
    Xkeep = x[keep_mask]
    # sequentially insert
    for _ in np.asarray(idx):
        cand = rng.standard_normal((ncand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xkeep.T
        mxs = dots.max(axis=1)
        bi = int(np.argmin(mxs))
        chosen = cand[bi:bi+1]
        Xkeep = np.concatenate([Xkeep, chosen], axis=0)
    # rebuild full array: keep + new
    # place new points at idx positions
    y = x.copy()
    new_pts = Xkeep[-len(idx):]
    y[np.asarray(idx)] = new_pts
    return y

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    SEARCH_END = t0 + 122.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((n, d))
    best = np.asarray(best, dtype=float)
    if best.shape != (n, d):
        best = rng.standard_normal((n, d))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    # candidate pool for diversity
    pool = [(inc_val, incumbent.copy())]

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        # choose base
        rb = rng.random()
        if rb < 0.60:
            base = incumbent
        elif rb < 0.85:
            base = best
        else:
            base = pool[int(rng.integers(len(pool)))][1]
        x = base.copy()
        mode = rng.random()
        # hyperparams for smooth
        k0 = float(rng.choice(np.array([8.0, 15.0, 30.0, 50.0])))
        k1 = float(rng.choice(np.array([150.0, 300.0, 600.0, 1000.0])))
        lr0 = float(rng.choice(np.array([0.03, 0.06, 0.12])))
        lr1 = float(rng.choice(np.array([0.003, 0.008, 0.015])))
        mu = float(rng.choice(np.array([0.0, 0.5, 0.8, 0.9])))
        noise = float(rng.choice(np.array([0.0, 0.005, 0.015])))
        iters = int(rng.choice(np.array([800, 1500, 2500])))

        if mode < 0.35:
            sigma = float(rng.choice(np.array([0.02, 0.05, 0.09, 0.15, 0.25])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        elif mode < 0.75:
            # surgery: remove worst, reinsert voids
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            # contact count near max
            m = maxper.max()
            thr = m - 1e-7
            cnt = (G > thr).sum(axis=1)
            # score = maxper + 0.005*cnt to prefer overloaded
            score = maxper + 0.003 * cnt + 0.001 * rng.standard_normal(n)
            k = int(rng.integers(2, 9))
            idx = np.argsort(-score)[:k]
            ncand = int(rng.choice(np.array([2000, 4000, 8000])))
            x = _void_reinsert(x, idx, rng, ncand=ncand)
            # small jitter after insert
            x = x + 0.01 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        elif mode < 0.90:
            # patch shake around random center
            c0 = int(rng.integers(n))
            diff = x - x[c0]
            # angular distance via dots
            dots = x @ x[c0]
            nn = int(rng.choice(np.array([8, 12, 20])))
            patch = np.argsort(-dots)[:nn]
            sigma = float(rng.choice(np.array([0.08, 0.15, 0.30])))
            x[patch] = x[patch] + sigma * rng.standard_normal((nn, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # smooth with a bit larger lr
            x = _smooth(x, iters, k0, k1, lr0 * 1.5, lr1, mu, noise, rng)
        else:
            # large shake
            sigma = float(rng.choice(np.array([0.30, 0.45])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, iters, k0, k1, lr0, lr1, mu, 0.02, rng)

        vs = _max_cos(x)
        # keep pool diverse (top 5)
        if len(pool) < 6 or vs < max(p[0] for p in pool) + 1e-9:
            pool.append((vs, x.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:6]

        # decide to LP polish: if promising
        if vs < inc_val + 0.004:
            remain = SEARCH_END - time.time()
            if remain < 1.0:
                break
            budget = 0.8 if vs > inc_val - 1e-6 else 2.0
            budget = min(budget, remain - 0.1)
            if budget > 0.25:
                y = _lp_polish(x, budget)
                vy = _max_cos(y)
                if vy < inc_val:
                    incumbent = y.copy()
                    inc_val = vy
                    if vy < best_val:
                        best = y.copy()
                        best_val = vy
                # also keep y in pool
                pool.append((vy, y.copy()))
                pool = sorted(pool, key=lambda z: z[0])[:6]
        # early exit if time
        if time.time() > SEARCH_END - 0.5:
            break

    # final polish of best with remaining time
    rem = DEADLINE - time.time()
    if rem > 3.0:
        # polish incumbent (which equals best if improved, else best)
        target = best if best_val <= inc_val else incumbent
        tval = min(best_val, inc_val)
        polished = _lp_polish(target, min(rem - 1.0, 40.0))
        pv = _max_cos(polished)
        if pv < tval:
            target = polished
            tval = pv
        target = target / np.linalg.norm(target, axis=1, keepdims=True)
        return target
    else:
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        return best

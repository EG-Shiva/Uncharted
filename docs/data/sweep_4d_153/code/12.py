import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def farthest_candidates(Xkeep, nc, rng, n_draw=2000):
    # draw many, return best void (min max-dot)
    cand = rng.standard_normal((n_draw, Xkeep.shape[1]))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    m = dots.max(axis=1)
    idx = np.argsort(m)[:nc]
    return cand[idx]

def solve():
    t0 = time.time()
    BUDGET = 172.0
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
    best_val = max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    cands = [(inc_val, incumbent.copy())]

    def time_left():
        return t0 + BUDGET - time.time()

    # quick initial polish of loaded best to ensure baseline (short)
    # skip long polish, parent already polished

    trial = 0
    SEARCH_END = t0 + 118.0
    while time.time() < SEARCH_END:
        trial += 1
        # choose base
        rb = rng.random()
        if rb < 0.60:
            base = incumbent
        elif rb < 0.85:
            base = best
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        x = base.copy()
        mode = rng.random()
        if mode < 0.14:
            # pure random restart
            x = rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.50:
            sigma = float(rng.choice(np.array([0.015, 0.03, 0.06, 0.10, 0.16, 0.25])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.78:
            # remove worst / overloaded
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            # contact degree near max
            m = G.max()
            deg = (G > m - 2e-3).sum(axis=1)
            score = maxper + 0.002 * deg + 0.0005 * rng.standard_normal(n)
            k = int(rng.integers(2, 11))
            idx = np.argsort(-score)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            # sequential void insertion
            new_pts = []
            cur = Xkeep
            for _ in range(k):
                pool = farthest_candidates(cur, 1, rng, n_draw=2500)
                pt = pool[0]
                # small jitter
                pt = pt + 0.005 * rng.standard_normal(d)
                pt /= np.linalg.norm(pt)
                new_pts.append(pt)
                cur = np.vstack([cur, pt.reshape(1, d)])
            x[keep_mask] = Xkeep
            # place new points at idx positions
            for ii, jj in enumerate(idx):
                x[jj] = new_pts[ii]
        else:
            # spatial patch removal
            center = int(rng.integers(n))
            msz = int(rng.integers(6, 15))
            dots = x @ x[center]
            order = np.argsort(-dots)
            idx = order[:msz]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            cur = Xkeep.copy()
            new_pts = []
            for _ in range(msz):
                pool = farthest_candidates(cur, 1, rng, n_draw=2000)
                pt = pool[0]
                new_pts.append(pt)
                cur = np.vstack([cur, pt.reshape(1, d)])
            for ii, jj in enumerate(idx):
                x[jj] = new_pts[ii]
            # extra jitter on patch
            x[idx] = x[idx] + 0.02 * rng.standard_normal((msz, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # random anneal hyperparams
        iters = int(rng.integers(700, 2200))
        k0 = float(rng.uniform(8, 30))
        k1 = float(rng.choice(np.array([120.0, 250.0, 450.0, 800.0])))
        if k1 < k0:
            k1 = k0 * 5
        lr0 = float(rng.uniform(0.03, 0.12))
        lr1 = float(rng.uniform(0.002, 0.015))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.88, 0.93])))
        noise = float(rng.choice(np.array([0.0, 0.0, 0.003, 0.008])))
        # shorten iters if low time
        if time.time() > SEARCH_END - 10:
            iters = min(iters, 700)
        x = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        v = max_cos(x)
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = x.copy()
            cands.append((v, x.copy()))
            if len(cands) > 8:
                cands = sorted(cands, key=lambda t: t[0])[:8]
        elif v < inc_val + 0.008 and rng.random() < 0.15:
            # keep diverse near-best
            cands.append((v, x.copy()))
            if len(cands) > 8:
                cands = sorted(cands, key=lambda t: t[0])[:8]

        # LP polish promising ones
        if v < inc_val + 0.003 and time.time() < SEARCH_END:
            # only polish if clearly promising or new best
            do_polish = (v < inc_val + 1e-9) or (rng.random() < 0.08)
            if do_polish:
                tl = time.time()
                # give 1-2 sec
                xp = lp_polish(x, min(2.0, SEARCH_END - time.time() - 0.2))
                vp = max_cos(xp)
                if vp < inc_val:
                    inc_val = vp
                    incumbent = xp.copy()
                    cands.append((vp, xp.copy()))
                    cands = sorted(cands, key=lambda t: t[0])[:8]
                elif vp < cands[-1][0]:
                    cands.append((vp, xp.copy()))
                    cands = sorted(cands, key=lambda t: t[0])[:8]

    # final intensification on top candidates
    cands = sorted(cands, key=lambda t: t[0])[:4]
    # also include best
    for _, cx in list(cands):
        if time_left() < 6:
            break
        xp = lp_polish(cx, min(8.0, time_left() - 1.0))
        vp = max_cos(xp)
        if vp < inc_val:
            inc_val = vp
            incumbent = xp.copy()
    # extra final polish of incumbent
    if time_left() > 4:
        incumbent = lp_polish(incumbent, time_left() - 1.0)
        inc_val = max_cos(incumbent)
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    # safety: replace NaN
    if not np.all(np.isfinite(incumbent)):
        incumbent = best / np.linalg.norm(best, axis=1, keepdims=True)
    return incumbent

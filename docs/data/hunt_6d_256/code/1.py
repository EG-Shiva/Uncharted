import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal_track(x0, steps, a0, a1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = _max_cos(x)
    best_x = x.copy()
    # geometric schedule
    la0, la1 = np.log(a0), np.log(a1)
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        alpha = np.exp(la0 + (la1 - la0) * t / steps)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # weights, avoid overflow (G-gmax<=0)
        W = np.exp((G - gmax) * alpha)
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (t % 200 == 0) or (t == steps):
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            val = float(G2.max())
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    TOTAL = 168.0
    ANNEAL_END = t0 + 128.0
    LP_END = t0 + TOTAL - 2.0
    rng = _np.random.default_rng(0)
    # load start
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            x_best = _np.load(fn)
            break
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
    x_best = _np.asarray(x_best, dtype=float)
    if x_best.shape != (256, 6):
        # fallback reshape/truncate
        tmp = rng.normal(size=(256, 6))
        r = min(x_best.shape[0], 256)
        tmp[:r] = x_best[:r, :6] if x_best.shape[1] >= 6 else x_best[:r]
        x_best = tmp
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    best = _max_cos(x_best)
    n, d = (256, 6)
    cycle = 0
    # basin hopping
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 2:
            break
        x = x_best.copy()
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        cnt = _np.sum(G > gmax - 1e-9, axis=1)
        ratt = _np.where(cnt == 0)[0]
        order = _np.argsort(-cnt)
        if cycle % 5 == 0:
            sigma = 0.25
        elif cycle % 2 == 0:
            sigma = 0.08
        else:
            sigma = 0.03
        # re-insert rattlers into largest voids
        if len(ratt) > 0:
            for r in ratt:
                # sample candidates
                Z = rng.normal(size=(12000, 6))
                Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
                S = Z @ x.T
                S[:, r] = -2.0
                m2 = S.max(axis=1)
                idx = int(_np.argmin(m2))
                x[r] = Z[idx]
        # perturb overloaded hubs
        k_over = 12
        over = order[:k_over]
        x[over] += sigma * rng.normal(size=(k_over, 6))
        rnd = rng.choice(n, size=8, replace=False)
        x[rnd] += (sigma * 0.7) * rng.normal(size=(8, 6))
        if cycle % 3 == 0:
            c0 = int(rng.integers(n))
            dists = x @ x[c0]
            nearest = _np.argsort(-dists)[:10]
            x[nearest] += (sigma * 0.8) * rng.normal(size=(10, 6))
        x /= _np.linalg.norm(x, axis=1, keepdims=True)
        if cycle % 5 == 0:
            steps, a0, a1, lr = 1200, 30.0, 800.0, 0.015
        elif cycle % 2 == 0:
            steps, a0, a1, lr = 800, 60.0, 1200.0, 0.01
        else:
            steps, a0, a1, lr = 600, 80.0, 1500.0, 0.008
        if rem < 25:
            steps = min(steps, 400)
        xb, vb = _anneal_track(x, steps, a0, a1, lr)
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb
            best = vb
        # opportunistic small random-accept to diversify if stuck: every 10 cycles try from second perturbation
        if cycle % 10 == 0 and _time.time() < ANNEAL_END - 5:
            # extra large kick from best
            x2 = x_best.copy()
            pick = rng.choice(n, size=20, replace=False)
            x2[pick] += 0.35 * rng.normal(size=(20, 6))
            x2 /= _np.linalg.norm(x2, axis=1, keepdims=True)
            xb2, vb2 = _anneal_track(x2, 500, 50.0, 1000.0, 0.012)
            if vb2 < best - 1e-12:
                x_best = xb2
                best = vb2
    # final polish
    if _time.time() < LP_END:
        try:
            x_best = _lp_polish_deadline(x_best, LP_END)
        except Exception:
            pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard not needed; ensure finite
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best

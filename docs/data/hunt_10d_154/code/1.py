import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=0.02, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_full = G[iu0]
        m = c_full.max()
        sel = c_full > m - max(20.0 * radius, 1e-9)
        # cap number of constraints for speed
        if np.count_nonzero(sel) > 2500:
            # keep tightest 2500
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:2500]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_full[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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

def _soft_push(x, steps, k, lr):
    x = x.copy()
    n = x.shape[0]
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # weights focused on near-max pairs
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        # tangent projection
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad_t = grad - dot * x
        # normalize step by mean norm to keep stable
        nrm = np.linalg.norm(grad_t, axis=1, keepdims=True).mean() + 1e-12
        x = x - (lr / nrm) * grad_t * 1.0
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_fill(x, rng, n_probe=20000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    deg = (G > m - 1e-4).sum(axis=1)
    ratt = np.where(deg == 0)[0]
    if len(ratt) == 0:
        ratt = np.argsort(deg)[:3]
    fixed_idx = np.setdiff1d(np.arange(n), ratt)
    Xf = x[fixed_idx]
    y = x.copy()
    # sequential greedy hole filling
    placed = []
    for r in ratt:
        P = rng.standard_normal((n_probe, d))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        if len(placed) > 0:
            Pall = np.vstack([Xf] + placed)
        else:
            Pall = Xf
        S = P @ Pall.T
        worst = S.max(axis=1)
        best_i = int(np.argmin(worst))
        y[r] = P[best_i]
        placed.append(y[r:r+1].copy())
    return y

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    BUDGET = 168.0
    rng = _np.random.default_rng(7)
    try:
        xb = _np.load("best.npy").astype(float)
    except Exception:
        xb = _np.load("record.npy").astype(float)
    xb = xb / _np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    n, d = xb.shape
    # reserve final polish time
    loop_deadline = t0 + BUDGET - 28.0
    it = 0
    sigmas = [0.02, 0.05, 0.12, 0.03, 0.20, 0.01]
    while _time.time() < loop_deadline:
        it += 1
        mode = it % 4
        sigma = sigmas[it % len(sigmas)]
        y = xb.copy()
        if mode == 0:
            # global shake
            y = y + sigma * rng.standard_normal((n, d))
            y /= _np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_push(y, steps=150, k=40.0, lr=0.03)
            y = _soft_push(y, steps=100, k=90.0, lr=0.02)
            y = _lp_polish(y, seconds=3.0, r0=0.02, t_deadline=loop_deadline)
        elif mode == 1:
            # rattler hole fill + light noise
            y = _hole_fill(xb, rng, n_probe=15000)
            y = y + 0.015 * rng.standard_normal((n, d))
            y /= _np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_push(y, steps=120, k=60.0, lr=0.02)
            y = _lp_polish(y, seconds=3.5, r0=0.015, t_deadline=loop_deadline)
        elif mode == 2:
            # overloaded cluster shake
            G = xb @ xb.T
            _np.fill_diagonal(G, -2.0)
            m = G.max()
            deg = (G > m - 1e-4).sum(axis=1)
            order = _np.argsort(-deg)
            kbig = min(25, n)
            big = order[:kbig]
            noise = 0.01 * rng.standard_normal((n, d))
            noise[big] = 0.22 * rng.standard_normal((kbig, d))
            y = y + noise
            y /= _np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_push(y, steps=150, k=50.0, lr=0.03)
            y = _lp_polish(y, seconds=3.0, r0=0.02, t_deadline=loop_deadline)
        else:
            # worst-edge kick
            G = xb @ xb.T
            _np.fill_diagonal(G, -2.0)
            iu = _np.triu_indices(n, 1)
            c = G[iu]
            topk = _np.argsort(-c)[:30]
            # kick 4 random worst edges (8 points) hard
            pts = _np.unique(_np.concatenate([iu[0][topk[:12]], iu[1][topk[:12]]]))
            # pick subset
            _np.random.seed(it)  # keep deterministic variety (local)
            sel = rng.choice(pts, size=min(10, len(pts)), replace=False)
            y = y + 0.012 * rng.standard_normal((n, d))
            y[sel] = y[sel] + 0.25 * rng.standard_normal((len(sel), d))
            y /= _np.linalg.norm(y, axis=1, keepdims=True)
            y = _soft_push(y, steps=150, k=70.0, lr=0.025)
            y = _lp_polish(y, seconds=3.0, r0=0.02, t_deadline=loop_deadline)
        v = _max_cos(y)
        if v < best - 1e-12:
            best = v
            xb = y
            # quick extra tighten on improvement if time allows
            if _time.time() < loop_deadline - 5:
                xb = _lp_polish(xb, seconds=2.0, r0=0.008, t_deadline=loop_deadline)
                best = _max_cos(xb)
        # stop if very low time
        if _time.time() >= loop_deadline:
            break
    # final long polish of best
    remain = max(0.0, (t0 + BUDGET - 2.0) - _time.time())
    if remain > 1.0:
        xb = _lp_polish(xb, seconds=min(remain, 25.0), r0=0.01)
    xb = xb / _np.linalg.norm(xb, axis=1, keepdims=True)
    # safety: replace non-finite
    if not _np.all(_np.isfinite(xb)):
        xb = _np.load("best.npy").astype(float)
        xb = xb / _np.linalg.norm(xb, axis=1, keepdims=True)
    return xb

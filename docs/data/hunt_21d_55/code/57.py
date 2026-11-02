import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _find_hole(Y, rng, n_cand=2000):
    d = Y.shape[1]
    cand = rng.randn(n_cand, d)
    cand = _norm_rows(cand)
    md = (cand @ Y.T).max(axis=1)
    return cand[np.argmin(md)].copy()

def _quench(x, steps, a0, a1, lr0, lr1, mu=0.85, clip=0.05):
    n, d = x.shape
    vel = np.zeros_like(x)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (E @ x) / S
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        step = lr * vel
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, clip / sn)
        x = x - step
        x = _norm_rows(x)
    return x

def _lp_polish_deadline(x, end_time, r0=0.002):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = float(r0)
    iu = np.triu_indices(n, 1)
    niter = 0
    while time.time() < end_time and radius > 1e-13 and niter < 60:
        niter += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1200:
            idx = np.argsort(-c[sel])[:1200]
            I = I[idx]; J = J[idx]
            p = 1200
        cb = (x[I] * x[J]).sum(axis=1)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq,
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            if v < m + 1e-12:
                x = y
            radius *= 0.5
    return xb, best

def solve():
    t_start = time.time()
    DEAD = t_start + 168.0
    SEARCH_END = t_start + 148.0
    rng = np.random.RandomState(12345)
    n, d = 55, 21
    cands = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (55, 21):
                cands.append(_norm_rows(a))
        except Exception:
            pass
    best_val = 1e9
    x_best = None
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; x_best = c.copy()
    if x_best is None:
        x_best = _norm_rows(rng.randn(n, d))
        best_val = _max_cos(x_best)
    base = x_best.copy()
    base_val = best_val
    it = 0
    while time.time() < SEARCH_END and it < 500:
        it += 1
        if it % 40 == 0:
            base = x_best.copy(); base_val = best_val
        cur = base.copy()
        mode = it % 7
        if mode in (0, 1, 2):
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mx = float(G.max())
            cnt = (G > mx - 1e-3).sum(axis=1)
            k = {0: 1, 1: 2, 2: 3}[mode]
            victims = np.argsort(-cnt)[:k]
            if rng.rand() < 0.3:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = cur[keep]
            for vi in victims:
                h = _find_hole(Y, rng, n_cand=2000)
                cur[vi] = h
                Y = np.vstack([Y, h[None, :]])
            cur = _norm_rows(cur + (0.006 + rng.rand() * 0.014) * rng.randn(n, d))
            steps = 380; a0 = 25.0; a1 = 420.0; lr0 = 0.02; lr1 = 0.002
        elif mode == 3:
            # random subspace kick: displace few points inside random low-dim subspace
            k = 5 + int(rng.rand() * 3)
            idx = rng.choice(n, size=k, replace=False)
            rdim = 4
            Q, _ = np.linalg.qr(rng.randn(d, rdim))
            amp = 0.05 + rng.rand() * 0.08
            kick = (rng.randn(k, rdim) @ Q.T) * amp
            cur[idx] = cur[idx] + kick
            cur = _norm_rows(cur)
            steps = 380; a0 = 20.0; a1 = 350.0; lr0 = 0.018; lr1 = 0.002
        elif mode == 4:
            cur = _norm_rows(cur + (0.02 + rng.rand() * 0.04) * rng.randn(n, d))
            steps = 350; a0 = 20.0; a1 = 350.0; lr0 = 0.016; lr1 = 0.002
        elif mode == 5:
            cur = _norm_rows(cur + (0.08 + rng.rand() * 0.10) * rng.randn(n, d))
            steps = 420; a0 = 15.0; a1 = 300.0; lr0 = 0.022; lr1 = 0.002
        else:
            # partial reinit: 8 random points fresh
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            cur[idx] = _norm_rows(rng.randn(k, d))
            cur = _norm_rows(cur + 0.01 * rng.randn(n, d))
            steps = 450; a0 = 15.0; a1 = 400.0; lr0 = 0.022; lr1 = 0.002
        # cheap single-point pre-relax for hole modes: move victims only briefly
        cur = _quench(cur, steps, a0, a1, lr0, lr1)
        v = _max_cos(cur)
        if v < best_val - 1e-12:
            best_val = v; x_best = cur.copy()
            base = cur.copy(); base_val = v
        elif v < base_val and rng.rand() < 0.25:
            base = cur.copy(); base_val = v
        elif rng.rand() < 0.02:
            base = cur.copy(); base_val = v
    # single final LP squeeze (sparingly) + tiny greedy margin shrink
    rem = DEAD - time.time()
    if rem > 2.0 and x_best is not None:
        end = time.time() + min(rem - 0.6, 14.0)
        try:
            sq, vs = _lp_polish_deadline(x_best, end, r0=1e-3)
            if vs < best_val:
                best_val = vs; x_best = sq
        except Exception:
            pass
    x_best = _norm_rows(x_best)
    x_best[~np.isfinite(x_best)] = 0.01
    x_best = _norm_rows(x_best)
    return x_best

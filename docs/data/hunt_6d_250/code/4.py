import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth_run(x, beta, steps, lr, mu):
    n = x.shape[0]
    v = np.zeros_like(x)
    best_x = x.copy()
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(np.max(G0))
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights, G-m <= 0 so no overflow
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = (Grad - dot * x) / S
        # clip extreme
        nv = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # avoid huge steps
        Gt = Gt / np.maximum(nv, 1.0)
        v = mu * v + Gt
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if m < best_m:
            best_m = m
            best_x = x.copy()
    return best_x, best_m

def _minimax_run(x, steps, lr0, lr1, K=1500, mu=0.5):
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    v = np.zeros_like(x)
    best_x = x.copy()
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(np.max(G0))
    for s in range(steps):
        lr = lr0 + (lr1 - lr0) * (s / max(1, steps - 1))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tri = G[iu0, iu1]
        if tri.size > K:
            thr = np.partition(tri, -K)[-K]
            # keep width bounded
            if thr < m - 0.005:
                thr = m - 0.005
        else:
            thr = float(np.min(tri))
        W = (G >= thr).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        cnt = W.sum(axis=1, keepdims=True)
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        Gt = Gt / np.maximum(cnt, 1.0)
        # clip
        nv = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / np.maximum(nv, 2.0)
        v = mu * v + Gt
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if m < best_m:
            best_m = m
            best_x = x.copy()
    return best_x, best_m

def _lp_polish(x, seconds, t0, tlim):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < tlim and time.time() < t0 + seconds + 1e9:
        # respect outer limit inside via tlim; seconds is inner
        c = (x @ x.T)[iu]
        m = float(c.max())
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
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if radius < 1e-14:
            break
        if time.time() - t0 > tlim:
            break
    return x

def solve():
    t0 = time.time()
    TLIM = 168.0
    rng = np.random.default_rng(0)
    # load start
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            xc = np.load(fn).astype(np.float64)
            if xc.shape == (250, 6):
                xc = xc / np.linalg.norm(xc, axis=1, keepdims=True)
                m = _max_cos(xc)
                if m < best:
                    best = m
                    x_best = xc.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal((250, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)

    # immediate greedy squeeze of single-pair bottleneck
    xg, mg = _minimax_run(x_best.copy(), steps=2500, lr0=0.003, lr1=0.00002, K=1200, mu=0.5)
    if mg < best:
        best, x_best = mg, xg

    sigmas = [0.08, 0.05, 0.03, 0.02, 0.012, 0.007]
    ntrials = 36
    for t in range(ntrials):
        if time.time() - t0 > 118.0:
            break
        sigma = sigmas[t % len(sigmas)]
        xt = x_best + sigma * rng.standard_normal((250, 6))
        # occasional random re-insertion of a few points into holes
        if t % 6 == 5:
            k = 2 + (t // 6) % 3
            idx = rng.choice(250, size=k, replace=False)
            rr = rng.standard_normal((k, 6))
            rr /= np.linalg.norm(rr, axis=1, keepdims=True)
            xt[idx] = rr
        xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        # smooth stages
        xt, _ = _smooth_run(xt, beta=25.0, steps=500, lr=0.06, mu=0.85)
        if time.time() - t0 > 120.0:
            # quick check
            m = _max_cos(xt)
            if m < best:
                best, x_best = m, xt.copy()
            continue
        xt, _ = _smooth_run(xt, beta=70.0, steps=500, lr=0.04, mu=0.85)
        xt, m2 = _minimax_run(xt, steps=700, lr0=0.002, lr1=0.00003, K=1500, mu=0.5)
        if m2 < best - 1e-12:
            best, x_best = m2, xt.copy()
        elif m2 < best:
            best, x_best = m2, xt.copy()

    # final intensive squeeze
    if time.time() - t0 < TLIM - 30.0:
        xf, mf = _minimax_run(x_best.copy(), steps=2500, lr0=0.0015, lr1=0.000008, K=1800, mu=0.6)
        if mf < best:
            best, x_best = mf, xf
    # final LP polish with remaining time
    remain = TLIM - (time.time() - t0) - 2.0
    if remain > 5.0:
        # use at most 28s for LP
        lp_time = min(remain, 28.0)
        tlim_lp = TLIM - 1.0
        # _lp_polish loops until tlim; emulate seconds via tlim
        start_lp = time.time()
        # run with internal time check
        x_best = _lp_polish(x_best, seconds=lp_time, t0=start_lp, tlim=start_lp + lp_time)
        # re-evaluate (no need to revert, LP only keeps improvements internally)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

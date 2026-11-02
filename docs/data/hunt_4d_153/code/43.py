import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

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
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    Iu0, Ju0 = iu
    lb = np.log(lr1 / lr0) / max(steps - 1, 1)
    bb = np.log(beta1 / beta0) / max(steps - 1, 1)
    for s in range(steps):
        beta = beta0 * np.exp(bb * s)
        lr = lr0 * np.exp(lb * s)
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

def _small_rot(rng, s):
    A = np.eye(4) + s * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    ph = np.sign(d)
    ph[ph == 0] = 1.0
    Q = Q * ph
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((153, 4))
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    x_cur = x_best.copy()
    cur_val = best_val
    # temperature schedule for basin hopping
    T0, T1 = 0.004, 0.0004
    it = 0
    # precompute for speed
    while time.time() - t0 < 150.0:
        it += 1
        frac = (time.time() - t0) / 150.0
        Temp = T0 * (T1 / T0) ** frac
        mode = it % 8
        y = x_cur.copy()
        if mode == 0 or mode == 1 or mode == 7:
            # rigid cap rotation: cooperative move of many core points
            if mode == 0: s, thr = 0.06, 0.90
            elif mode == 1: s, thr = 0.16, 0.88
            else: s, thr = 0.32, 0.85
            c = rng.standard_normal(4); c /= np.linalg.norm(c)
            dots = y @ c
            S = np.where(dots > thr)[0]
            if len(S) < 4 or len(S) > 45:
                # pick cap around a random existing point
                ci = rng.integers(n)
                c = y[ci] + 0.3 * rng.standard_normal(4)
                c /= np.linalg.norm(c)
                dots = y @ c
                # adaptive threshold to get 8..25 points
                order = np.argsort(-dots)
                k = int(rng.integers(8, 26))
                S = order[:k]
            R = _small_rot(rng, s)
            y[S] = y[S] @ R.T
            y += rng.standard_normal(y.shape) * (0.008 if mode == 0 else 0.015)
        elif mode == 2 or mode == 3:
            s = 0.05 if mode == 2 else 0.12
            M = rng.standard_normal((4, 4))
            E = (M + M.T) * 0.5
            E -= np.trace(E) / 4.0 * np.eye(4)
            E *= s / np.linalg.norm(E)
            # global anisotropic strain (coherent)
            y = y @ (np.eye(4) + E)
            y += rng.standard_normal(y.shape) * 0.01
        elif mode == 4:
            # local jitter of neighbourhood
            ci = rng.integers(n)
            dd = y @ y[ci]
            order = np.argsort(-dd)
            k = int(rng.integers(8, 20))
            S = order[:k]
            y[S] += rng.standard_normal((len(S), d)) * 0.09
        elif mode == 5:
            # rattler / hole re-insertion: find low-contact points, replace best-of-random
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            tight = G > mx - 1e-7
            cnt = tight.sum(axis=1)
            # pick worst point
            idx = int(np.argmin(cnt))
            cand = rng.standard_normal((4000, 4))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            sims = cand @ y.T
            sims[:, idx] = -2.0
            mc = sims.max(axis=1)
            bi = int(np.argmin(mc))
            y[idx] = cand[bi]
            y += rng.standard_normal(y.shape) * 0.008
        else:
            # global moderate jitter
            y += rng.standard_normal(y.shape) * 0.045
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed soft-max repulsion
        if mode in (7, 3):
            steps, b0, b1, l0, l1 = 900, 25.0, 350.0, 0.008, 0.0006
        elif mode in (1, 2):
            steps, b0, b1, l0, l1 = 700, 30.0, 400.0, 0.006, 0.0005
        else:
            steps, b0, b1, l0, l1 = 500, 30.0, 400.0, 0.006, 0.0005
        try:
            y2 = _smooth_opt(y, steps, b0, b1, l0, l1, iu)
        except Exception:
            continue
        v2 = _max_cos(y2)
        # only LP-polish promising candidates to save time
        if v2 > cur_val + 0.025 and v2 > best_val + 0.025:
            # too bad, reject without LP
            if rng.random() < 0.02:
                x_cur = y2; cur_val = v2
            continue
        remain = HARD - (time.time() - t0)
        if remain < 8:
            break
        lp_budget = 1.2 if v2 < min(cur_val, best_val) + 0.002 else 0.7
        try:
            y3, v3 = _lp_polish(y2, lp_budget)
        except Exception:
            continue
        if v3 < best_val:
            x_best = y3.copy(); best_val = v3
            x_cur = y3.copy(); cur_val = v3
        elif v3 < cur_val:
            x_cur = y3.copy(); cur_val = v3
        else:
            # Metropolis to change basin
            if rng.random() < np.exp(-(v3 - cur_val) / max(Temp, 1e-9)):
                x_cur = y3.copy(); cur_val = v3
        if time.time() - t0 > HARD - 12:
            break
    # final high-precision polish of incumbent best
    try:
        remain = HARD - (time.time() - t0)
        if remain > 3:
            x_best, best_val = _lp_polish(x_best, min(remain - 1, 10.0))
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

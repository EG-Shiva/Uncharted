import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _refine(X0, stages, mu=0.85, max_step=0.05):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    for beta, steps, lr in stages:
        for _ in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            m = float(np.max(G))
            # weights for LogSumExp(max)
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            rs = np.sum(W, axis=1, keepdims=True) + 1e-12
            grad = W @ X
            d = np.sum(grad * X, axis=1, keepdims=True)
            gt = (grad - d * X) / rs
            vel = mu * vel - lr * gt
            # clip per-point step
            vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True)) + 1e-12
            scale = np.minimum(1.0, max_step / vn)
            # scale is (n,1) broadcast
            vel = vel * scale
            X = X + vel
            X = _normalize(X)
    return X


def _lp_polish(x, seconds, t_end):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() + 1.0 > t_end and radius < 5e-4:
            break
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_HARD = t0 + 175.0
    T_PHASE1 = t0 + 122.0
    rng = np.random.default_rng(0)

    try:
        best = np.load("best.npy")
    except Exception:
        best = np.load("record.npy")
    best = _normalize(np.asarray(best, dtype=np.float64))
    if best.shape != (272, 4):
        # fallback random if shape wrong
        best = _normalize(rng.standard_normal((272, 4)))
    best_val = _max_cos(best)

    stages = [(60.0, 250, 0.012), (150.0, 300, 0.008), (400.0, 350, 0.004), (800.0, 250, 0.002)]

    n = 272
    hop = 0
    # pre-allocate random buffer reuse not needed
    while time.time() < T_PHASE1 and hop < 300:
        mode = hop % 6
        Xp = best.copy()
        if mode == 0:
            Xp = Xp + 0.015 * rng.standard_normal(Xp.shape)
        elif mode == 1:
            Xp = Xp + 0.040 * rng.standard_normal(Xp.shape)
        elif mode == 2:
            Xp = Xp + 0.085 * rng.standard_normal(Xp.shape)
        elif mode == 3:
            # targeted kick of most-contacted points
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 1e-6, axis=1)
            idx = np.argsort(-cnt)[:25]
            Xp = Xp + 0.010 * rng.standard_normal(Xp.shape)
            Xp[idx] = Xp[idx] + 0.12 * rng.standard_normal((len(idx), 4))
        elif mode == 4:
            # relocate worst points to holes
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 1e-6, axis=1)
            k = 4
            worst = np.argsort(-cnt)[:k]
            # find holes
            S = _normalize(rng.standard_normal((3000, 4)))
            dots = S @ best.T
            md = np.max(dots, axis=1)
            hole_idx = np.argsort(md)[:k]
            Xp[worst] = S[hole_idx] + 0.005 * rng.standard_normal((k, 4))
            # small shake rest
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xp[mask] = Xp[mask] + 0.008 * rng.standard_normal((n - k, 4))
        else:
            # random subset resample
            k = 6
            idx = rng.choice(n, size=k, replace=False)
            Xp[idx] = rng.standard_normal((k, 4))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xp[mask] = Xp[mask] + 0.010 * rng.standard_normal((n - k, 4))
        Xp = _normalize(Xp)
        # time check before heavy refine
        if time.time() + 0.5 > T_PHASE1:
            break
        Xnew = _refine(Xp, stages, mu=0.85, max_step=0.05)
        v = _max_cos(Xnew)
        if v < best_val:
            best = Xnew
            best_val = v
        hop += 1

    # final polish with remaining time
    t_left = T_HARD - time.time()
    if t_left > 5.0:
        # reserve 2 sec margin
        best = _lp_polish(best, t_left - 2.0, T_HARD - 1.0)
        best = _normalize(best)

    return best

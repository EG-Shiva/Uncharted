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
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c_all > m - thr
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap size for speed
        csel = c_all[sel]
        if p > 3500:
            keep = np.argsort(csel)[-3500:]
            I = I[keep]
            J = J[keep]
            csel = csel[keep]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_run(x, steps, k0, k1, lr0, lr1, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights exp(k*(G-m)), max 1
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        proj = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - proj * x
        norms = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        direction = gt / norms
        vel = mu * vel - lr * direction
        x = x + vel
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x


def hole_reinsert(x, idx, rng, n_try=5000):
    n, d = x.shape
    others = np.delete(x, idx, axis=0)
    cand = rng.normal(size=(n_try, d))
    cn = np.linalg.norm(cand, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    cand = cand / cn
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    x[idx] = cand[j]
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    # load starts
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None and a.shape == (181, 9) and np.all(np.isfinite(a)):
            nrm = np.linalg.norm(a, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            a = a / nrm
            cands.append(a)
    if len(cands) == 0:
        z = rng.normal(size=(181, 9))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi]
    best = float(vals[bi])

    # tiny initial squeeze if time allows (already near optimum)
    if time.time() - t0 < 165:
        x_best = lp_polish(x_best, 4.0)
        best = max_cos(x_best)

    stall = 0
    it = 0
    # explore until ~150s, reserve rest for final polish
    while time.time() - t0 < 148.0:
        it += 1
        # schedule perturbation strength
        if stall < 4:
            sigma = 0.008
            n_hole = 2
            n_try = 4000
        elif stall < 12:
            sigma = 0.02
            n_hole = 3
            n_try = 6000
        elif stall < 25:
            sigma = 0.045
            n_hole = 4
            n_try = 8000
        else:
            sigma = 0.09
            n_hole = 5
            n_try = 8000

        x = x_best.copy()
        n, d = x.shape
        # find loosest points (smallest per-point max)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        per_max = G.max(axis=1)
        order = np.argsort(per_max)  # ascending: loosest first
        # global jitter
        x = x + rng.normal(size=(n, d)) * sigma
        # extra cluster shake when stalled: pick hub (largest contacts) neighbourhood
        if stall >= 4:
            # hub = point with largest per_max (most jammed)
            hub = int(np.argmax(per_max))
            sims = x_best[hub] @ x_best.T
            nn = np.argsort(-sims)[:10]
            x[nn] += rng.normal(size=(len(nn), d)) * (sigma * 2.0 + 0.02)
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
        # greedy hole reinsertion for loosest
        for k in range(min(n_hole, n)):
            idx = int(order[k])
            # re-evaluate hole with current x (sequential)
            hole_reinsert(x, idx, rng, n_try=n_try)

        # smooth repulsion annealing
        if stall < 12:
            x = smooth_run(x, steps=280, k0=60.0, k1=280.0, lr0=0.014, lr1=0.002, mu=0.8)
        else:
            x = smooth_run(x, steps=350, k0=40.0, k1=220.0, lr0=0.020, lr1=0.003, mu=0.75)

        v_smooth = max_cos(x)
        if v_smooth > best + 0.008:
            stall += 1
            continue
        # short LP burst
        remain = 148.0 - (time.time() - t0)
        if remain < 2.0:
            break
        burst = 3.0 if v_smooth < best + 0.002 else 2.0
        burst = min(burst, remain)
        x = lp_polish(x, burst)
        v = max_cos(x)
        if v < best - 1e-12:
            x_best = x
            best = v
            stall = 0
        else:
            stall += 1
        # safety: if single iteration ate too much time, exit
        if time.time() - t0 >= 150.0:
            break

    # final long polish with remaining budget
    remaining = 176.0 - (time.time() - t0)
    if remaining > 3.0:
        x_best = lp_polish(x_best, remaining - 1.0)
    # normalize strictly
    nrm = np.linalg.norm(x_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x_best = x_best / nrm
    return x_best

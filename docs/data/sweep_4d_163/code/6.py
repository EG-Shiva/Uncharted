import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, init_radius=0.008):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        G = x @ x.T
        c = G[iu]
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
        obj = np.zeros(nv); obj[-1] = 1.0
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

def smooth_relax(x, steps, a0, a1, lr):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights for soft-max of max
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        tot = W.sum() * 0.5
        if not np.isfinite(tot) or tot <= 0:
            break
        Gr = (W @ x) / tot
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        # clip extreme steps
        nm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # limit per-point step to 5*lr to avoid blowup
        scale = np.minimum(1.0, (5 * lr) / (lr * nm + 1e-12))
        x = x - lr * Gt * scale
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
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
            c = np.asarray(c, dtype=float)
            if c.shape == (163, 4):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append(c)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        best = rng0.standard_normal((163, 4))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
    else:
        vals = [max_cos(c) for c in cands]
        best = cands[int(np.argmin(vals))]
    best_val = max_cos(best)
    rng = np.random.default_rng(0)
    it = 0
    # reserve ~7s for final polish
    while time.time() < deadline - 7.0:
        it += 1
        time_left = deadline - 7.0 - time.time()
        if time_left < 0.5:
            break
        r = rng.random()
        trial = None
        if r < 0.45:
            sigma = float(10 ** rng.uniform(-1.9, -0.8))  # ~0.012..0.16
            trial = best + sigma * rng.standard_normal(best.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif r < 0.8:
            trial = best.copy()
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            m = int(rng.integers(2, 11))
            cand = order[:max(2 * m, 12)]
            idx = rng.choice(cand, size=m, replace=False)
            trial[idx] = rng.standard_normal((m, 4))
            trial[idx] /= np.linalg.norm(trial[idx], axis=1, keepdims=True)
            trial += 0.004 * rng.standard_normal(trial.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            trial = best.copy()
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            i = int(np.argmax(per))
            S = rng.standard_normal((2500, 4))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            others = np.delete(trial, i, axis=0)
            sm = (S @ others.T).max(axis=1)
            j = int(np.argmin(sm))
            trial[i] = S[j]
            # small shake of neighbours of i to open space
            trial += 0.003 * rng.standard_normal(trial.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        # cheap smooth descent with randomized schedule for diversity
        steps = int(rng.choice([150, 200, 250]))
        a0 = float(rng.choice([80.0, 150.0, 250.0]))
        a1 = float(rng.choice([600.0, 900.0, 1300.0]))
        lr = float(rng.choice([0.006, 0.009, 0.013]))
        trial = smooth_relax(trial, steps, a0, a1, lr)
        v = max_cos(trial)
        # only LP-polish if promising
        if v > best_val + 0.015:
            continue
        secs = min(3.2, deadline - 7.0 - time.time())
        if secs < 0.6:
            # no time for LP, just check
            if v < best_val - 1e-12:
                best, best_val = trial, v
            continue
        init_r = 0.01 if v > best_val else 0.006
        trial2 = lp_polish(trial, secs, init_radius=init_r)
        v2 = max_cos(trial2)
        if v2 < best_val - 1e-12:
            best, best_val = trial2, v2
        # opportunistic: if we just improved a lot, spend a bit more
        if v2 < best_val and time.time() < deadline - 10:
            pass
    rem = deadline - time.time() - 1.0
    if rem > 1.0:
        best = lp_polish(best, min(rem, 10.0), init_radius=0.002)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

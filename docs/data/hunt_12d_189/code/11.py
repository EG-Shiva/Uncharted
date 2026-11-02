import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _degrees(x, topk=1200):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    iu = np.triu_indices(n, 1)
    vals = g[iu]
    k = min(topk, vals.size)
    part = np.argpartition(vals, -k)[-k:]
    I = iu[0][part]; J = iu[1][part]
    deg = np.zeros(n, dtype=int)
    np.add.at(deg, I, 1)
    np.add.at(deg, J, 1)
    return deg

def anneal(x, steps, a0, a1, l0, l1):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = l0 * (l1 / l0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights focused on near-max pairs
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        GX = W @ x
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        GX = GX / rs
        dot = (GX * x).sum(axis=1, keepdims=True)
        Gt = GX - dot * x
        # clip extreme tangents for stability
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # scale down huge steps (should be ~1)
        scale = np.minimum(1.0, 1.0 / gn)
        Gt = Gt * scale
        x = x - lr * Gt
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    return x

def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() + 1.0 > t_end:
            break
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

def solve():
    t0 = time.time()
    T_LIMIT = 178.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.normal(size=(189, 12))
    x0 = np.asarray(x0, dtype=np.float64)
    # ensure shape (189,12)
    if x0.shape != (189, 12):
        # fallback: tile/reshape deterministically
        flat = x0.ravel()
        need = 189 * 12
        rep = np.tile(flat, (need // flat.size + 1))[:need]
        x0 = rep.reshape(189, 12)
    x0 = x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-30)
    best = x0.copy()
    best_val = max_cos(best)
    base = best.copy()

    deg = _degrees(best, topk=1200)
    order = np.argsort(-deg)
    hubs = order[:15].copy()
    rattlers = np.where(deg == 0)[0].copy()
    if rattlers.size == 0:
        rattlers = order[-8:].copy()

    trial_count = 0
    # search budget: leave ~55s for LP polish
    search_end = t0 + 115.0
    # pre-draw hyperparams deterministically via rng sequence
    while time.time() < search_end and time.time() < t_end - 60:
        trial_count += 1
        # log-uniform sigma
        u = rng.random()
        if u < 0.4:
            sigma = float(np.exp(rng.uniform(np.log(0.004), np.log(0.015))))
        elif u < 0.8:
            sigma = float(np.exp(rng.uniform(np.log(0.015), np.log(0.05))))
        else:
            sigma = float(np.exp(rng.uniform(np.log(0.05), np.log(0.12))))
        steps = int(rng.integers(700, 1800))
        a0 = float(rng.uniform(12, 35))
        a1 = float(rng.uniform(200, 600))
        l0 = float(rng.uniform(0.008, 0.028))
        l1 = float(rng.uniform(0.001, 0.005))

        noise = rng.normal(0.0, sigma, size=base.shape)
        # boost hubs + rattlers to force topology change
        boost = np.unique(np.concatenate([hubs, rattlers]))
        if boost.size > 0:
            noise[boost] *= 3.0
            noise[boost] += rng.normal(0.0, sigma, size=(boost.size, base.shape[1]))
        # occasional random subset boost
        if rng.random() < 0.5:
            k = int(rng.integers(5, 25))
            idx = rng.choice(base.shape[0], size=k, replace=False)
            noise[idx] += rng.normal(0.0, sigma * 2.0, size=(k, base.shape[1]))

        trial = base + noise
        trial /= (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-30)
        trial = anneal(trial, steps, a0, a1, l0, l1)
        v = max_cos(trial)
        if v < best_val:
            best_val = float(v)
            best = trial.copy()
            base = trial.copy()
            # refresh hubs/rattlers
            try:
                deg = _degrees(best, topk=1200)
                order = np.argsort(-deg)
                hubs = order[:15].copy()
                rr = np.where(deg == 0)[0]
                rattlers = rr.copy() if rr.size > 0 else order[-8:].copy()
            except Exception:
                pass
        else:
            # occasional diversification from non-best basin
            if v < best_val + 0.008 and rng.random() < 0.08:
                base = trial.copy()
        if trial_count % 25 == 0:
            base = best.copy()

    # final polish with remaining time
    remain = t_end - time.time()
    if remain > 5:
        # reserve 2s margin
        try:
            best = lp_polish(best, remain - 2.0, t_end - 2.0)
        except Exception:
            pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    # strict validity: no NaN/zero rows
    best = np.nan_to_num(best, nan=1.0, posinf=1.0, neginf=-1.0)
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    return best

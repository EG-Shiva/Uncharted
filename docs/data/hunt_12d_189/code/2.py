import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _soft_anneal(x, steps, a0, a1, l0, l1):
    n, d = x.shape
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights focused on near-max pairs
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        # clip extreme tangents for stability
        # (per-row norm clip)
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-30
        # scale down huge steps (should be <=~1)
        scale = np.minimum(1.0, 1.0 / (nr + 1e-12))
        # actually keep direction, limit to 1
        Gt = Gt * np.minimum(1.0, 1.0 / np.maximum(1.0, nr))
        x = x - lr * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(12345)
    # load start
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
            try:
                a = np.array(c, dtype=np.float64)
                if a.shape == (189, 12):
                    a = a / np.linalg.norm(a, axis=1, keepdims=True)
                    cands.append(( _max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((189, 12))
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    # quick initial short polish if time allows (already polished, skip long)
    # main basin-hopping loop
    trial = 0
    while time.time() < deadline - 12.0:
        if trial > 60:
            break
        mode = trial % 3
        x = best_x.copy()
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        pt_max = np.max(G0, axis=1)
        order = np.argsort(pt_max)  # loose first
        n = 189
        if mode == 0:
            k = 6 + (trial % 5)  # 6..10 loose points
            loose = order[:k]
            # sequential hole filling
            M = 15000
            for idx in loose:
                R = rng.standard_normal((M, 12))
                R = R / np.linalg.norm(R, axis=1, keepdims=True)
                D = R @ x.T
                D[:, idx] = -2.0
                vals = np.max(D, axis=1)
                j = int(np.argmin(vals))
                x[idx] = R[j]
            # small kick to tight hubs
            tight = order[-4:]
            x[tight] = x[tight] + 0.008 * rng.standard_normal((len(tight), 12))
            x[tight] = x[tight] / np.linalg.norm(x[tight], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 700, 30.0, 220.0, 0.25, 0.02
            lp_budget = 4.0
        elif mode == 1:
            tight = order[-8:]
            rnd = rng.choice(n, size=8, replace=False)
            sel = np.unique(np.concatenate([tight, rnd]))
            sigma = 0.015 + 0.008 * (trial % 4)
            x[sel] = x[sel] + sigma * rng.standard_normal((len(sel), 12))
            x[sel] = x[sel] / np.linalg.norm(x[sel], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 800, 25.0, 180.0, 0.30, 0.02
            lp_budget = 4.0
        else:
            sigma = 0.004 + 0.002 * (trial % 5)
            x = x + sigma * rng.standard_normal(x.shape)
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 600, 35.0, 250.0, 0.20, 0.015
            lp_budget = 3.5
        # anneal
        # adapt steps if little time left
        if time.time() + 6.0 > deadline - 12.0:
            steps = min(steps, 300)
            lp_budget = min(lp_budget, 2.0)
        x = _soft_anneal(x, steps, a0, a1, l0, l1)
        # polish
        lp_dead = min(time.time() + lp_budget, deadline - 8.0)
        if lp_dead > time.time() + 0.5:
            x, v = _lp_polish(x, lp_dead)
        else:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            v = _max_cos(x)
        if v < best_val - 1e-12:
            best_val, best_x = v, x.copy()
        trial += 1
    # final squeeze on best if time remains
    if time.time() < deadline - 3.0:
        best_x, best_val2 = _lp_polish(best_x, min(deadline - 1.0, time.time() + 8.0), radius_init=5e-4)
        best_val = min(best_val, best_val2)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

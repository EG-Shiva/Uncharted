import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _pt_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _anneal(x, steps, a0, a1, l0, l1):
    n, d = x.shape
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt * np.minimum(1.0, 1.0 / np.maximum(1.0, nr))
        x = x - lr * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _deep_hole_refined(x, exclude, M, rng):
    n, d = x.shape
    chunk = 4000
    best_R = None
    best_v = 2.0
    done = 0
    while done < M:
        c = min(chunk, M - done)
        R = rng.standard_normal((c, d))
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ x.T
        if exclude >= 0:
            D[:, exclude] = -2.0
        vals = np.max(D, axis=1)
        j = int(np.argmin(vals))
        if float(vals[j]) < best_v:
            best_v = float(vals[j])
            best_R = R[j].copy()
        done += c
    # local refinement: move away from nearest neighbours
    R = best_R
    for it in range(150):
        f = it / 149.0
        alpha = 40.0 * (300.0 / 40.0) ** f
        lr = 0.08 - 0.07 * f
        D = x @ R
        if exclude >= 0:
            D[exclude] = -2.0
        m = float(np.max(D))
        W = np.exp(alpha * (D - m))
        W[exclude] = 0.0 if exclude >= 0 else W[exclude]
        s = float(np.sum(W)) + 1e-30
        S = (W @ x) / s
        Gt = S - float(S @ R) * R
        nrm = float(np.linalg.norm(Gt))
        if nrm > 1.0:
            Gt = Gt / nrm
        R = R - lr * Gt
        R = R / np.linalg.norm(R)
    return R

def _lp_polish(x, deadline, radius_init=6e-4, max_pairs=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(8.0 * radius, 1.5e-4)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt > max_pairs:
            idx = np.argpartition(-c, max_pairs)[:max_pairs]
            mask = np.zeros_like(c, dtype=bool)
            mask[idx] = True
            sel = mask
            cnt = max_pairs
        if cnt == 0:
            radius *= 0.5
            continue
        I, J = iu[0][sel], iu[1][sel]
        p = cnt
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(987654321)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=np.float64)
            if a.shape == (189, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((189, 12))
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    n, d = 189, 12
    trial = 0
    while time.time() < deadline - 14.0 and trial < 300:
        mode = trial % 4
        x = best_x.copy()
        pm = _pt_max(x)
        order = np.argsort(pm)
        if mode == 0:
            k = 8
            loose = order[:k]
            for idx in loose:
                h = _deep_hole_refined(x, int(idx), 6000, rng)
                x[int(idx)] = h
            tight = order[-3:]
            x[tight] += 0.012 * rng.standard_normal((len(tight), d))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            s1 = (320, 30.0, 220.0, 0.25, 0.02)
        elif mode == 1:
            nh = 4 + (trial % 3)
            hubs = order[-nh:]
            for idx in hubs:
                h = _deep_hole_refined(x, int(idx), 5000, rng)
                x[int(idx)] = h
            s1 = (380, 25.0, 200.0, 0.30, 0.02)
        elif mode == 2:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            proj = x @ v
            thr = np.quantile(proj, 0.80 - 0.05 * (trial % 4))
            idxs = np.where(proj > thr)[0]
            sigma = 0.03 + 0.01 * (trial % 3)
            x[idxs] += sigma * rng.standard_normal((len(idxs), d))
            x[idxs] /= np.linalg.norm(x[idxs], axis=1, keepdims=True)
            h = _deep_hole_refined(x, int(order[0]), 4000, rng)
            x[int(order[0])] = h
            s1 = (320, 28.0, 240.0, 0.28, 0.018)
        else:
            sigma = 0.008 + 0.004 * (trial % 3)
            sel = rng.choice(n, size=24, replace=False)
            x[sel] += sigma * rng.standard_normal((len(sel), d))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            if trial % 8 == 7:
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                pick = rng.choice(n, size=3, replace=False)
                x[pick] = x[pick] - 2.0 * np.outer((x[pick] @ u), u)
                x[pick] /= np.linalg.norm(x[pick], axis=1, keepdims=True)
            s1 = (280, 35.0, 250.0, 0.20, 0.015)
        steps, a0, a1, l0, l1 = s1
        x = _anneal(x, steps, a0, a1, l0, l1)
        # sharp convergence phase to approximate LP optimum without LP cost
        x = _anneal(x, 140, 300.0, 800.0, 0.02, 0.005)
        vfast = _max_cos(x)
        if vfast < best_val - 1e-12:
            best_val, best_x = vfast, x.copy()
        trial += 1
    if time.time() < deadline - 2.0:
        rem = min(deadline - 1.0, time.time() + 11.0)
        best_x, v2 = _lp_polish(best_x, rem, radius_init=5e-4)
        best_val = min(best_val, v2)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

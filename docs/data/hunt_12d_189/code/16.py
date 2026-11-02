import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, budget, radius_init=3e-3, max_it=60):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    start = time.time()
    it = 0
    while time.time() - start < budget and radius > 1e-13 and it < max_it:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = max(25.0 * radius, 2e-9)
        sel = c > m - thr
        if np.count_nonzero(sel) < 10:
            # take top 500 closest
            k = min(500, c.size)
            idx = np.argpartition(-c, k - 1)[:k]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, ci)), shape=(p, nv)).tocsr()
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x, best

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
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_fill(x, idx_list, rng, M=8000):
    # sequential deepest-hole reinsertion
    for idx in idx_list:
        R = rng.standard_normal((M, 12))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ x.T
        D[:, idx] = -2.0
        vals = np.max(D, axis=1)
        j = int(np.argmin(vals))
        x[idx] = R[j]
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=np.float64)
            if a.shape == (189, 12):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((189, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    n = 189
    trial = 0
    # basin-hopping with tight budgets
    while time.time() < deadline - 30.0 and trial < 10:
        trial += 1
        x = best_x.copy()
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        pt_max = np.max(G0, axis=1)
        # contacts at 1e-7 for robustness
        cmax = float(np.max(G0))
        contacts = np.sum(G0 > cmax - 1e-7, axis=1)
        loose_order = np.argsort(pt_max)  # ascending: loosest first
        tight_order = np.argsort(-pt_max)
        hub_order = np.argsort(-contacts)
        mode = trial % 4
        if mode == 1:
            k = 4 + (trial % 3)  # 4-6 loosest
            idxs = loose_order[:k].tolist()
            x = _hole_fill(x, idxs, rng, M=8000)
            # kick hubs lightly
            hk = hub_order[:3]
            x[hk] += 0.006 * rng.standard_normal((len(hk), 12))
            x[hk] /= np.linalg.norm(x[hk], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 220, 25.0, 250.0, 0.20, 0.015
            lp_b = 5.0
        elif mode == 2:
            k = 3 + (trial % 2)
            idxs = hub_order[:k].tolist()
            x = _hole_fill(x, idxs, rng, M=8000)
            lk = loose_order[:3]
            x[lk] += 0.01 * rng.standard_normal((len(lk), 12))
            x[lk] /= np.linalg.norm(x[lk], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 220, 40.0, 300.0, 0.22, 0.015
            lp_b = 5.0
        elif mode == 3:
            # global shake + hole fill one worst
            x += 0.004 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            idxs = loose_order[:2].tolist()
            x = _hole_fill(x, idxs, rng, M=12000)
            steps, a0, a1, l0, l1 = 260, 30.0, 200.0, 0.18, 0.01
            lp_b = 5.0
        else:
            # random subset shake harder
            sel = rng.choice(n, size=10, replace=False)
            x[sel] += 0.03 * rng.standard_normal((len(sel), 12))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            idxs = loose_order[:4].tolist()
            x = _hole_fill(x, idxs, rng, M=6000)
            steps, a0, a1, l0, l1 = 200, 20.0, 180.0, 0.25, 0.02
            lp_b = 4.0
        if time.time() > deadline - 30.0:
            break
        x = _anneal(x, steps, a0, a1, l0, l1)
        if time.time() > deadline - 28.0:
            v = _max_cos(x)
            if v < best_val:
                best_val, best_x = v, x.copy()
            continue
        x2, v2 = _lp_polish(x, min(lp_b, max(1.0, deadline - 28.0 - time.time())), radius_init=4e-3)
        if v2 < best_val:
            best_val, best_x = v2, x2.copy()
    # final polish of best with remaining time
    remain = deadline - time.time() - 1.0
    if remain > 3.0:
        remain = min(remain, 24.0)
        best_x, best_val = _lp_polish(best_x, remain, radius_init=1e-3)
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

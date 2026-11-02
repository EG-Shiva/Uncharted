import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline, radius_init=1e-3, max_iters=40, p_cap=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iters:
        it += 1
        G = x @ x.T
        cfull = G[iuI, iuJ]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 2e-9)
        sel = cfull > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > p_cap:
            # keep largest p_cap
            idx = np.where(sel)[0]
            part = np.argpartition(cfull[idx], -p_cap)[-p_cap:]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iuI[sel]; J = iuJ[sel]
        csel = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cI, J[:, None] * d + cI, np.full((p, 1), n * d)], 1).ravel()
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
        if res.status != 0 or time.time() >= deadline:
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

def _soft_anneal(x, steps, a0, a1, l0, l1, deadline):
    n, d = x.shape
    for s in range(steps):
        if time.time() >= deadline:
            break
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # avoid overflow: exp(alpha*(G-m)) in [0,1]
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        # limit tangent step to 1
        Gt = Gt / np.maximum(1.0, nr)
        x = x - lr * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.default_rng(7)
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
                    if np.all(np.isfinite(a)):
                        cands.append((_max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((189, 12))
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    n = 189
    trial = 0
    # time-boxed basin hopping
    while time.time() < deadline - 12.0 and trial < 50:
        x = best_x.copy()
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        pt_max = np.max(G0, axis=1)
        order = np.argsort(pt_max)  # loose first
        mode = trial % 4
        if mode == 0:
            k = 5 + (trial % 4)  # 5..8
            loose = order[:k]
            M = 8000
            for idx in loose:
                if time.time() >= deadline - 11.0:
                    break
                R = rng.standard_normal((M, 12))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                D = R @ x.T
                D[:, idx] = -2.0
                vals = np.max(D, axis=1)
                j = int(np.argmin(vals))
                x[idx] = R[j]
            tight = order[-4:]
            x[tight] += 0.008 * rng.standard_normal((len(tight), 12))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 300, 30.0, 220.0, 0.25, 0.02
            lp_budget = 2.2
        elif mode == 1:
            tight = order[-8:]
            rnd = rng.choice(n, size=6, replace=False)
            sel = np.unique(np.concatenate([tight, rnd]))
            sigma = 0.012 + 0.005 * (trial % 3)
            x[sel] += sigma * rng.standard_normal((len(sel), 12))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 320, 25.0, 180.0, 0.30, 0.02
            lp_budget = 2.2
        elif mode == 2:
            # single worst-point hole fill + small global shake
            worst = int(order[-1])
            M = 8000
            R = rng.standard_normal((M, 12))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            D = R @ x.T
            D[:, worst] = -2.0
            vals = np.max(D, axis=1)
            x[worst] = R[int(np.argmin(vals))]
            sigma = 0.005
            x += sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 260, 35.0, 250.0, 0.20, 0.015
            lp_budget = 1.8
        else:
            sigma = 0.010
            sel = rng.choice(n, size=14, replace=False)
            x[sel] += sigma * rng.standard_normal((len(sel), 12))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 280, 28.0, 200.0, 0.28, 0.02
            lp_budget = 2.0
        # anneal with inner deadline
        ann_dead = min(time.time() + 1.5, deadline - 10.0)
        x = _soft_anneal(x, steps, a0, a1, l0, l1, ann_dead)
        # polish
        lp_dead = min(time.time() + lp_budget, deadline - 9.0)
        if lp_dead > time.time() + 0.3:
            x, v = _lp_polish(x, lp_dead)
        else:
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            v = _max_cos(x)
        if v < best_val - 1e-12:
            best_val, best_x = v, x.copy()
        trial += 1
    if time.time() < deadline - 2.0:
        best_x, v2 = _lp_polish(best_x, min(deadline - 0.5, time.time() + 6.0), radius_init=5e-4, max_iters=30)
        best_val = min(best_val, v2)
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

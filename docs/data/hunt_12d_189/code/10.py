import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish_cap(x, deadline, radius_init=1e-3, max_iter=25, p_cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    for _ in range(max_iter):
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(np.max(c))
        band = max(20.0 * radius, 2e-9)
        sel = c > m - band
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > p_cap:
            # keep tightest p_cap
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[-p_cap:]
            keep = idx[order]
            mask = np.zeros_like(sel, dtype=bool)
            mask[keep] = True
            sel = mask
            cnt = p_cap
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2*d+1)] = 0  # placeholder, will overwrite below (vectorized instead)
        # build cols vectorized
        base = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        cols = base
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x, best

def _soft_anneal(x, steps, a0, a1, l0, l1, deadline=1e30):
    n, d = x.shape
    for s in range(steps):
        if (s & 63) == 0 and time.time() >= deadline:
            break
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # avoid overflow: exp(alpha*(G-m)), G-m<=0
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        # limit tangent step to 1
        Gt = Gt * np.minimum(1.0, 1.0 / np.maximum(1.0, nr))
        x = x - lr * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _diverse_holes(x, loose, rng, M=6000, sep=0.95):
    n, d = x.shape
    R = rng.standard_normal((M, d))
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ x.T
    if len(loose) > 0:
        D[:, np.asarray(loose)] = -2.0
    vals = np.max(D, axis=1)
    order = np.argsort(vals)
    picked = []
    for j in order:
        if len(picked) >= len(loose):
            break
        r = R[j]
        ok = True
        for q in picked:
            if float(r @ R[q]) > sep:
                ok = False
                break
        if ok:
            picked.append(int(j))
        if len(picked) + (len(order) - int(np.searchsorted(order, j)) - 1) < len(loose):
            pass
    # fallback fill if too few due to sep
    if len(picked) < len(loose):
        for j in order:
            if int(j) not in picked:
                picked.append(int(j))
            if len(picked) >= len(loose):
                break
    return R[np.array(picked)]

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.default_rng(20261101)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=np.float64)
            if a.shape == (189, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
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
    # trial budget: keep each trial ~2-4s
    while time.time() < deadline - 14.0 and trial < 35:
        mode = trial % 4
        x = best_x.copy()
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        pt_max = np.max(G0, axis=1)
        order = np.argsort(pt_max)  # loose first
        # contact counts approx within 1e-7 of max for hub detection
        m0 = float(np.max(G0))
        deg = np.sum(G0 > m0 - 1e-7, axis=1)
        hub_order = np.argsort(deg)  # overloaded last
        if mode == 0:
            k = 6 + (trial % 3)  # 6..8
            loose = order[:k].tolist()
            holes = _diverse_holes(x, loose, rng, M=6000, sep=0.93)
            for ii, idx in enumerate(loose):
                x[idx] = holes[ii]
            # kick overloaded hubs slightly
            hubs = hub_order[-4:]
            x[hubs] = x[hubs] + 0.006 * rng.standard_normal((len(hubs), d))
            x[hubs] = x[hubs] / np.linalg.norm(x[hubs], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 350, 30.0, 220.0, 0.22, 0.02
            lp_budget = 2.2
        elif mode == 1:
            # shake hubs + random
            hubs = hub_order[-6:]
            rnd = rng.choice(n, size=6, replace=False)
            sel = np.unique(np.concatenate([hubs, rnd]))
            sigma = 0.012 + 0.005 * (trial % 3)
            x[sel] = x[sel] + sigma * rng.standard_normal((len(sel), d))
            x[sel] = x[sel] / np.linalg.norm(x[sel], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 400, 25.0, 180.0, 0.28, 0.02
            lp_budget = 2.2
        elif mode == 2:
            sigma = 0.004 + 0.0015 * (trial % 4)
            x = x + sigma * rng.standard_normal(x.shape)
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 300, 35.0, 250.0, 0.18, 0.015
            lp_budget = 1.8
        else:
            # break top contacts: kick one endpoint of worst pairs apart
            iuI, iuJ = np.triu_indices(n, 1)
            c = G0[iuI, iuJ]
            top = np.argsort(c)[-6:]
            kick = np.unique(np.concatenate([iuI[top][::2], iuJ[top][1::2]]))
            # push along away-from-partner direction approximated by noise orthogonal
            x[kick] = x[kick] + 0.015 * rng.standard_normal((len(kick), d))
            x[kick] = x[kick] / np.linalg.norm(x[kick], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 380, 28.0, 200.0, 0.25, 0.02
            lp_budget = 2.0
        # time guard: shrink work if low time
        tleft = deadline - 14.0 - time.time()
        if tleft < 6.0:
            steps = min(steps, 180)
            lp_budget = min(lp_budget, 1.0)
        x = _soft_anneal(x, steps, a0, a1, l0, l1, deadline=min(deadline - 9.0, time.time() + 3.0))
        lp_dead = min(time.time() + lp_budget, deadline - 9.0)
        if lp_dead > time.time() + 0.4:
            x, v = _lp_polish_cap(x, lp_dead, radius_init=1e-3, max_iter=22)
        else:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            v = _max_cos(x)
        if v < best_val - 1e-12 and np.all(np.isfinite(x)):
            best_val, best_x = v, x.copy()
        trial += 1
    # final squeeze
    if time.time() < deadline - 4.0:
        fd = min(deadline - 1.5, time.time() + 7.0)
        best_x, v2 = _lp_polish_cap(best_x, fd, radius_init=4e-4, max_iter=30)
        best_val = min(best_val, v2)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

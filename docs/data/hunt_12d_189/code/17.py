import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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

def _greedy_descent(x, steps, lr0, lr1, delta0=0.01):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        delta = delta0 * (1.0 - f) + 1e-4 * f
        # active edges
        c = G[iu]
        sel = c > m - delta
        I = iu[0][sel]; J = iu[1][sel]
        if len(I) == 0:
            continue
        cij = c[sel]
        # tangential push: dx_i -= (x_j - c*x_i), dx_j -= (x_i - c*x_j)
        F = np.zeros_like(x)
        # accumulate (vectorized via bincount-like add)
        np.add.at(F, I, -(x[J] - cij[:, None] * x[I]))
        np.add.at(F, J, -(x[I] - cij[:, None] * x[J]))
        # project to tangent (already tangent by construction) and normalize step
        # limit per-row norm
        nr = np.linalg.norm(F, axis=1, keepdims=True) + 1e-30
        F = F / np.maximum(1.0, nr / 5.0)
        x = x + lr * F
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
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
        try:
            a = np.array(c, dtype=np.float64)
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
    n, d = 189, 12

    # hopping phase until reserve for final polish
    final_reserve = 25.0
    trial = 0
    while time.time() < deadline - final_reserve and trial < 40:
        trial += 1
        x = best_x.copy()
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        ptm = np.max(G0, axis=1)
        order = np.argsort(ptm)  # loose first
        # contacts count approx with 1e-9
        m0 = float(np.max(G0))
        cnt = np.sum(G0 > m0 - 1e-9, axis=1)
        ratt = np.where(cnt == 0)[0]
        if len(ratt) == 0:
            ratt = order[:3]
        # choose k points: mostly rattlers/loose
        if trial % 2 == 0:
            k = min(5, len(ratt) + 2)
            loose = list(ratt[:min(len(ratt), k)])
            need = k - len(loose)
            if need > 0:
                loose += list(order[:need])
            loose = np.array(loose[:k], dtype=int)
        else:
            k = 3 + (trial % 3)
            loose = order[:k].copy()
            # also include a random hub to relieve overload
            hub = order[-(trial % 4) - 1]
            loose = np.unique(np.append(loose, hub))

        M = 4000
        R = rng.standard_normal((M, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        # sequential hole fill for loose points
        for idx in loose:
            D = R @ x.T
            D[:, idx] = -2.0
            vals = np.max(D, axis=1)
            j = int(np.argmin(vals))
            x[idx] = R[j]
            # small extra random candidates around the hole for next idx
            if M > 1000 and (trial % 3 == 0):
                pass
        # small kick to tight hubs
        tight = order[-3:]
        x[tight] += 0.005 * rng.standard_normal((len(tight), d))
        x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)

        # cheap descent
        x = _greedy_descent(x, steps=250, lr0=0.02, lr1=0.002, delta0=0.005 + 0.002 * (trial % 3))
        v = _max_cos(x)
        # only LP-polish promising ones
        if v < best_val + 0.003 and time.time() < deadline - final_reserve - 2.0:
            xp, vp = _lp_polish(x, min(deadline - final_reserve, time.time() + 2.5), r0=2e-3)
            if vp < best_val - 1e-12:
                best_val, best_x = vp, xp.copy()
                # shrink reserve slightly since we improved
                continue
            # also keep unpolished if it is very good for diversity? no
        # occasional keep of near-best to diversify base
        if v < best_val + 0.0005 and rng.random() < 0.1:
            # quick polish with tiny budget to see basin
            xp, vp = _lp_polish(x, min(deadline - final_reserve, time.time() + 1.0), r0=1e-3)
            if vp < best_val - 1e-12:
                best_val, best_x = vp, xp.copy()
        # time check
        if time.time() >= deadline - final_reserve:
            break

    # final polish of global best with remaining time
    best_x, best_val = _lp_polish(best_x, deadline, r0=1e-3)
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    # tiny shrink guard against rounding pushing over max (no-op, ensures valid)
    return best_x

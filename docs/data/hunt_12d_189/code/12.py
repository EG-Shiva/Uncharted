import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    iuA, iuB = iu0
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iuA, iuB]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c_all > thresh)[0]
        if idx.size == 0:
            radius *= 0.5
            continue
        if idx.size > 4000:
            part = np.argpartition(c_all[idx], -4000)[-4000:]
            idx = idx[part]
        I = iuA[idx]; J = iuB[idx]
        cs = c_all[idx]
        p = idx.size
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
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
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
        Gt = Gt / np.maximum(1.0, nr)
        x = x - lr * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_vec(x, exclude, M, rng):
    d = x.shape[1]
    R = rng.standard_normal((M, d))
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ x.T
    if exclude is not None:
        D[:, exclude] = -2.0
    vals = np.max(D, axis=1)
    j = int(np.argmin(vals))
    return R[j]

def solve():
    t0 = time.time()
    hard_end = t0 + 175.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.array(a, dtype=np.float64)
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
    # elite pool
    elites = [(best_val, best_x.copy())]
    if len(cands) > 1:
        elites.append((cands[1][0], cands[1][1].copy()))
        elites.sort(key=lambda t: t[0])

    trial = 0
    n = 189
    while time.time() < hard_end - 13.0 and trial < 200:
        mode = trial % 4
        # pick base: mostly best, sometimes second elite
        if len(elites) > 1 and (trial % 7 == 6):
            base = elites[1][1]
        else:
            base = best_x
        x = base.copy()
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        pt_max = np.max(G0, axis=1)
        cnt = np.sum(G0 > m0 - 1e-7, axis=1)
        order_loose = np.argsort(pt_max)
        order_hub = np.argsort(-cnt)
        if mode == 0:
            k = 6 + (trial % 4)
            loose = order_loose[:k]
            for ii in loose:
                x[ii] = _hole_vec(x, int(ii), 8000, rng)
            tight = order_hub[:4]
            x[tight] = x[tight] + 0.008 * rng.standard_normal((len(tight), 12))
            x[tight] = x[tight] / np.linalg.norm(x[tight], axis=1, keepdims=True)
            x = _anneal(x, 500, 30.0, 250.0, 0.25, 0.03)
            budget = 3.0
        elif mode == 1:
            k = 5 + (trial % 4)
            hubs = order_hub[:k]
            for ii in hubs:
                x[int(ii)] = _hole_vec(x, int(ii), 8000, rng)
            # small shake to neighbours
            x = x + 0.004 * rng.standard_normal(x.shape)
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 600, 25.0, 300.0, 0.28, 0.025)
            budget = 3.0
        elif mode == 2:
            sig = 0.015 + 0.005 * (trial % 5)
            x = x + sig * rng.standard_normal(x.shape)
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 700, 20.0, 300.0, 0.30, 0.02)
            budget = 2.5
        else:
            # break random tight edges
            flat_thresh = m0 - 2e-7
            iuA, iuB = np.triu_indices(n, 1)
            c_all = (base @ base.T)[iuA, iuB]
            tight_idx = np.where(c_all > flat_thresh)[0]
            if tight_idx.size > 0:
                sel_n = min(15, tight_idx.size)
                pick = rng.choice(tight_idx, size=sel_n, replace=False)
                for pp in pick:
                    i = int(iuA[pp]); j = int(iuB[pp])
                    dvec = x[i] - x[j]
                    x[i] = x[i] + 0.02 * dvec + 0.005 * rng.standard_normal(12)
                    x[j] = x[j] - 0.02 * dvec + 0.005 * rng.standard_normal(12)
                x = x / np.linalg.norm(x, axis=1, keepdims=True)
            else:
                x = x + 0.01 * rng.standard_normal(x.shape)
                x = x / np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 400, 30.0, 220.0, 0.22, 0.03)
            budget = 2.5
        dl = min(time.time() + budget, hard_end - 12.0)
        if dl - time.time() < 0.4:
            break
        xp, vp = _lp_polish(x, dl, r0=2.5e-3)
        # update elites
        if vp < best_val - 5e-13:
            best_val, best_x = vp, xp.copy()
            elites.insert(0, (vp, xp.copy()))
            elites = sorted(elites, key=lambda t: t[0])[:3]
        elif vp < elites[-1][0] - 1e-12 and len(elites) < 3:
            elites.append((vp, xp.copy()))
            elites = sorted(elites, key=lambda t: t[0])[:3]
        elif vp < elites[-1][0] and len(elites) == 3 and rng.random() < 0.2:
            # keep diversity occasionally
            elites[-1] = (vp, xp.copy())
            elites = sorted(elites, key=lambda t: t[0])
        trial += 1

    # final squeeze on best
    try:
        fp_dead = min(hard_end - 1.0, time.time() + max(4.0, hard_end - 1.0 - time.time()))
        if time.time() < fp_dead - 0.5:
            best_x, best_val = _lp_polish(best_x, fp_dead, r0=8e-4)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

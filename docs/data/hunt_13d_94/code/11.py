import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(x, steps, k0, k1, lr0, mu, rng=None, jitter=0.0):
    n, d = x.shape
    if jitter > 0 and rng is not None:
        x = x + jitter * rng.standard_normal(size=x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        m = float(np.max(G - np.eye(n) * 2.0))
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            break
        Grad = E @ x / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _reinsert_one(others, rng, n_cand=3000, steps=120):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = cand @ others.T
    best = int(np.argmin(sc.max(axis=1)))
    c = cand[best].copy()
    for it in range(steps):
        dots = others @ c
        m = float(dots.max())
        w = np.exp(40.0 * (dots - m))
        w /= w.sum()
        g = w @ others
        gt = g - (g @ c) * c
        st = 0.15 * (1.0 - 0.5 * it / steps)
        c = c - st * gt
        nc = float(np.linalg.norm(c))
        if nc == 0 or not np.isfinite(nc):
            break
        c /= nc
    return c

def _cluster_pick(x, k, rng):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - 5e-4).sum(axis=1)
    # seed among most jammed
    top = np.argsort(-cnt)[:8]
    seed = int(top[rng.integers(0, len(top))])
    chosen = [seed]
    chosen_set = {seed}
    # greedy expansion by contact strength to cluster
    while len(chosen) < k:
        # score each outside by sum of G to chosen
        scores = G[:, chosen].sum(axis=1)
        scores[list(chosen_set)] = -1e99
        # add randomness among top candidates
        order = np.argsort(-scores)
        # pick among top 5
        cand = [int(o) for o in order[:5] if o not in chosen_set]
        if not cand:
            # fill random
            rem = [i for i in range(n) if i not in chosen_set]
            chosen.append(int(rem[rng.integers(0, len(rem))]))
        else:
            chosen.append(int(cand[rng.integers(0, len(cand))]))
        chosen_set.add(chosen[-1])
    return chosen

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 168.0
    HOP_END = T0 + 122.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xb_n = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb_n):
            x_best = xr.copy()
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    cur_x = x_best.copy()
    cur_val = best_val
    n, d = x_best.shape

    def scores_deg(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        s = G.max(axis=1)
        deg = (G > mx - 1e-3).sum(axis=1)
        deg2 = (G > mx - 5e-4).sum(axis=1)
        return s, deg, deg2, G, mx

    it = 0
    while time.time() < HOP_END and time.time() - T0 < BUDGET - 55:
        it += 1
        s, deg, deg2, Gcur, mxcur = scores_deg(cur_x)
        y = cur_x.copy()
        mode = it % 8
        if mode == 0:
            # worst single reinsert
            order = np.argsort(-s)
            idx = int(order[0])
            # second worst sometimes
            idxs = [idx]
            if rng.random() < 0.5:
                idxs.append(int(order[1]))
                idxs = list(dict.fromkeys(idxs))
            for idx in idxs:
                others = np.delete(y, idx, axis=0)
                y[idx] = _reinsert_one(others, rng, n_cand=3000, steps=120)
            y = _smooth(y, steps=550, k0=25, k1=320, lr0=0.25, mu=0.85, rng=rng, jitter=0.003)
            y = _smooth(y, steps=150, k0=300, k1=600, lr0=0.08, mu=0.9)
        elif mode == 1:
            k = int(rng.integers(3, 6))
            cl = _cluster_pick(y, k, rng)
            y[cl] = rng.standard_normal(size=(k, d))
            y[cl] /= np.linalg.norm(y[cl], axis=1, keepdims=True)
            y = _smooth(y, steps=600, k0=20, k1=300, lr0=0.28, mu=0.87, rng=rng, jitter=0.002)
            y = _smooth(y, steps=150, k0=300, k1=600, lr0=0.08, mu=0.9)
        elif mode == 2:
            k = int(rng.integers(6, 10))
            cl = _cluster_pick(y, k, rng)
            # farthest-point init for hole
            newp = rng.standard_normal(size=(k, d))
            newp /= np.linalg.norm(newp, axis=1, keepdims=True)
            y[cl] = newp
            y = _smooth(y, steps=700, k0=18, k1=280, lr0=0.3, mu=0.88, rng=rng)
            y = _smooth(y, steps=150, k0=300, k1=600, lr0=0.08, mu=0.9)
        elif mode == 3:
            sigma = float(rng.choice([0.06, 0.09, 0.12, 0.16]))
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=700, k0=20, k1=350, lr0=0.3, mu=0.9, rng=rng)
        elif mode == 4:
            # shake jammed + loose together
            lo = np.argsort(deg)[:8]
            hi = np.argsort(-deg2)[:8]
            worst = np.argsort(-s)[:4]
            idx = np.unique(np.concatenate([lo, hi, worst]))
            y[idx] += 0.25 * rng.standard_normal(size=(len(idx), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=650, k0=25, k1=320, lr0=0.24, mu=0.88, rng=rng)
        elif mode == 5:
            # contact-breaking: push top pairs apart
            iu = np.triu_indices(n, 1)
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            flat = G[iu]
            topk = np.argsort(-flat)[:6]
            for t in topk:
                a = int(iu[0][t]); b = int(iu[1][t])
                # tangential repulsion
                ya = y[a].copy(); yb = y[b].copy()
                # remove components along self
                pa = yb - (yb @ ya) * ya
                pb = ya - (ya @ yb) * yb
                na = float(np.linalg.norm(pa)); nb = float(np.linalg.norm(pb))
                if na > 1e-12:
                    y[a] = y[a] - 0.08 * pa / na
                if nb > 1e-12:
                    y[b] = y[b] - 0.08 * pb / nb
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=600, k0=30, k1=350, lr0=0.22, mu=0.87, rng=rng, jitter=0.004)
        elif mode == 6:
            # loose-focused reinsert (near-rattlers)
            lo = np.argsort(deg)[:4]
            for idx in lo:
                if rng.random() < 0.7:
                    others = np.delete(y, int(idx), axis=0)
                    y[int(idx)] = _reinsert_one(others, rng, n_cand=4000, steps=150)
                else:
                    y[int(idx)] += 0.2 * rng.standard_normal(size=(d,))
                    y[int(idx)] /= np.linalg.norm(y[int(idx)])
            y = _smooth(y, steps=550, k0=25, k1=330, lr0=0.24, mu=0.86, rng=rng, jitter=0.002)
        else:
            # exploitation around best
            base = x_best if rng.random() < 0.7 else cur_x
            y = base.copy()
            kpick = int(rng.integers(4, 12))
            idx = rng.choice(n, size=kpick, replace=False)
            y[idx] += 0.05 * rng.standard_normal(size=(kpick, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=500, k0=40, k1=450, lr0=0.15, mu=0.9, rng=rng)

        new_val = _max_cos(y)
        if new_val < best_val - 1e-15:
            x_best = y.copy()
            best_val = new_val
            cur_x = y.copy()
            cur_val = new_val
        else:
            elapsed = time.time() - T0
            T = 0.002 * max(0.08, 1.0 - elapsed / max(1.0, HOP_END - T0))
            T = max(T, 0.00015)
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
                cur_x = y.copy()
                cur_val = new_val

    # final tighten + LP polish of best
    remain_deadline = T0 + BUDGET - 2.0
    try:
        x_best = _smooth(x_best, steps=300, k0=80, k1=700, lr0=0.07, mu=0.9)
        nv = _max_cos(x_best)
        if nv < best_val:
            best_val = nv
    except Exception:
        pass
    try:
        x_pol = _lp_polish(x_best, deadline=remain_deadline)
        pv = _max_cos(x_pol)
        if pv < best_val and np.all(np.isfinite(x_pol)):
            x_best = x_pol
            best_val = pv
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0) + 1e-12:
            # keep better one
            if _max_cos(x0) < _max_cos(x_best):
                x_best = x0
    except Exception:
        pass
    return x_best

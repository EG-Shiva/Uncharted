import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thresh=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    iu = np.triu_indices(x.shape[0], 1)
    sel = g[iu] > m - thresh
    return m, iu[0][sel], iu[1][sel]

def _smooth_refine(x, steps, k0, k1, lr0, mu, rng, jitter=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    # optional tiny jitter at start
    if jitter > 0:
        x = x + jitter * rng.standard_normal(size=x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        # max off diag
        # diag of G is 1, subtract 2 on diag to ignore
        m = np.max(G - np.eye(n) * 2.0)
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Grad = E @ x / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def _reinsert_point(others, rng, n_cand=5000, steps=200):
    n_o, d = others.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    c = cand[np.argmin(scores)].copy()
    # local tangential refinement to minimize smooth max
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        kk = 40.0
        w = np.exp(kk * (dots - m))
        w /= w.sum()
        grad = w @ others  # (d,)
        gt = grad - (grad @ c) * c
        # decaying step
        step = 0.15 * (1.0 - 0.5 * it / steps)
        c = c - step * gt
        nc = np.linalg.norm(c)
        if nc == 0 or not np.isfinite(nc):
            break
        c /= nc
    return c

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
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
    rng = np.random.default_rng(0)
    np.random.seed(0)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    try:
        x_rec = np.load("record.npy").astype(float)
        x_rec /= np.linalg.norm(x_rec, axis=1, keepdims=True)
        if _max_cos(x_rec) < _max_cos(x_best / np.linalg.norm(x_best, axis=1, keepdims=True)):
            x_best = x_rec.copy()
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    cur_x = x_best.copy()
    cur_val = best_val

    # quick contact info for targeted moves
    def point_scores(x):
        G = x @ x.T
        np.fill_diagonal(G, -2)
        s = G.max(axis=1)  # worst neighbor per point
        deg = (G > G.max() - 1e-3).sum(axis=1)
        return s, deg

    it = 0
    # hopping phase until ~115s
    HOP_END = T0 + 112.0
    while time.time() - T0 < HOP_END and time.time() - T0 < BUDGET - 55:
        it += 1
        s, deg = point_scores(cur_x)
        y = cur_x.copy()
        mode = it % 6
        if mode == 0 or mode == 3:
            # re-insert 1-2 worst / low-degree points
            n, d = y.shape
            # worst points: largest s; also include smallest degree (loose)
            order_worst = np.argsort(-s)  # descending
            order_loose = np.argsort(deg)  # ascending
            cand_idx = []
            cand_idx.append(int(order_worst[0]))
            if n > 1:
                # alternate second pick
                if mode == 0:
                    cand_idx.append(int(order_worst[1]))
                else:
                    cand_idx.append(int(order_loose[0]))
            cand_idx = list(dict.fromkeys(cand_idx))
            for idx in cand_idx:
                others = np.delete(y, idx, axis=0)
                c_new = _reinsert_point(others, rng, n_cand=4000, steps=150)
                y[idx] = c_new
            # smooth refine after reinsert
            y = _smooth_refine(y, steps=600, k0=25, k1=300, lr0=0.25, mu=0.85, rng=rng, jitter=0.005)
        elif mode == 1:
            # strong global shake to new basin
            sigma = 0.12
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=800, k0=20, k1=350, lr0=0.3, mu=0.9, rng=rng)
        elif mode == 2:
            # targeted shake of loose + overloaded points
            idx_loose = np.argsort(deg)[:6]
            idx_over = np.argsort(-deg)[:6]
            idx = np.unique(np.concatenate([idx_loose, idx_over, np.argsort(-s)[:4]]))
            noise = rng.standard_normal(size=(len(idx), y.shape[1])) * 0.25
            y[idx] += noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=700, k0=30, k1=300, lr0=0.2, mu=0.88, rng=rng)
        elif mode == 4:
            # small shake of random subset for exploitation
            kpick = 8
            idx = rng.choice(y.shape[0], size=kpick, replace=False)
            y[idx] += 0.04 * rng.standard_normal(size=(kpick, y.shape[1]))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=500, k0=40, k1=400, lr0=0.15, mu=0.9, rng=rng)
        else:
            # shake around best with medium noise
            sigma = 0.05 + 0.05 * rng.random()
            base = x_best if rng.random() < 0.7 else cur_x
            y = base + sigma * rng.standard_normal(size=base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=600, k0=25, k1=320, lr0=0.22, mu=0.87, rng=rng)

        new_val = _max_cos(y)
        if new_val < best_val - 1e-15:
            x_best = y.copy()
            best_val = new_val
            cur_x = y.copy()
            cur_val = new_val
        else:
            # simulated-annealing acceptance for exploration
            T = 0.002 * (1.0 - (time.time() - T0) / (HOP_END - T0 + 1e-9))
            T = max(T, 0.0002)
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
                cur_x = y.copy()
                cur_val = new_val

    # final polish of best with remaining time
    remain = BUDGET - (time.time() - T0) - 2.0
    if remain > 5:
        # short smooth high-k tightening before LP
        try:
            x_best = _smooth_refine(x_best, steps=300, k0=80, k1=600, lr0=0.08, mu=0.9, rng=rng)
            nv = _max_cos(x_best)
            if nv < best_val:
                best_val = nv
        except Exception:
            pass
        try:
            x_pol = _lp_polish(x_best, seconds=remain, t0=time.time())
            pv = _max_cos(x_pol)
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol
                best_val = pv
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: if somehow worse than input, return input best
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0):
            # keep our best only if better, else return ours anyway if within tiny tol?
            # return the better one to guarantee no regression
            x_best = x0 if _max_cos(x0) < _max_cos(x_best) else x_best
    except Exception:
        pass
    return x_best

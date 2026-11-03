import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal(y, steps, T0, T1, lr):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # normalize step by mean norm to keep stable
        y = y - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
    return y


def _deepest_hole(Ykeep, rng, n_cand=2000):
    d = Ykeep.shape[1]
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ Ykeep.T).max(axis=1)
    return cand[int(np.argmin(scores))]


def _lp_polish(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = int(sel.sum())
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    HOP_END = t0 + 132.0
    n, d = 123, 13
    rng = np.random.default_rng(0)
    np.random.seed(0)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    cur = best.copy()
    cur_val = best_val
    it = 0
    while time.time() < HOP_END:
        remain = HOP_END - time.time()
        if remain < 0.5:
            break
        mode = it % 5
        y = cur.copy()
        try:
            if mode == 0:
                # subset shake
                k = 6
                idx = rng.choice(n, size=k, replace=False)
                y[idx] += 0.18 * rng.normal(size=(k, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, 220, 0.02, 0.002, 0.6)
            elif mode == 1:
                # global small shake
                y = y + 0.05 * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, 220, 0.015, 0.0015, 0.5)
            elif mode == 2:
                # worst-hub deletion + hole refill
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                worst = np.max(G, axis=1)
                nh = 3 + (it % 3)  # 3-5
                hubs = np.argsort(-worst)[:nh]
                mask = np.ones(n, dtype=bool)
                mask[hubs] = False
                Ykeep = y[mask]
                for h in hubs:
                    nhv = _deepest_hole(Ykeep, rng, n_cand=1500)
                    # slight random tilt to break symmetry
                    nhv = nhv + 0.01 * rng.normal(size=d)
                    nhv /= np.linalg.norm(nhv)
                    y[h] = nhv
                    Ykeep = y[mask | (np.arange(n) == h)]
                    # update keep for next hole (approx: use full set minus remaining hubs)
                y = _anneal(y, 260, 0.025, 0.002, 0.6)
            elif mode == 3:
                # fresh random + long anneal (different basin)
                y = rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, 350, 0.03, 0.002, 0.7)
                # blend a bit toward best to keep quality? no, pure random
            else:
                # rattler/low-contact move to hole
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 0.005).sum(axis=1)
                # pick lowest-degree points
                nl = 2
                lows = np.argsort(deg)[:nl]
                mask = np.ones(n, dtype=bool)
                mask[lows] = False
                Ykeep = y[mask]
                for h in lows:
                    nhv = _deepest_hole(Ykeep, rng, n_cand=1500)
                    y[h] = nhv / np.linalg.norm(nhv)
                y = _anneal(y, 240, 0.02, 0.0015, 0.55)
            v = _max_cos(y)
            if v < best_val:
                best_val = v
                best = y.copy()
                cur = y.copy()
                cur_val = v
            else:
                # random-walk acceptance: allow uphill within band
                if v < cur_val or (v < cur_val + 0.015 and rng.random() < 0.25):
                    cur = y.copy()
                    cur_val = v
                else:
                    # occasionally restart from best with shake
                    if rng.random() < 0.25:
                        cur = best.copy()
                        cur = cur + 0.08 * rng.normal(size=(n, d))
                        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                        cur_val = _max_cos(cur)
        except Exception:
            pass
        it += 1
        # safety: if very close to hard end, break
        if time.time() > HARD_END - 40:
            break

    # final snap: LP polish best with remaining time (capped)
    try:
        t_rem = HARD_END - time.time() - 1.0
        if t_rem > 5:
            t_pol = min(t_rem, 38.0)
            cand = _lp_polish(best, t_pol)
            v = _max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # tiny shrink guard not needed; return as-is
    return best

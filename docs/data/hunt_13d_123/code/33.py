import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = (g > m - thr).sum(axis=1)
    return m, cnt, g

def _smooth_run(y, steps, T0, T1, lr0, lr1):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # normalize gradient step size
        gn = np.linalg.norm(Gt)
        if gn > 1e-12:
            y = y - lr * Gt
            y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _lp_polish(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 8e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20 * radius, 1e-9)
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
            cnt = p_cap
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = cnt
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(0)
    n, d = 123, 13
    X0 = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val = _max_cos(X0)
    best = X0.copy()
    cur = X0.copy()
    cur_val = float(best_val)

    # contact info for targeted kicks
    try:
        _, cnt0, _ = _contacts(best)
    except Exception:
        cnt0 = np.zeros(n)

    it = 0
    # reserve 12s for final LP
    while time.time() - t0 < LIMIT - 14:
        it += 1
        # choose kick size: cycle small/medium/large
        cyc = it % 6
        if cyc == 0:
            k = 3; sigma = 0.015
        elif cyc == 1:
            k = 6; sigma = 0.03
        elif cyc == 2:
            k = 12; sigma = 0.06
        elif cyc == 3:
            k = 4; sigma = 0.10
        elif cyc == 4:
            k = 20; sigma = 0.05
        else:
            k = n; sigma = 0.008  # global micro shake
        y = cur.copy()
        # pick victims: worst by max-pair + contacts + random
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)  # worst first
        nworst = min(k // 2 + 1, n)
        victims = set(order[:nworst].tolist())
        # add overloaded hubs (high contacts) periodically
        if it % 3 == 0:
            try:
                _, cnt, _ = _contacts(y)
                hubs = np.argsort(-cnt)[:3]
                for h in hubs:
                    victims.add(int(h))
            except Exception:
                pass
        while len(victims) < k:
            victims.add(int(rng.integers(0, n)))
        victims = np.array(sorted(victims), dtype=int)
        # special: occasionally reinsert one victim into deepest hole
        if it % 4 == 0 and len(victims) > 0:
            # deepest hole: random candidate search maximizing min distance
            cand = rng.normal(size=(600, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # min cosine to y (excluding one victim to be replaced?)
            C = cand @ y.T  # 600 x 123
            # hole score = max cosine (want smallest)
            hole = C.max(axis=1)
            bi = int(np.argmin(hole))
            y[victims[0]] = cand[bi]
            if len(victims) > 1:
                noise = rng.normal(size=(len(victims) - 1, d)) * sigma
                y[victims[1:]] += noise
                y[victims[1:]] /= np.linalg.norm(y[victims[1:]], axis=1, keepdims=True)
        else:
            y[victims] += rng.normal(size=(len(victims), d)) * sigma
            y[victims] /= np.linalg.norm(y[victims], axis=1, keepdims=True)
        # anneal params vary per cycle
        if cyc in (0, 5):
            steps, T0a, T1a, lr0, lr1 = 220, 0.015, 0.0015, 0.06, 0.01
        elif cyc == 1:
            steps, T0a, T1a, lr0, lr1 = 260, 0.02, 0.002, 0.08, 0.015
        else:
            steps, T0a, T1a, lr0, lr1 = 180, 0.025, 0.002, 0.09, 0.015
        # two-stage: coarse then fine within same budget
        y = _smooth_run(y, steps, T0a, T1a, lr0, lr1)
        v = _max_cos(y)
        if v < cur_val and time.time() - t0 < LIMIT - 14:
            # short greedy hill-climb on worst point
            cur = y
            cur_val = v
            if v < best_val:
                best_val = v
                best = y.copy()
        elif v < best_val + 0.004 and rng.random() < 0.08:
            # sideways move to new basin
            cur = y
            cur_val = v
        # time guard
        if time.time() - t0 >= LIMIT - 14:
            break
        # restart from best if stuck long
        if it % 40 == 0 and cur_val > best_val + 1e-6:
            cur = best.copy()
            cur_val = float(best_val)

    # final squeeze: short LP on best (bounded time)
    remain = LIMIT - (time.time() - t0)
    if remain > 4:
        try:
            lp_time = min(remain - 1.0, 11.0)
            cand = _lp_polish(best, lp_time)
            v = _max_cos(cand)
            if v < best_val and np.all(np.isfinite(cand)):
                best_val = v
                best = cand
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

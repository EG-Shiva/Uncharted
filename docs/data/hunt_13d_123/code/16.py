import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _smooth_anneal(y, steps, T0, T1, lr, mu=0.6):
    n, d = y.shape
    vel = np.zeros_like(y)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        # geometric T schedule
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-30
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # normalize row-wise to stabilize
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # clip huge gradients
        Gt = Gt / np.maximum(gn, 0.5)
        vel = mu * vel + (1.0 - mu) * Gt
        y = y - lr * vel
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _deep_hole(X_keep, rng, n_cand=4000):
    n, d = X_keep.shape
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ X_keep.T
    worst = S.max(axis=1)
    return C[int(np.argmin(worst))]

def _lp_polish(x, seconds, p_cap=2000):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 8e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 2e-9)
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
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(12345)
    n, d = 123, 13
    X0 = None
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

    it = 0
    # basin hopping with cheap gradient annealing; reserve ~28s for LP
    while time.time() - t0 < LIMIT - 28:
        it += 1
        ppm = _per_point_max(cur)
        order_lo = np.argsort(ppm)   # smallest = loosest
        order_hi = np.argsort(-ppm)  # largest = most jammed
        y = cur.copy()
        r = rng.random()
        if r < 0.45:
            # hole reinsertion of loosest point
            loose = int(order_lo[0])
            mask = np.ones(n, dtype=bool); mask[loose] = False
            hole = _deep_hole(cur[mask], rng, n_cand=4000)
            y[loose] = hole
            # small shake of 2-4 hubs
            k = int(rng.integers(2, 5))
            hubs = order_hi[:12]
            ch = hubs[rng.choice(len(hubs), size=min(k, len(hubs)), replace=False)]
            y[ch] += rng.normal(scale=0.06, size=(len(ch), d))
            y[ch] /= np.linalg.norm(y[ch], axis=1, keepdims=True)
        elif r < 0.75:
            # kick hubs moderately
            k = int(rng.integers(3, 9))
            # bias to high-contact points + random
            cand_hi = order_hi[:20]
            sel = list(cand_hi[rng.choice(len(cand_hi), size=min(k - 1, len(cand_hi)), replace=False)])
            sel += [int(rng.integers(0, n))]
            sel = np.array(sel, dtype=int)
            scale = float(rng.uniform(0.05, 0.22))
            y[sel] += rng.normal(scale=scale, size=(len(sel), d))
            y[sel] /= np.linalg.norm(y[sel], axis=1, keepdims=True)
        else:
            # global small shake to change topology
            scale = float(rng.uniform(0.01, 0.06))
            k = int(rng.integers(8, 24))
            sel = rng.choice(n, size=k, replace=False)
            y[sel] += rng.normal(scale=scale, size=(k, d))
            y[sel] /= np.linalg.norm(y[sel], axis=1, keepdims=True)

        # anneal params varied per basin
        T0 = float(rng.choice([0.012, 0.008, 0.005]))
        T1 = float(rng.choice([0.0004, 0.0002, 0.0001]))
        lr = float(rng.uniform(0.015, 0.045))
        steps = int(rng.choice([400, 600, 900]))
        # cap steps if little time
        if time.time() - t0 > LIMIT - 60:
            steps = min(steps, 400)
        try:
            y = _smooth_anneal(y, steps, T0, T1, lr)
        except Exception:
            continue
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
        elif v < cur_val + 0.003 and rng.random() < 0.35:
            # random-walk acceptance to new basin
            cur = y.copy()
            cur_val = v
        elif v < cur_val:
            cur = y.copy()
            cur_val = v
        # opportunistic restart from best occasionally
        if it % 25 == 0 and rng.random() < 0.5:
            cur = best.copy()
            cur_val = float(best_val)
        if best_val < 0.2715:
            break

    # final tighten: short gradient cool + LP
    try:
        remain = LIMIT - (time.time() - t0)
        if remain > 6:
            # cool down best with tiny T
            cool_steps = 300 if remain > 15 else 120
            best = _smooth_anneal(best.copy(), cool_steps, 0.0012, 0.00008, 0.008, mu=0.5)
            bv = _max_cos(best)
            if bv < best_val:
                best_val = bv
    except Exception:
        pass
    try:
        remain = LIMIT - (time.time() - t0)
        if remain > 5:
            lp_time = min(remain - 1.0, 26.0)
            cand = _lp_polish(best, lp_time)
            v = _max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

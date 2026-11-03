import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _smooth_run(y, steps, T0, T1, lr0, lr1, mu=0.5):
    n, d = y.shape
    vel = np.zeros_like(y)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        S = E @ y
        Avg = S / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # clip extreme grads for stability
        # Gt norm can be large; scale down if needed
        vel = mu * vel - lr * Gt
        # keep velocity tangential
        vd = (vel * y).sum(axis=1, keepdims=True)
        vel = vel - vd * y
        y = y + vel
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def _refine_hole(v, X, steps=200, T=0.005, lr=0.08):
    v = v / np.linalg.norm(v)
    for _ in range(steps):
        dots = X @ v  # (m,)
        m = float(dots.max())
        w = np.exp((dots - m) / T) + 1e-30
        w /= w.sum()
        avg = w @ X  # (d,)
        dot = float(avg @ v)
        g = avg - dot * v
        v = v - lr * g
        n = np.linalg.norm(v)
        if n == 0:
            break
        v = v / n
    return v


def _deepest_hole(X_keep, rng, n_cand=6000, T=0.005):
    m, d = X_keep.shape
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_keep.T  # (n_cand, m)
    worst = dots.max(axis=1)
    idx = int(np.argmin(worst))
    v = cand[idx].copy()
    v = _refine_hole(v, X_keep, steps=200, T=T, lr=0.08)
    return v


def _replace_worst(x, k, rng):
    n, d = x.shape
    pm = _per_point_max(x)
    worst_idx = np.argsort(-pm)[:k]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[worst_idx] = False
    X_keep = x[keep_mask]  # (n-k, d)
    new_pts = []
    cur = X_keep
    for _ in range(k):
        v = _deepest_hole(cur, rng)
        new_pts.append(v)
        cur = np.vstack([cur, v[None, :]])
    y = np.vstack([X_keep, np.vstack(new_pts)])
    # shuffle to avoid ordering bias (deterministic permutation)
    perm = rng.permutation(n)
    y = y[perm]
    return y


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
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
    HARD = 168.0
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

    def remain():
        return HARD - (time.time() - t0)

    # Phase B: basin hopping with fast smooth repair only
    it = 0
    # reserve 22s for final tightening
    while remain() > 22.0:
        it += 1
        cur = best.copy()
        mode = it % 6
        try:
            if mode == 0:
                # small global shake
                cur = cur + rng.normal(scale=0.015, size=cur.shape)
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur = _smooth_run(cur, steps=700, T0=0.015, T1=0.002, lr0=0.02, lr1=0.006, mu=0.5)
            elif mode == 1:
                # medium global shake
                cur = cur + rng.normal(scale=0.05, size=cur.shape)
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur = _smooth_run(cur, steps=1100, T0=0.02, T1=0.002, lr0=0.025, lr1=0.005, mu=0.5)
            elif mode == 2:
                # hub-targeted shake
                pm = _per_point_max(cur)
                w = np.argsort(-pm)[:12]
                cur[w] += rng.normal(scale=0.12, size=(len(w), d))
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur = _smooth_run(cur, steps=1200, T0=0.02, T1=0.0015, lr0=0.03, lr1=0.005, mu=0.6)
            elif mode == 3:
                # replace 2-3 worst by holes
                k = 2 if (it % 2 == 0) else 3
                cur = _replace_worst(cur, k, rng)
                cur = _smooth_run(cur, steps=1200, T0=0.018, T1=0.0015, lr0=0.028, lr1=0.005, mu=0.6)
            elif mode == 4:
                # replace 5 worst by holes (big topology jump)
                cur = _replace_worst(cur, 5, rng)
                cur = _smooth_run(cur, steps=1500, T0=0.022, T1=0.002, lr0=0.03, lr1=0.005, mu=0.6)
            else:
                # large global shake
                cur = cur + rng.normal(scale=0.10, size=cur.shape)
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur = _smooth_run(cur, steps=1300, T0=0.025, T1=0.002, lr0=0.03, lr1=0.005, mu=0.6)
        except Exception:
            continue
        try:
            v = _max_cos(cur)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best_val = v
            best = cur.copy()
        if remain() <= 22.0:
            break
        # safety: avoid infinite loop
        if it > 400:
            break

    # final tightening: short smooth then short LP
    try:
        if remain() > 20.0:
            best = _smooth_run(best, steps=800, T0=0.008, T1=0.001, lr0=0.015, lr1=0.004, mu=0.4)
            bv = _max_cos(best)
            if bv < best_val:
                best_val = bv
    except Exception:
        pass
    try:
        if remain() > 6.0:
            sec = min(remain() - 1.5, 14.0)
            if sec > 2.0:
                cand = _lp_polish(best, seconds=sec)
                bv = _max_cos(cand)
                if bv < best_val:
                    best = cand
                    best_val = bv
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

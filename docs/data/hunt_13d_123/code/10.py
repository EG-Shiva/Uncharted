import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _contacts(x, thr=1e-9):
    G = x @ x.T
    m = G.max() - 2.0 / x.shape[0]  # dummy to avoid diag; recompute properly
    G2 = x @ x.T
    np.fill_diagonal(G2, -2.0)
    m = G2.max()
    C = G2 > m - thr
    cnt = C.sum(axis=1)
    return m, cnt, G2


def _smooth_run(y, steps, T0, T1, lr):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # normalize step to avoid blow-up
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # scale: keep magnitude information but cap
        step = Gt / np.maximum(gn.mean(), 1e-12) * 0.5
        y = y - lr * step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def _lp_polish(x, seconds, p_cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 2e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
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
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([ (I[:, None] * d + np.arange(d)).reshape(p, d),
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x


def _best_hole(X_keep, rng, n_try=3000):
    # find unit vector minimizing max dot to X_keep
    n_keep, d = X_keep.shape
    R = rng.normal(size=(n_try, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    dots = R @ X_keep.T  # n_try x n_keep
    vals = dots.max(axis=1)
    i = int(np.argmin(vals))
    return R[i]


def solve():
    t0 = time.time()
    LIMIT = 176.0
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
    # initial quick tighten
    try:
        if time.time() - t0 < LIMIT - 20:
            cand = _lp_polish(best, 1.5)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
    except Exception:
        pass

    it = 0
    while time.time() - t0 < LIMIT - 11.0:
        it += 1
        remain = LIMIT - 11.0 - (time.time() - t0)
        if remain < 1.0:
            break
        y = best.copy()
        mode = it % 6
        try:
            _, cnt, Gmat = _contacts(best, thr=1e-9)
        except Exception:
            cnt = np.ones(n)
        order = np.argsort(-cnt)
        if mode == 0:
            y = y + rng.normal(0, 0.010, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_run(y, steps=350, T0=0.012, T1=0.002, lr=0.06)
        elif mode == 1:
            k = int(6 + rng.integers(0, 6))
            idx = order[:k]
            y[idx] += rng.normal(0, 0.09, size=(k, d))
            mask = np.ones(n, dtype=bool); mask[idx] = False
            y[mask] += rng.normal(0, 0.004, size=(int(mask.sum()), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_run(y, steps=450, T0=0.015, T1=0.002, lr=0.07)
        elif mode == 2:
            # remove 1 worst, reinsert in hole
            kw = 1
            rem = order[:kw]
            keep_mask = np.ones(n, dtype=bool); keep_mask[rem] = False
            Xkeep = y[keep_mask]
            for r in rem:
                h = _best_hole(Xkeep, rng, n_try=4000)
                # local refine hole: few steps of minimizing max dot (gradient on single point)
                # simple hill climb
                for _ in range(60):
                    dots = Xkeep @ h
                    # softmax weights
                    T = 0.005
                    m = dots.max()
                    w = np.exp((dots - m) / T)
                    w /= w.sum() + 1e-20
                    grad = w @ Xkeep
                    grad = grad - (grad @ h) * h
                    h = h - 0.1 * grad
                    h /= np.linalg.norm(h)
                y[r] = h
                Xkeep = np.vstack([Xkeep, h[None, :]])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_run(y, steps=400, T0=0.010, T1=0.0015, lr=0.06)
        elif mode == 3:
            kw = int(2 + rng.integers(0, 2))  # 2-3
            rem = order[:kw]
            keep_mask = np.ones(n, dtype=bool); keep_mask[rem] = False
            Xkeep = y[keep_mask]
            for r in rem:
                h = _best_hole(Xkeep, rng, n_try=3000)
                y[r] = h
                Xkeep = np.vstack([Xkeep, h[None, :]])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_run(y, steps=500, T0=0.015, T1=0.002, lr=0.07)
        elif mode == 4:
            y = y + rng.normal(0, 0.030, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_run(y, steps=400, T0=0.020, T1=0.003, lr=0.08)
        else:
            # rattler / low-contact reinsertion + small global shake
            low = np.argsort(cnt)[:3]
            keep_mask = np.ones(n, dtype=bool); keep_mask[low[:1]] = False
            Xkeep = y[keep_mask]
            h = _best_hole(Xkeep, rng, n_try=3000)
            y[low[0]] = h
            y = y + rng.normal(0, 0.008, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_run(y, steps=350, T0=0.008, T1=0.001, lr=0.05)

        v = _max_cos(y)
        # polish if promising
        if v < best_val + 0.004 and time.time() - t0 < LIMIT - 11.0:
            # adapt polish time: shorter if far
            pt = 0.9 if v < best_val + 0.001 else 0.5
            # cap to remaining
            pt = min(pt, LIMIT - 11.0 - (time.time() - t0))
            if pt > 0.25:
                try:
                    y2 = _lp_polish(y, pt)
                    v2 = _max_cos(y2)
                except Exception:
                    y2, v2 = y, v
                if v2 < best_val - 1e-12:
                    best = y2.copy()
                    best_val = float(v2)
                elif v < best_val - 1e-12:
                    best = y.copy()
                    best_val = float(v)
        else:
            if v < best_val - 1e-12:
                best = y.copy()
                best_val = float(v)

    # final long polish
    try:
        t_rem = LIMIT - (time.time() - t0) - 0.5
        if t_rem > 1.0:
            cand = _lp_polish(best, min(t_rem, 9.0))
            v = _max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

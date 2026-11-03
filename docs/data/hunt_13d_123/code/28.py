import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=1200, max_it=6):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < max_it:
        it += 1
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
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            x = best_x.copy() if radius < 1e-7 else x
        if time.time() - start >= seconds:
            break
    return best_x

def _cheap_descent(y, steps, T0, T1, lr0, lr1):
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
        y = y - lr * Gt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    TIME_LIMIT = 165.0
    RESERVE = 10.0
    rng = _np.random.default_rng(0)
    n, d = 123, 13
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                if _np.all(_np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val0:
                        best_val0 = v
                        X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = _max_cos(X0)
    best = X0.copy()
    best_val = float(best_val0)
    cur = X0.copy()
    cur_val = float(best_val0)
    # quick initial snap (capped)
    try:
        if _time.time() - t0 < TIME_LIMIT - RESERVE - 5:
            cand = _lp_polish(best, 1.0, p_cap=1500, max_it=5)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v; best = cand.copy(); cur = cand.copy(); cur_val = v
    except Exception:
        pass
    it = 0
    while True:
        now = _time.time()
        if now - t0 > TIME_LIMIT - RESERVE:
            break
        remaining = TIME_LIMIT - RESERVE - (now - t0)
        if remaining < 2.5:
            break
        it += 1
        mode = it % 4
        trial = cur.copy()
        # diagnostics (cheap)
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        wmax = Gc.max(axis=1)
        thr = cur_val - 0.01
        counts = (Gc > thr).sum(axis=1)
        worst_idx = _np.argsort(-wmax)
        ratt_idx = _np.argsort(counts)
        if mode == 0:
            sigma = 0.012 if (it % 8 != 0) else 0.03
            trial = trial + sigma * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 1:
            targets = set()
            for k in range(min(4, n)):
                targets.add(int(worst_idx[k]))
            for k in range(min(2, n)):
                targets.add(int(ratt_idx[k]))
            for k in range(2):
                targets.add(int(rng.integers(0, n)))
            trial += 0.003 * rng.normal(size=trial.shape)
            for idx in list(targets):
                trial[idx] += 0.15 * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 2:
            # hole reinsertion of single worst/rattler
            r = int(ratt_idx[0]) if int(counts[ratt_idx[0]]) < 8 else int(worst_idx[0])
            mask = _np.ones(n, dtype=bool); mask[r] = False
            Xr = trial[mask]
            n_cand = 5000
            Y = rng.normal(size=(n_cand, d))
            Y /= _np.linalg.norm(Y, axis=1, keepdims=True)
            dots = Y @ Xr.T
            md = dots.max(axis=1)
            order = _np.argpartition(md, 4)[:4]
            pick = order[int(rng.integers(0, len(order)))]
            v = Y[pick] + 0.004 * rng.normal(size=(d,))
            v /= _np.linalg.norm(v)
            trial[r] = v
            trial += 0.002 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # neighbourhood repack around worst point: kick its close neighbours
            w = int(worst_idx[0])
            gw = (cur @ cur[w])
            gw[w] = -2
            nbrs = _np.argsort(-gw)[:8]
            trial += 0.003 * rng.normal(size=trial.shape)
            trial[w] += 0.15 * rng.normal(size=(d,))
            for idx in nbrs:
                trial[int(idx)] += 0.06 * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        # cheap descent: 120-180 steps
        try:
            if it % 3 == 0:
                trial = _cheap_descent(trial, steps=150, T0=0.02, T1=0.004, lr0=0.4, lr1=0.05)
            elif it % 3 == 1:
                trial = _cheap_descent(trial, steps=120, T0=0.012, T1=0.003, lr0=0.3, lr1=0.03)
            else:
                trial = _cheap_descent(trial, steps=180, T0=0.03, T1=0.006, lr0=0.45, lr1=0.05)
        except Exception:
            pass
        # snap only if time allows; very capped
        if _time.time() - t0 < TIME_LIMIT - RESERVE - 1.0:
            try:
                trial = _lp_polish(trial, 0.7, p_cap=1000, max_it=4)
            except Exception:
                pass
        v = _max_cos(trial)
        if v < best_val - 1e-14:
            best_val = v
            best = trial.copy()
            cur = trial.copy()
            cur_val = v
        else:
            tol = 0.003 * _np.exp(-it / 25.0) + 0.0002
            if v < cur_val + tol:
                cur = trial.copy(); cur_val = v
            if it % 25 == 0:
                cur = best + 0.008 * rng.normal(size=best.shape)
                cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = _max_cos(cur)
        if _time.time() - t0 >= TIME_LIMIT - RESERVE:
            break
    # final polish within remaining budget
    try:
        left = TIME_LIMIT - (_time.time() - t0) - 0.5
        if left > 1.0:
            left = min(left, 9.0)
            polished = _lp_polish(best, left, p_cap=2000, max_it=10)
            pv = _max_cos(polished)
            if pv < best_val and _np.all(_np.isfinite(polished)):
                best = polished
                best_val = pv
    except Exception:
        pass
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best

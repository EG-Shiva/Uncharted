import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_run(y, steps, T0, T1, lr0, lr1):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        y = y - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
    return y

def _lp_polish(x, seconds, p_cap=1000):
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
        I = iu0[sel]; J = iu1[sel]
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
        y /= yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.4, 0.03)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    LIMIT = 178.0
    HOP_END = 158.0
    rng = _np.random.default_rng(0)
    n, d = 123, 13
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= _np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    it = 0
    while _time.time() - t0 < min(HOP_END, LIMIT - 22):
        it += 1
        if _time.time() - t0 > min(HOP_END, LIMIT - 22):
            break
        mode = it % 6
        trial = cur.copy()
        # diagnostics
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        wmax = Gc.max(axis=1)
        thr = cur_val - 0.01
        counts = (Gc > thr).sum(axis=1)
        worst_idx = _np.argsort(-wmax)
        ratt_idx = _np.argsort(counts)
        if mode == 0:
            sigma = 0.020 if (it % 2 == 1) else 0.035
            trial = trial + sigma * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode in (1, 3):
            targets = set()
            for k in range(4):
                targets.add(int(worst_idx[k]))
            for k in range(2):
                targets.add(int(ratt_idx[k]))
            for k in range(2):
                targets.add(int(rng.integers(0, n)))
            targets = list(targets)
            trial += 0.004 * rng.normal(size=trial.shape)
            for idx in targets:
                trial[idx] += 0.20 * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 5:
            trial = trial + 0.045 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # hole reinsertion, 1 or 2 points
            n_replace = 1 if mode == 2 else 2
            repl = [int(worst_idx[k]) for k in range(n_replace)]
            # occasionally replace rattler instead of second worst
            if n_replace == 2 and it % 4 == 0:
                repl[1] = int(ratt_idx[0])
            for r in repl:
                mask = _np.ones(n, dtype=bool); mask[r] = False
                Xr = trial[mask]
                Nc = 5000
                Y = rng.normal(size=(Nc, d))
                Y /= _np.linalg.norm(Y, axis=1, keepdims=True)
                dots = Y @ Xr.T
                md = dots.max(axis=1)
                order = _np.argpartition(md, 4)[:4]
                pick = order[int(rng.integers(0, len(order)))]
                v = Y[pick] + 0.003 * rng.normal(size=(d,))
                v /= _np.linalg.norm(v)
                trial[r] = v
            trial += 0.002 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        # fast annealed repulsion, no LP inside loop
        try:
            cyc = it % 3
            if cyc == 0:
                trial = _smooth_run(trial, steps=260, T0=0.022, T1=0.004, lr0=0.45, lr1=0.05)
            elif cyc == 1:
                trial = _smooth_run(trial, steps=220, T0=0.014, T1=0.003, lr0=0.35, lr1=0.03)
            else:
                trial = _smooth_run(trial, steps=300, T0=0.028, T1=0.005, lr0=0.50, lr1=0.06)
        except Exception:
            pass
        v = _max_cos(trial)
        if v < best_val - 1e-15:
            best_val = v
            best = trial.copy()
            cur = trial.copy()
            cur_val = v
        else:
            tol = 0.003 * _np.exp(-it / 30.0) + 0.00015
            if v < cur_val + tol:
                cur = trial.copy()
                cur_val = v
            if it % 25 == 0:
                cur = best + 0.012 * rng.normal(size=best.shape)
                cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = _max_cos(cur)
        # safety
        if _time.time() - t0 >= min(HOP_END, LIMIT - 22):
            break
    # final single LP snap, strictly time capped
    remain = LIMIT - (_time.time() - t0) - 1.0
    if remain > 2.0:
        fsec = min(remain, 16.0)
        try:
            polished = _lp_polish(best, fsec)
            pv = _max_cos(polished)
            if pv < best_val:
                best = polished
                best_val = pv
        except Exception:
            pass
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget, p_cap=1500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    x = X.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
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
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([ (I[:, None] * d + np.arange(d)), (J[:, None] * d + np.arange(d)), np.full((p, 1), n * d)], axis=1).ravel()
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
        if time.time() - start >= budget:
            break
    return best_x

def _smooth_softmax(y, steps, T0, T1, lr0, lr1):
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _power_descend(y, steps, lr0, lr1, thresh_off=0.06, pw=6.0):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        thr = m - thresh_off
        W = np.maximum(G - thr, 0.0) ** (pw - 1.0)
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-20
        Avg = (W @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # scale to stable step
        y = y - lr * Gt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    LIMIT = 172.0
    rng = _np.random.default_rng(7)
    n, d = 123, 13
    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best0:
                    best0 = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        best0 = _max_cos(X0)
    cur = X0.copy()
    cur_val = float(best0)
    best = X0.copy()
    best_val = float(best0)
    # tiny initial snap
    if _time.time() - t0 < LIMIT - 40:
        try:
            cand = _lp_polish(best, 1.5, p_cap=1500)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
                cur = cand.copy()
                cur_val = v
        except Exception:
            pass
    it = 0
    iu0_dummy = None
    while _time.time() - t0 < LIMIT - 22:
        it += 1
        remain = LIMIT - 22 - (_time.time() - t0)
        if remain < 2.5:
            break
        trial = cur.copy()
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        wmax = Gc.max(axis=1)
        thr_c = cur_val - 0.012
        counts = (Gc > thr_c).sum(axis=1)
        worst = _np.argsort(-wmax)
        ratt = _np.argsort(counts)
        mode = it % 6
        if mode == 0:
            sigma = 0.012 if (it % 12 != 0) else 0.030
            trial = trial + sigma * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode in (1, 3, 5):
            targets = set()
            for k in range(min(5, n)):
                targets.add(int(worst[k]))
            for k in range(min(3, n)):
                targets.add(int(ratt[k]))
            for k in range(2):
                targets.add(int(rng.integers(0, n)))
            targets = list(targets)
            trial += 0.003 * rng.normal(size=trial.shape)
            amp = 0.16 if (mode != 5) else 0.25
            for idx in targets:
                trial[idx] += amp * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # hole reinsertion of 1-2 points
            n_replace = 2 if (mode == 4) else 1
            repl = [int(worst[k]) for k in range(n_replace)]
            # also sometimes replace a rattler instead
            if it % 4 == 0:
                repl[0] = int(ratt[0])
            for r in repl:
                mask = _np.ones(n, dtype=bool)
                mask[r] = False
                Xr = trial[mask]
                n_cand = 12000
                Y = rng.normal(size=(n_cand, d))
                Y /= _np.linalg.norm(Y, axis=1, keepdims=True)
                dots = Y @ Xr.T
                md = dots.max(axis=1)
                order = _np.argpartition(md, 4)[:4]
                pick = order[int(rng.integers(0, len(order)))]
                v = Y[pick] + 0.004 * rng.normal(size=(d,))
                v /= _np.linalg.norm(v)
                trial[r] = v
            trial += 0.0015 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        # descend: alternate schedules
        try:
            if it % 3 == 0:
                trial = _smooth_softmax(trial, steps=220, T0=0.022, T1=0.004, lr0=0.42, lr1=0.05)
            elif it % 3 == 1:
                trial = _smooth_softmax(trial, steps=180, T0=0.013, T1=0.0025, lr0=0.32, lr1=0.03)
            else:
                trial = _power_descend(trial, steps=200, lr0=0.35, lr1=0.04, thresh_off=0.05 + 0.02 * ((it % 5) / 4.0), pw=6.0)
        except Exception:
            pass
        lp_sec = 0.75 if remain > 25 else 0.5
        try:
            trial = _lp_polish(trial, lp_sec, p_cap=1200)
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
                cur = best + 0.008 * rng.normal(size=best.shape)
                cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = _max_cos(cur)
        if _time.time() - t0 >= LIMIT - 22:
            break
    final_budget = max(1.0, LIMIT - (_time.time() - t0) - 1.0)
    final_budget = min(final_budget, 21.0)
    try:
        polished = _lp_polish(best, final_budget, p_cap=3000)
        pv = _max_cos(polished)
        if pv < best_val:
            best = polished
            best_val = pv
    except Exception:
        pass
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best

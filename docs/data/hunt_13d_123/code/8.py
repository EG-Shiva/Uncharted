import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, p_cap=2200):
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
        cols = np.concatenate([ (I[:, None] * d + np.arange(d)),
                                (J[:, None] * d + np.arange(d)),
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
        y = y / yn
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


def _smooth_run(y, steps, T0, T1, lr0, lr1, active=None, mu=0.85):
    n, d = y.shape
    vel = np.zeros_like(y)
    if active is None:
        amask = None
    else:
        amask = np.zeros(n, dtype=bool)
        amask[np.asarray(active)] = True
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
        if amask is not None:
            Gt[~amask, :] = 0.0
        vel = mu * vel + Gt
        y = y - lr * vel
        if amask is not None:
            y[amask] /= np.linalg.norm(y[amask], axis=1, keepdims=True)
        else:
            y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def _find_hole(Xfrozen, rng, n_cand=25000, chunk=8000, jitter=0.003):
    n, d = Xfrozen.shape
    best_v = None
    best_m = 1e9
    # search in chunks to bound memory
    for s in range(0, n_cand, chunk):
        c = min(chunk, n_cand - s)
        Y = rng.normal(size=(c, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        dots = Y @ Xfrozen.T
        md = dots.max(axis=1)
        j = int(np.argmin(md))
        if md[j] < best_m:
            best_m = float(md[j])
            best_v = Y[j].copy()
    v = best_v + jitter * rng.normal(size=(d,))
    v /= np.linalg.norm(v)
    return v


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    TIME_LIMIT = 178.0
    rng = _np.random.default_rng(1)
    _np.random.seed(1)
    n, d = 123, 13
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
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
    # quick initial tightening
    try:
        if _time.time() - t0 < TIME_LIMIT - 40:
            cand = _lp_polish(X0, 1.5)
            v = _max_cos(cand)
            if v < best_val0:
                X0 = cand
                best_val0 = v
    except Exception:
        pass
    best = X0.copy()
    best_val = float(best_val0)
    cur = X0.copy()
    cur_val = float(best_val0)

    it = 0
    patch_sizes = [12, 16, 20, 25, 14, 18, 22]
    while _time.time() - t0 < TIME_LIMIT - 14:
        it += 1
        remaining = TIME_LIMIT - 14 - (_time.time() - t0)
        if remaining < 3:
            break
        mode = it % 6
        trial = cur.copy()
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        wmax = Gc.max(axis=1)
        thr = cur_val - 0.008
        counts = (Gc > thr).sum(axis=1)
        worst_order = _np.argsort(-wmax)
        ratt_order = _np.argsort(counts)

        if mode in (0, 1, 3):
            # --- patch surgery: repack cap around worst point ---
            k = patch_sizes[(it // 6) % len(patch_sizes)]
            # pick centre: cycle through worst few for diversity
            ci = int(worst_order[(it // 2) % 6])
            dots_c = cur @ cur[ci]
            patch = _np.argsort(-dots_c)[:k]
            # centroid direction of patch (background-aware)
            centre = cur[patch].mean(axis=0)
            centre /= _np.linalg.norm(centre) + 1e-20
            # from-scratch repack of patch around centre with wide spread
            spread = 0.55 if (it % 2 == 0) else 0.9
            for idx in patch:
                v = centre + spread * rng.normal(size=(d,))
                v /= _np.linalg.norm(v)
                trial[idx] = v
            # frozen-background inner optimisation (topology change inside cap)
            try:
                trial = _smooth_run(trial, steps=220, T0=0.02, T1=0.004,
                                    lr0=0.5, lr1=0.06, active=patch, mu=0.85)
                # unfreeze global anneal
                trial = _smooth_run(trial, steps=160, T0=0.012, T1=0.0025,
                                    lr0=0.3, lr1=0.03, active=None, mu=0.9)
            except Exception:
                pass
            # hole-fix rattler if exists: replace lowest-contact point by hole
            try:
                Gt2 = trial @ trial.T
                _np.fill_diagonal(Gt2, -2.0)
                c2 = (Gt2 > cur_val - 0.008).sum(axis=1)
                r = int(_np.argmin(c2))
                if c2[r] < 8:
                    mask = _np.ones(n, dtype=bool); mask[r] = False
                    hv = _find_hole(trial[mask], rng, n_cand=15000)
                    trial[r] = hv
            except Exception:
                pass
        elif mode == 2 or mode == 5:
            # --- targeted kick to overloaded + rattlers ---
            targets = set()
            for kk in range(5):
                targets.add(int(worst_order[kk]))
            for kk in range(3):
                targets.add(int(ratt_order[kk]))
            for kk in range(2):
                targets.add(int(rng.integers(0, n)))
            targets = list(targets)
            trial += 0.003 * rng.normal(size=trial.shape)
            for idx in targets:
                trial[idx] += 0.22 * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            try:
                if it % 2 == 0:
                    trial = _smooth_run(trial, steps=320, T0=0.022, T1=0.003, lr0=0.42, lr1=0.04, active=None, mu=0.88)
                else:
                    trial = _smooth_run(trial, steps=380, T0=0.03, T1=0.005, lr0=0.5, lr1=0.05, active=None, mu=0.85)
            except Exception:
                pass
        else:  # mode == 4 hole re-insertion of 2 worst
            n_replace = 2
            repl = [int(worst_order[kk]) for kk in range(n_replace)]
            for r in repl:
                mask = _np.ones(n, dtype=bool); mask[r] = False
                hv = _find_hole(trial[mask], rng, n_cand=20000, jitter=0.004)
                trial[r] = hv
            trial += 0.002 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            try:
                trial = _smooth_run(trial, steps=300, T0=0.018, T1=0.003, lr0=0.4, lr1=0.04, active=None, mu=0.88)
            except Exception:
                pass

        lp_sec = 0.9 if remaining > 25 else 0.6
        try:
            trial = _lp_polish(trial, lp_sec)
        except Exception:
            pass
        v = _max_cos(trial)
        if v < best_val - 1e-14:
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
        if _time.time() - t0 >= TIME_LIMIT - 14:
            break

    final_sec = max(1.0, TIME_LIMIT - (_time.time() - t0) - 1.0)
    final_sec = min(final_sec, 13.0)
    try:
        polished = _lp_polish(best, final_sec)
        pv = _max_cos(polished)
        if pv < best_val:
            best = polished
            best_val = pv
    except Exception:
        pass
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best

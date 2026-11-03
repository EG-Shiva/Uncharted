import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, p_cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    # precompute iu arrays
    iu0 = iu[0]
    iu1 = iu[1]
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20 * radius, 1e-9)
        sel = c_all > m - window
        # cap to tightest pairs for speed
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            # keep top p_cap
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
        # cols
        Ioff = (I[:, None] * d + np.arange(d)).ravel()
        Joff = (J[:, None] * d + np.arange(d)).ravel()
        cols_I = Ioff
        cols_J = Joff
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = None  # placeholder not used
        # build efficiently with concatenate
        cols = np.concatenate([Ioff.reshape(p, d), Joff.reshape(p, d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            # equality: x_i . dx_i = 0
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
            # accept move of x even if not best? no, shrink
            radius *= 0.5
            # if stuck, keep x as best_x
            x = best_x.copy() if radius < 1e-7 else x
        # time check inside
        if time.time() - start >= seconds:
            break
    return best_x


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
        y = y - lr * Gt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    TIME_LIMIT = 178.0
    _np.random.seed(0)
    rng = _np.random.default_rng(0)
    n, d = 123, 13
    # load start
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
    cur = X0.copy()
    cur_val = float(best_val0)
    best = X0.copy()
    best_val = float(best_val0)

    # quick squeeze of starting point if time allows (2s)
    if _time.time() - t0 < TIME_LIMIT - 30:
        try:
            cand = _lp_polish(best, 2.0)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
                cur = cand.copy()
                cur_val = v
        except Exception:
            pass

    it = 0
    # reserve ~18s for final polish
    while _time.time() - t0 < TIME_LIMIT - 18:
        it += 1
        remaining = TIME_LIMIT - 18 - (_time.time() - t0)
        if remaining <= 3:
            break
        mode = it % 5
        trial = cur.copy()
        # compute diagnostics on cur occasionally
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        wmax = Gc.max(axis=1)
        # contact counts with loose threshold for targeting
        thr = cur_val - 0.01
        counts = (Gc > thr).sum(axis=1)
        worst_idx = _np.argsort(-wmax)  # descending
        ratt_idx = _np.argsort(counts)  # ascending: rattlers first

        if mode == 0:
            # global small shake
            sigma = 0.015 if (it % 10 != 0) else 0.035
            trial = trial + sigma * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 1 or mode == 3:
            # targeted shake of worst + rattlers + random
            targets = set()
            for k in range(min(4, n)):
                targets.add(int(worst_idx[k]))
            for k in range(min(2, n)):
                targets.add(int(ratt_idx[k]))
            # + 2 random
            for k in range(2):
                targets.add(int(rng.integers(0, n)))
            targets = list(targets)
            # big kick on targets, tiny on rest
            trial += 0.004 * rng.normal(size=trial.shape)
            for idx in targets:
                trial[idx] += 0.18 * rng.normal(size=(d,))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
        else:
            # hole re-insertion: replace 1-2 worst points by best holes
            n_replace = 2 if (mode == 4 and it % 2 == 0) else 1
            # choose points to replace: worst ones (cycle)
            repl = [int(worst_idx[k]) for k in range(n_replace)]
            # for each replacement, search hole w.r.t. remaining
            for r in repl:
                mask = _np.ones(n, dtype=bool)
                mask[r] = False
                Xr = trial[mask]
                n_cand = 20000
                Y = rng.normal(size=(n_cand, d))
                Y /= _np.linalg.norm(Y, axis=1, keepdims=True)
                dots = Y @ Xr.T
                md = dots.max(axis=1)
                # pick among top-5 best with randomness to diversify
                order = _np.argpartition(md, 5)[:5]
                pick = order[int(rng.integers(0, len(order)))]
                # small jitter around hole
                v = Y[pick] + 0.005 * rng.normal(size=(d,))
                v /= _np.linalg.norm(v)
                trial[r] = v
            # tiny global jitter
            trial += 0.002 * rng.normal(size=trial.shape)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)

        # smooth repulsion descent
        try:
            # vary schedule slightly by iteration for diversity
            if it % 3 == 0:
                trial = _smooth_run(trial, steps=350, T0=0.025, T1=0.004, lr0=0.45, lr1=0.05)
            elif it % 3 == 1:
                trial = _smooth_run(trial, steps=300, T0=0.015, T1=0.003, lr0=0.35, lr1=0.03)
            else:
                trial = _smooth_run(trial, steps=400, T0=0.03, T1=0.006, lr0=0.5, lr1=0.06)
        except Exception:
            pass

        # short LP snap
        lp_sec = 1.6 if remaining > 20 else 1.0
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
            # annealing acceptance for walk
            tol = 0.004 * _np.exp(-it / 25.0) + 0.0002
            if v < cur_val + tol:
                cur = trial.copy()
                cur_val = v
            # periodic restart from best with shake
            if it % 22 == 0:
                cur = best + 0.01 * rng.normal(size=best.shape)
                cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
                cur_val = _max_cos(cur)
        # early exit if time low
        if _time.time() - t0 >= TIME_LIMIT - 18:
            break

    # final long polish of best
    final_sec = max(1.0, TIME_LIMIT - (_time.time() - t0) - 1.0)
    final_sec = min(final_sec, 20.0)
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

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import linprog as _linprog
    from scipy.sparse import coo_matrix as _coo
    t0 = _time.time()
    HARD = 172.0  # must return before 180s

    rng = _np.random.default_rng(0)
    n, d = 123, 13

    def max_cos(x):
        g = x @ x.T
        _np.fill_diagonal(g, -2.0)
        return float(g.max())

    # precompute pair indices
    _iu0, _iu1 = _np.triu_indices(n, 1)

    def lp_polish(x, seconds, p_cap=2000):
        start = _time.time()
        x = x / _np.linalg.norm(x, axis=1, keepdims=True)
        best = max_cos(x)
        best_x = x.copy()
        radius = 1e-3
        nv = n * d + 1
        while _time.time() - start < seconds and radius > 1e-14:
            if _time.time() - t0 > HARD - 2.0:
                break
            G = x @ x.T
            c_all = G[_iu0, _iu1]
            m = c_all.max()
            window = max(20.0 * radius, 1e-9)
            sel = c_all > m - window
            cnt = int(_np.count_nonzero(sel))
            if cnt == 0:
                break
            if cnt > p_cap:
                idx = _np.where(sel)[0]
                part = _np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
                keep = _np.zeros_like(sel)
                keep[idx[part]] = True
                sel = keep
                cnt = p_cap
            I = _iu0[sel]
            J = _iu1[sel]
            csel = c_all[sel]
            p = cnt
            rows = _np.repeat(_np.arange(p), 2 * d + 1)
            cols = _np.concatenate(
                [(I[:, None] * d + _np.arange(d)).ravel(),
                 (J[:, None] * d + _np.arange(d)).ravel(),
                 _np.full(p, n * d)], axis=0
            )
            # need rows-aligned order: I-block, J-block, t-block per row
            # rebuild in per-row order
            Ioff = (I[:, None] * d + _np.arange(d))
            Joff = (J[:, None] * d + _np.arange(d))
            cols2 = _np.empty(p * (2 * d + 1), dtype=_np.int64)
            cols2 = _np.concatenate([Ioff, Joff, _np.full((p, 1), n * d)], axis=1).ravel()
            vals = _np.concatenate([x[J], x[I], -_np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = _coo((vals, (rows, cols2)), shape=(p, nv)).tocsr()
                erows = _np.repeat(_np.arange(n), d)
                ecols = _np.arange(n * d)
                A_eq = _coo((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
                obj = _np.zeros(nv)
                obj[-1] = 1.0
                bd = (-radius, radius)
                res = _linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                               b_eq=_np.zeros(n),
                               bounds=[bd] * (n * d) + [(None, None)],
                               method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            yn = _np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = max_cos(y)
            if v < best:
                best = v
                best_x = y.copy()
                x = y
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return best_x

    def smooth(y, steps, T0, T1, lr0, lr1):
        for s in range(steps):
            frac = s / max(steps - 1, 1)
            T = T0 + (T1 - T0) * frac
            lr = lr0 + (lr1 - lr0) * frac
            G = y @ y.T
            _np.fill_diagonal(G, -10.0)
            m = G.max()
            E = _np.exp((G - m) / T)
            _np.fill_diagonal(E, 0.0)
            row = E.sum(axis=1, keepdims=True) + 1e-20
            Avg = (E @ y) / row
            dot = (Avg * y).sum(axis=1, keepdims=True)
            Gt = Avg - dot * y
            y = y - lr * Gt
            y /= _np.linalg.norm(y, axis=1, keepdims=True)
            if (s & 127) == 0 and _time.time() - t0 > HARD - 3.0:
                break
        return y

    # load start
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                if _np.any(~_np.isfinite(a)):
                    continue
                v = max_cos(a)
                if v < best_val0:
                    best_val0 = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = max_cos(X0)

    best = X0.copy()
    best_val = float(best_val0)

    # quick initial tighten
    try:
        if _time.time() - t0 < HARD - 30:
            cand = lp_polish(best, 3.0)
            v = max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
    except Exception:
        pass

    def contacts_of(X):
        G = X @ X.T
        m = G.max() - 1e-9  # exclude diag (1.0 > m generally); but guard
        # diag is 1.0, m~0.27, so threshold ~0.27, diag counts; subtract 1
        C = (G > m).sum(axis=1) - 1
        return C, float(G[_iu0, _iu1].max())

    contacts, _ = contacts_of(best)

    it = 0
    # main hopping, reserve ~15s for final polish
    while _time.time() - t0 < HARD - 16.0 and it < 300:
        it += 1
        # refresh contacts every few iters or after improvement
        if it % 3 == 1:
            try:
                contacts, _ = contacts_of(best)
            except Exception:
                pass
        y = best.copy()
        r = float(rng.random())
        try:
            if r < 0.32:
                # rattler / loose hole reinsertion
                order_lo = _np.argsort(contacts)
                # pick among 5 loosest at random
                cand_lo = order_lo[:5]
                k = int(rng.choice(cand_lo))
                R = rng.normal(size=(12000, d))
                R /= _np.linalg.norm(R, axis=1, keepdims=True)
                mask = _np.ones(n, dtype=bool)
                mask[k] = False
                S = R @ y[mask].T
                mm = S.max(axis=1)
                idx = int(_np.argmin(mm))
                y[k] = R[idx]
                # tiny jitter to others to break symmetry
                y += rng.normal(scale=0.008, size=y.shape)
            elif r < 0.62:
                # kick overloaded jammed points
                order_hi = _np.argsort(-contacts)
                nkick = int(rng.integers(3, 7))
                pool = order_hi[:20]
                chosen = rng.choice(pool, size=min(nkick, len(pool)), replace=False)
                sigma = float(rng.uniform(0.08, 0.25))
                y[chosen] += rng.normal(scale=sigma, size=(len(chosen), d))
            elif r < 0.87:
                nkick = int(rng.integers(5, 14))
                chosen = rng.choice(n, size=nkick, replace=False)
                sigma = float(rng.uniform(0.15, 0.45))
                y[chosen] += rng.normal(scale=sigma, size=(len(chosen), d))
            else:
                sigma = float(rng.uniform(0.02, 0.08))
                y += rng.normal(scale=sigma, size=y.shape)
        except Exception:
            continue
        yn = _np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn

        steps = int(rng.integers(600, 1300))
        T0s = float(rng.uniform(0.008, 0.025))
        T1s = T0s * 0.1
        lr0 = float(rng.uniform(0.08, 0.20))
        lr1 = lr0 * 0.15
        try:
            y = smooth(y, steps, T0s, T1s, lr0, lr1)
        except Exception:
            continue
        if _time.time() - t0 > HARD - 16.0:
            break
        try:
            v = max_cos(y)
        except Exception:
            continue
        if v > best_val + 0.012:
            continue
        # promising -> short LP
        try:
            y2 = lp_polish(y, 1.0, p_cap=2000)
            v2 = max_cos(y2)
            if v2 < best_val - 1e-12:
                best = y2
                best_val = v2
                contacts, _ = contacts_of(best)
        except Exception:
            continue

    # final polish with remaining budget
    try:
        remain = HARD - (_time.time() - t0) - 1.0
        if remain > 2.0:
            cand = lp_polish(best, min(remain, 12.0), p_cap=3000)
            v = max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
    except Exception:
        pass

    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    # shrink tiny margin guard (no-op, ensures finite)
    best[~_np.isfinite(best)] = 0.0
    nn = _np.linalg.norm(best, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    best = best / nn
    return best

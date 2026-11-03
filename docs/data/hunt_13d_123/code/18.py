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


def _smooth_descent(y, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = y.shape
    vel = np.zeros_like(y)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        m = G.max()
        # exp in stable way; diag will be zeroed
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        # symmetric weights
        W = E / (E.sum() + 1e-30)
        Svec = W @ y
        dot = (Svec * y).sum(axis=1, keepdims=True)
        Gt = Svec - dot * y
        gn = np.sqrt((Gt ** 2).sum(axis=1, keepdims=True)) + 1e-12
        Gn = Gt / gn
        vel = mu * vel - lr * Gn
        y = y + vel
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def _lp_polish(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
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
        cols = np.concatenate(
            [(I[:, None] * d + np.arange(d)).reshape(p, d),
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


def solve():
    t0 = time.time()
    TIME_LIMIT = 178.0
    rng = np.random.default_rng(0)
    n, d = 123, 13
    best = None
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
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur_best = best.copy()
    cur_val = best_val

    it = 0
    # reserve ~22s for final LP
    while time.time() - t0 < TIME_LIMIT - 22.0:
        it += 1
        remain = TIME_LIMIT - 22.0 - (time.time() - t0)
        if remain <= 0:
            break
        y = cur_best.copy()
        # per-point worst
        try:
            pw = _per_point_max(y)
        except Exception:
            pw = np.zeros(n)
        order_worst = np.argsort(-pw)
        order_best = np.argsort(pw)  # rattlers / isolated
        mode = it % 5
        if mode == 0:
            # global small shake
            sigma = 0.03
            y = y + sigma * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 220, 0.012, 0.0015, 0.025, 0.003
        elif mode == 1:
            # targeted medium kick to hubs
            k = int(rng.integers(3, 7))
            idx = rng.choice(order_worst[:22], size=min(k, 22), replace=False)
            sigma = 0.22
            y[idx] += sigma * rng.normal(size=(len(idx), d))
            # also nudge rattler slightly
            try:
                r = order_best[0]
                y[r] += 0.1 * rng.normal(size=(d,))
            except Exception:
                pass
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 300, 0.015, 0.0012, 0.03, 0.002
        elif mode == 2:
            # hole reinsertion: replace 1-2 points randomly
            k = int(rng.integers(1, 3))
            # pick victims: one rattler/isolated + one hub
            victims = []
            victims.append(int(order_best[0]))
            if k > 1:
                victims.append(int(rng.choice(order_worst[:15])))
            for v in victims:
                rv = rng.normal(size=(d,))
                rv /= np.linalg.norm(rv)
                y[v] = rv
            # small shake to neighbours of victims to open space
            sigma = 0.05
            y += sigma * rng.normal(size=(n, d)) * 0.3
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 320, 0.018, 0.001, 0.035, 0.002
        elif mode == 3:
            # large kick to overloaded cluster
            k = int(rng.integers(5, 10))
            pool = order_worst[:30]
            idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
            sigma = 0.35
            y[idx] += sigma * rng.normal(size=(len(idx), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 350, 0.02, 0.0015, 0.04, 0.003
        else:
            # random subset shake
            k = int(rng.integers(8, 20))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.12
            y[idx] += sigma * rng.normal(size=(k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 250, 0.012, 0.001, 0.03, 0.002
        # smooth descent
        try:
            y = _smooth_descent(y, steps, T0, T1, lr0, lr1, mu=0.85)
            v = _max_cos(y)
        except Exception:
            continue
        if v < cur_val:
            cur_val = v
            cur_best = y.copy()
            if v < best_val:
                best_val = v
                best = y.copy()
        # occasional restart from best with extra noise if stuck
        if it % 40 == 0 and (time.time() - t0) < TIME_LIMIT - 40:
            # small random walk on cur_best to diversify
            cur_best = best.copy()
            cur_val = best_val
        if time.time() - t0 >= TIME_LIMIT - 22.0:
            break

    # final squeeze
    try:
        rem = TIME_LIMIT - (time.time() - t0) - 1.0
        if rem > 4.0:
            # first quick smooth anneal at low T to tighten
            cand = _smooth_descent(cur_best.copy(), 200, 0.006, 0.0006, 0.008, 0.001, mu=0.9)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
            # LP polish on the best
            rem2 = TIME_LIMIT - (time.time() - t0) - 0.5
            if rem2 > 3.0:
                polished = _lp_polish(best.copy(), min(rem2, 18.0))
                vp = _max_cos(polished)
                if vp < best_val:
                    best_val = vp
                    best = polished.copy()
                elif vp <= best_val + 1e-12:
                    # keep tighter (more contacts) if essentially equal
                    best = polished.copy()
                    best_val = vp
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: replace any non-finite row
    if not np.all(np.isfinite(best)):
        a = np.load("best.npy")
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        best = a
    return best

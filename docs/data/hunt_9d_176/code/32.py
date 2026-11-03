import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _quench(X, steps, a0=25.0, a1=320.0, lr0=0.02, lr1=0.002):
    n = X.shape[0]
    Y = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # exp weights, shift by max for stability
        W = np.exp((G - m) * alpha)
        np.fill_diagonal(W, 0.0)
        Grad = W @ Y
        proj = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - proj * Y
        gn = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        Y = Y - (lr / gn) * Gt
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        Y = Y / nr
    return Y

def _lp_polish_deadline(x, deadline, r_init=0.002, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            # keep largest `cap`
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d))
        cols_J = (J[:, None] * d + np.arange(d))
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = cols_I
        cols[:, d:2*d] = cols_J
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 118.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    best_X = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    a = a / nr
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = rng.standard_normal((n, d))
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        best_v = _max_cos(best_X)

    trial = 0
    # cheap screening loop: perturb + quench only
    while time.time() < SEARCH_END:
        trial += 1
        mode = trial % 5
        sigma_cycle = [0.002, 0.006, 0.015, 0.035, 0.08]
        sigma = sigma_cycle[(trial - 1) % len(sigma_cycle)]
        Xp = best_X.copy()
        if mode == 0:
            # global shake
            Xp = Xp + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 600
        elif mode == 1:
            # worst-point hole re-insert
            try:
                Gc = Xp @ Xp.T
                np.fill_diagonal(Gc, -2.0)
                rm = Gc.max(axis=1)
                idx = int(np.argmax(rm))
            except Exception:
                idx = int(rng.integers(0, n))
            M = 2500
            R = rng.standard_normal((M, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            dots = R @ Xp.T
            dots[:, idx] = -2.0
            worst = dots.max(axis=1)
            pick = int(np.argmin(worst))
            Xp[idx] = R[pick]
            Xp = Xp + 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 500
        elif mode == 2:
            # 3-hole sequential re-insert
            try:
                Gc = Xp @ Xp.T
                np.fill_diagonal(Gc, -2.0)
                rm = Gc.max(axis=1)
                order = np.argsort(rm)
                idxs = order[-3:]
            except Exception:
                idxs = rng.choice(n, size=3, replace=False)
            Xp = Xp + 0.003 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            for idx in idxs:
                M = 900
                R = rng.standard_normal((M, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                dots = R @ Xp.T
                dots[:, int(idx)] = -2.0
                worst = dots.max(axis=1)
                pick = int(np.argmin(worst))
                Xp[int(idx)] = R[pick]
            steps = 500
        elif mode == 3:
            # block shake
            k = 20
            idxs = rng.choice(n, size=k, replace=False)
            Xp[idxs] += sigma * rng.standard_normal((k, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 550
        else:
            Xp = Xp + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            steps = 400
        # quench with schedule depending on sigma
        if sigma > 0.02:
            a1 = 180.0
        else:
            a1 = 340.0
        try:
            Xq = _quench(Xp, steps=steps, a0=25.0, a1=a1, lr0=0.02, lr1=0.002)
            v = _max_cos(Xq)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_v:
            best_v = v
            best_X = Xq.copy()
        # time guard
        if time.time() >= SEARCH_END:
            break

    # single final LP squeeze on the best
    try:
        if time.time() < HARD_END - 2.0:
            best_X = _lp_polish_deadline(best_X, HARD_END - 1.0, r_init=0.0015, cap=2600)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def lp_polish(x, seconds):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for very large problems
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = m - max(20 * radius, 1e-9)
        sel = c > thr
        # cap number of constraints for speed
        if int(sel.sum()) > 2500:
            # keep tightest 2500
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[:2500]] = True
            sel = sel & keep
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def softmax_anneal(X0, steps, T0, T1, lr, rng):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        # exponential schedule
        T = T0 * (T1 / T0) ** frac if T1 > 0 else T0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        # scaled exp, avoid overflow/underflow
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()) + 1e-30
        Gr = (E @ X) / S
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X = _norm(X)
    return X


def solve():
    t0 = time.time()
    TIME_LIMIT = 178.0
    # reserve for final polish
    EXPLORE_END = 148.0
    rng = np.random.default_rng(12345)
    n, d = 76, 15

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (76, 15):
                a = _norm(a)
                va = _max_cos(a)
                if va < best_val:
                    best, best_val = a, va
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)

    # quick polish of starting point already done in files; don't waste time

    it = 0
    # precompute for contact analysis inside loop
    while time.time() - t0 < EXPLORE_END:
        it += 1
        time_left = EXPLORE_END - (time.time() - t0)
        if time_left <= 0:
            break
        X = best.copy()
        mode = it % 7

        # contact info of current best
        try:
            Gb = best @ best.T
            np.fill_diagonal(Gb, -2.0)
            rowmax = Gb.max(axis=1)
            # contacts within 5e-3 of max
            Ccnt = (Gb > best_val - 0.005).sum(axis=1)
        except Exception:
            rowmax = np.full(n, best_val)
            Ccnt = np.zeros(n)

        if mode == 0:
            # rattler reinsertion: fully randomize worst slack points
            ratt = np.where(rowmax < best_val - 5e-4)[0]
            if len(ratt) < 4:
                ratt = np.argsort(rowmax)[:8]
            for i in ratt:
                vv = rng.normal(size=d)
                vv /= (np.linalg.norm(vv) + 1e-30)
                X[i] = vv
            steps, T0, T1, lr = 1600, 0.010, 0.0008, 0.008
        elif mode == 1:
            # small global shake
            sig = 0.04 + 0.03 * rng.random()
            X = _norm(X + sig * rng.normal(size=X.shape))
            steps, T0, T1, lr = 900, 0.008, 0.0007, 0.006
        elif mode == 2:
            # shake overloaded / jammed points
            order = np.argsort(-Ccnt)
            k = 6
            idx = order[:k]
            X[idx] = _norm(X[idx] + 0.25 * rng.normal(size=(k, d)))
            steps, T0, T1, lr = 1400, 0.009, 0.0007, 0.008
        elif mode == 3:
            # random subset medium shake
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            X[idx] = _norm(X[idx] + 0.35 * rng.normal(size=(k, d)))
            steps, T0, T1, lr = 1400, 0.010, 0.0008, 0.008
        elif mode == 4:
            # single-point optimal reinsertion for worst rattler + anneal rest slightly
            worst = int(np.argmin(rowmax))
            # try several random candidates for that point, pick best
            cand_best = None
            cand_val = 1e9
            for _ in range(200):
                vv = rng.normal(size=d)
                vv /= (np.linalg.norm(vv) + 1e-30)
                # max cosine of vv to others
                dots = X @ vv
                dots[worst] = -2.0
                mv = float(dots.max())
                if mv < cand_val:
                    cand_val = mv
                    cand_best = vv
            X[worst] = cand_best
            # slight shake of neighbours
            X = _norm(X + 0.02 * rng.normal(size=X.shape))
            steps, T0, T1, lr = 1200, 0.007, 0.0006, 0.006
        elif mode == 5:
            # large global shake to change basin
            sig = 0.12 + 0.08 * rng.random()
            X = _norm(X + sig * rng.normal(size=X.shape))
            steps, T0, T1, lr = 1800, 0.012, 0.0008, 0.010
        else:
            # occasional fresh random start with long anneal
            if it % 21 == 0:
                X = _norm(rng.normal(size=(n, d)))
                steps, T0, T1, lr = 2500, 0.015, 0.0008, 0.010
            else:
                # PCA-inspired: push along random direction to split clusters
                direction = rng.normal(size=d)
                direction /= (np.linalg.norm(direction) + 1e-30)
                proj = X @ direction
                # shake points with extreme projection
                order = np.argsort(proj)
                idx = np.concatenate([order[:5], order[-5:]])
                X[idx] = _norm(X[idx] + 0.3 * rng.normal(size=(len(idx), d)))
                steps, T0, T1, lr = 1300, 0.009, 0.0007, 0.007

        Xa = softmax_anneal(X, steps, T0, T1, lr, rng)
        va = _max_cos(Xa)

        # only spend LP time on promising candidates
        thresh = best_val + (0.002 if mode in (0, 2, 4) else 0.0008)
        if va < thresh:
            # short polish, 0.8-1.2s
            budget = 1.0
            # if clearly better, give a bit more
            if va < best_val:
                budget = 1.5
            # guard overall time
            if time.time() - t0 + budget > EXPLORE_END + 5:
                budget = max(0.3, EXPLORE_END - (time.time() - t0))
            try:
                Xp = lp_polish(Xa, budget)
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = Xa, va
            if vp < best_val - 1e-12:
                best, best_val = Xp, vp
            elif va < best_val - 1e-12:
                # polish didn't help but raw is better (shouldn't happen)
                best, best_val = Xa, va
        # opportunistic: if raw is much better even without polish, take it
        # (already handled above)

    # final high-precision polish of best with remaining time
    remaining = TIME_LIMIT - (time.time() - t0) - 1.0
    if remaining > 4:
        try:
            best = lp_polish(best, remaining)
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    return best

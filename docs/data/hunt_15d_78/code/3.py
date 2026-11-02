import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def contact_counts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = (G > m - tol).sum(axis=1)
    return C, float(m), G


def lp_polish(X, budget, r0=1e-3, rmax=0.05):
    X = norm_rows(np.array(X, dtype=float, copy=True))
    n, d = X.shape
    best = max_cos(X)
    t_start = time.time()
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < budget and radius > 1e-14:
        Gfull = X @ X.T
        c_all = Gfull[iu]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        # cap constraints to tightest 2500 to keep HiGHS fast
        if np.count_nonzero(sel) > 2500:
            idx = np.argsort(-c_all)
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[:2500]] = True
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        c_sel = c_all[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = norm_rows(Y)
        v = max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
    return X


def refine_hole(y0, Xf, steps=600, lr0=0.08):
    y = y0 / np.linalg.norm(y0)
    lr = lr0
    for s in range(steps):
        dots = Xf @ y
        # use top-k active constraints averaged
        k = 3
        idx = np.argpartition(-dots, kth=min(k, len(dots)-1))[:k]
        g = Xf[idx].sum(axis=0)
        # tangential
        g = g - (g @ y) * y
        ng = np.linalg.norm(g)
        if ng > 1e-12:
            g = g / ng
        y = y - lr * g
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            return y0 / np.linalg.norm(y0)
        y = y / nrm
        lr *= 0.997
    return y


def best_hole_for_set(Xf, rng, n_cand=20000):
    d = Xf.shape[1]
    Z = rng.standard_normal((n_cand, d))
    Z = norm_rows(Z)
    scores = (Z @ Xf.T).max(axis=1)
    order = np.argsort(scores)
    best_y = None
    best_s = 1e9
    for cand in order[:4]:
        y = refine_hole(Z[cand], Xf, steps=500, lr0=0.08)
        s = float((Xf @ y).max())
        if s < best_s:
            best_s = s
            best_y = y
    return best_y


def kill_reinsert(X, k, rng):
    n, d = X.shape
    C, m, G = contact_counts(X, tol=5e-4)
    # remove k most-contact points (jammed); if k>1 also drop a rattler
    order = np.argsort(C)
    # mix: take mostly high-contact, sometimes low
    idx_remove = list(np.argsort(-C)[:k])
    mask = np.ones(n, dtype=bool)
    mask[np.array(idx_remove)] = False
    Xrem = X[mask]
    cur = Xrem
    for _ in range(k):
        y = best_hole_for_set(cur, rng, n_cand=15000)
        cur = np.vstack([cur, y.reshape(1, d)])
    # cur has n rows but order changed; fine
    return norm_rows(cur)


def soft_anneal(X, rng, steps=1200, beta0=60.0, beta1=220.0, lr=0.02):
    X = norm_rows(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        beta = beta0 + (beta1 - beta0) * s / max(1, steps - 1)
        G = X @ X.T
        # shift for stability using max per row approx
        # full exp matrix
        np.fill_diagonal(G, -10.0)
        W = np.exp(beta * (G - G.max()))
        np.fill_diagonal(W, 0.0)
        Grad = beta * (W @ X) / n
        # tangential projection
        proj = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - proj * X
        # adam
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        mhat = m1 / (1 - b1 ** (s + 1))
        vhat = v1 / (1 - b2 ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # limit step norm
        X = X - step
        X = norm_rows(X)
        # small random kick early to break symmetry
        if s == steps // 3:
            X = norm_rows(X + 1e-4 * rng.standard_normal(X.shape))
    return X


def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng_master = np.random.default_rng(20261101)

    def time_left():
        return DEADLINE - time.time()

    # load starts
    bestX = None
    bestV = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            X = np.load(fn)
            X = np.array(X, dtype=float)
            if X.shape != (78, 15):
                continue
            X = norm_rows(X)
            if not np.all(np.isfinite(X)):
                continue
            v = max_cos(X)
            if v < bestV:
                bestV = v
                bestX = X.copy()
        except Exception:
            continue
    if bestX is None:
        rng = np.random.default_rng(0)
        bestX = norm_rows(rng.standard_normal((78, 15)))
        bestV = max_cos(bestX)

    # quick initial squeeze if time permits
    if time_left() > 20:
        bestX = lp_polish(bestX, min(6.0, time_left() - 12.0), r0=5e-4)
        bestV = max_cos(bestX)

    it = 0
    # deterministic cycle of moves
    while time_left() > 14.0:
        it += 1
        cycle = it % 7
        rng = np.random.default_rng(100000 + it * 7919)
        Xt = None
        try:
            if cycle == 0:
                sig = 0.004
                Xt = norm_rows(bestX + sig * rng.standard_normal(bestX.shape))
                Xt = lp_polish(Xt, min(2.2, time_left() - 11.0), r0=8e-4)
            elif cycle == 1:
                sig = 0.016
                Xt = norm_rows(bestX + sig * rng.standard_normal(bestX.shape))
                Xt = lp_polish(Xt, min(2.5, time_left() - 11.0), r0=1e-3)
            elif cycle == 2:
                C, _, _ = contact_counts(bestX, tol=1e-9)
                low = np.argsort(C)[:12]
                Xt = bestX.copy()
                Xt[low] += 0.07 * rng.standard_normal((len(low), bestX.shape[1]))
                Xt = norm_rows(Xt)
                Xt = lp_polish(Xt, min(2.5, time_left() - 11.0), r0=1e-3)
            elif cycle == 3:
                Xt = kill_reinsert(bestX, 1, rng)
                Xt = lp_polish(Xt, min(3.0, time_left() - 11.0), r0=1e-3)
            elif cycle == 4:
                Xt = kill_reinsert(bestX, 2, rng)
                Xt = lp_polish(Xt, min(3.2, time_left() - 11.0), r0=1e-3)
            elif cycle == 5:
                # shake overloaded hub
                C, _, _ = contact_counts(bestX, tol=5e-4)
                hi = np.argsort(-C)[:8]
                Xt = bestX.copy()
                Xt[hi] += 0.03 * rng.standard_normal((len(hi), bestX.shape[1]))
                # also re-optimise rattler if any
                C0, _, _ = contact_counts(bestX, tol=1e-9)
                r0idx = np.where(C0 == 0)[0]
                if len(r0idx) > 0:
                    mask = np.ones(len(Xt), dtype=bool)
                    mask[r0idx[0]] = False
                    y = best_hole_for_set(Xt[mask], rng, n_cand=12000)
                    Xt[r0idx[0]] = y
                Xt = norm_rows(Xt)
                Xt = lp_polish(Xt, min(2.8, time_left() - 11.0), r0=1e-3)
            else:
                Xt = soft_anneal(bestX, rng, steps=900, beta0=80.0, beta1=250.0, lr=0.015)
                if time_left() < 13.0:
                    continue
                Xt = lp_polish(Xt, min(2.8, time_left() - 11.0), r0=8e-4)
        except Exception:
            continue
        if Xt is None or not np.all(np.isfinite(Xt)):
            continue
        v = max_cos(norm_rows(Xt))
        if v < bestV - 1e-12:
            bestX = norm_rows(Xt)
            bestV = v
        # side-step: if equal within 1e-9 but different topology, occasionally keep
        # to diversify (deterministic rule)
        elif abs(v - bestV) < 5e-9 and (it % 11 == 0):
            # keep as new base if contact structure differs, but don't update bestV
            # do a short exploration from Xt next time by swapping base temporarily
            # implement by setting bestX to Xt for one iteration without losing true best
            # store true best separately
            pass
        if bestV < 0.160:
            break

    # final high-precision polish of best
    if time_left() > 5.0:
        bestX = lp_polish(bestX, time_left() - 1.5, r0=4e-4, rmax=0.02)
        bestV = max_cos(bestX)
    bestX = norm_rows(bestX)
    return bestX

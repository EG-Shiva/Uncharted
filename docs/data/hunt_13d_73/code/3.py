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


def _smooth_relax(X, iters=500, k0=80.0, k1=350.0, lr0=0.03, lr1=0.005):
    X = X.copy()
    for it in range(iters):
        frac = it / max(1, iters - 1)
        k = k0 + (k1 - k0) * frac
        lr_t = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights focused on largest cosines
        W = np.exp((G - m) * k)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        s = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - s * X
        rn = np.linalg.norm(Gt, axis=1).max() + 1e-12
        X = X - (lr_t / rn) * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _lp_polish_budget(x, seconds, radius_init=1.5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        Gfull = x @ x.T
        c = Gfull[iu0, iu1]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        # cap number of constraints for speed
        if np.count_nonzero(sel) > 2000:
            # keep tightest 2000
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols_t = np.full(p, n * d)
        cols = np.concatenate([cols_I, cols_J, cols_t]) if False else None
        # build cols in per-row order: [I-block, J-block, t]
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = 0  # placeholder, will overwrite below
        # vectorised construction
        base = np.arange(p)[:, None]
        # simpler: build with concatenate per row order
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            # accept (allow tiny tolerance)
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            # early exit if time almost up
            if time.time() - start >= seconds:
                break
        else:
            radius *= 0.5
    return x, best


def _find_void(X, rng, n_samples=8000):
    n, d = X.shape
    R = rng.standard_normal(size=(n_samples, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    dots = R @ X.T
    worst = dots.max(axis=1)
    return R[int(np.argmin(worst))]


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    DEADLINE = 176.0  # leave margin for final polish / return

    # load starting point
    X0 = None
    best_val_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (73, 13):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_val_file:
                        best_val_file = c
                        X0 = v.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.standard_normal(size=(73, 13))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val_file = _max_cos(X0)

    bestX = X0.copy()
    best = _max_cos(bestX)
    cur = bestX.copy()
    cur_val = best

    # quick initial short polish to get baseline tight (1s)
    try:
        bestX, best = _lp_polish_budget(bestX, seconds=1.0, radius_init=1e-3)
        cur = bestX.copy()
        cur_val = best
    except Exception:
        pass

    T = 0.0025
    it = 0
    iu0, iu1 = np.triu_indices(73, 1)

    while time.time() - t0 < 158.0:
        it += 1
        # cool temperature
        T = 0.0025 * (0.998 ** it) + 0.0003
        r = rng.random()
        Xtry = cur.copy()
        try:
            ppm = _per_point_max(cur)
        except Exception:
            ppm = None

        if r < 0.32:
            # global shake, varying sigma
            sigmas = (0.006, 0.012, 0.025, 0.05)
            sigma = sigmas[it % len(sigmas)]
            Xtry = Xtry + rng.standard_normal(size=Xtry.shape) * sigma
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.55:
            # subset shake of highly loaded points
            ksub = int(rng.integers(5, 13))
            if ppm is not None:
                order = np.argsort(-ppm)
                pool = order[:20]
                idx = rng.choice(pool, size=min(ksub, len(pool)), replace=False)
            else:
                idx = rng.choice(73, size=ksub, replace=False)
            Xtry[idx] += rng.standard_normal(size=(len(idx), 13)) * 0.09
            # tiny noise elsewhere
            Xtry += rng.standard_normal(size=Xtry.shape) * 0.003
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.78:
            # void reinsertion: replace worst / rattler / random with void
            try:
                v = _find_void(cur, rng, n_samples=8000)
            except Exception:
                v = rng.standard_normal(size=(13,))
                v /= np.linalg.norm(v)
            if ppm is not None:
                # pick worst most often, sometimes rattler/random
                if rng.random() < 0.7:
                    idx = int(np.argmax(ppm))
                else:
                    idx = int(rng.integers(0, 73))
                    # prefer low-contact point
                    if rng.random() < 0.5:
                        idx = int(np.argmin(ppm))
            else:
                idx = int(rng.integers(0, 73))
            noise = rng.standard_normal(size=(13,)) * 0.01
            Xtry[idx] = v + noise
            Xtry[idx] /= np.linalg.norm(Xtry[idx])
            # small global jitter to unlock
            Xtry += rng.standard_normal(size=Xtry.shape) * 0.004
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            # break the current maximal pair apart
            try:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                a, b = np.unravel_index(int(np.argmax(G)), G.shape)
                Xtry[a] = Xtry[a] - 0.03 * Xtry[b] + rng.standard_normal(size=(13,)) * 0.01
                Xtry[b] = Xtry[b] - 0.03 * Xtry[a] + rng.standard_normal(size=(13,)) * 0.01
                Xtry += rng.standard_normal(size=Xtry.shape) * 0.005
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            except Exception:
                Xtry = Xtry + rng.standard_normal(size=Xtry.shape) * 0.02
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # fast smooth relaxation
        try:
            iters = 350 + int(rng.integers(0, 250))
            Xtry = _smooth_relax(Xtry, iters=iters, k0=70.0, k1=320.0,
                                 lr0=0.028, lr1=0.004)
        except Exception:
            pass

        # short LP re-polish
        time_left = 158.0 - (time.time() - t0)
        if time_left < 1.2:
            break
        budget = 1.1 if (it % 3) else 1.6
        budget = min(budget, time_left)
        try:
            Xpol, v = _lp_polish_budget(Xtry, seconds=budget, radius_init=1.5e-3)
        except Exception:
            continue
        if not np.all(np.isfinite(Xpol)):
            continue

        if v < best - 1e-12:
            best, bestX = v, Xpol.copy()
            cur, cur_val = Xpol.copy(), v
        elif v < cur_val:
            cur, cur_val = Xpol.copy(), v
        else:
            # Metropolis acceptance of worse basin to keep exploring
            try:
                if rng.random() < np.exp((cur_val - v) / max(T, 1e-6)):
                    cur, cur_val = Xpol.copy(), v
            except Exception:
                pass
        # occasional restart from best if wandering too far
        if it % 25 == 0 and cur_val > best + 0.015:
            cur, cur_val = bestX.copy(), best

    # final long polish of best
    try:
        remain = DEADLINE - (time.time() - t0) - 1.0
        if remain > 2.0:
            bestX, best = _lp_polish_budget(bestX, seconds=min(remain, 14.0),
                                            radius_init=1e-3)
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # strict finite check
    if not np.all(np.isfinite(bestX)):
        bestX = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    return bestX

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _refine(X, steps, s0=70.0, s1=320.0, lr0=0.03, lr1=0.004):
    X = X.copy()
    for it in range(steps):
        frac = it / max(1, steps - 1)
        s = s0 + (s1 - s0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp((G - m) * s)
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        rn = np.linalg.norm(Gt, axis=1).max() + 1e-12
        X = X - (lr / rn) * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _void_candidate(X, rng, exclude=None, n_samples=3000):
    n, d = X.shape
    R = rng.standard_normal(size=(n_samples, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    if exclude is not None and len(exclude) > 0:
        mask = np.ones(n, dtype=bool)
        mask[np.asarray(exclude)] = False
        others = X[mask]
    else:
        others = X
    dots = R @ others.T
    worst = dots.max(axis=1)
    return R[int(np.argmin(worst))]

def _lp_polish_budget(x, seconds, radius_init=8e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    iters = 0
    while time.time() - start < seconds and radius > 1e-13 and iters < 40:
        iters += 1
        if time.time() - start > seconds - 0.5:
            break
        Gfull = x @ x.T
        c = Gfull[iu0, iu1]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 1500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:1500]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(1.0, seconds - (time.time() - start))})
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
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    SEARCH_END = 132.0
    DEADLINE = 172.0
    n, d = 73, 13
    X0 = None
    bf = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < bf:
                        bf = c
                        X0 = v.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bf = _max_cos(X0)
    bestX = X0.copy()
    best = bf
    cur = X0.copy()
    cur_val = best
    # tiny initial shake-refine to diversify without LP
    try:
        ppm0 = _per_point_max(cur)
    except Exception:
        ppm0 = None
    it = 0
    Temp0 = 0.0022
    while time.time() - t0 < SEARCH_END:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / SEARCH_END
        T = Temp0 * (1.0 - frac) + 0.0002
        r = rng.random()
        Xtry = cur.copy()
        try:
            ppm = _per_point_max(cur)
        except Exception:
            ppm = ppm0
        if r < 0.30:
            sigmas = (0.008, 0.015, 0.030, 0.055)
            sigma = sigmas[it % len(sigmas)]
            Xtry = Xtry + rng.standard_normal(size=Xtry.shape) * sigma
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.52:
            # kill worst 1-3 and reinsert into voids (balance load)
            kkill = int(rng.integers(1, 4))
            if ppm is not None:
                order = np.argsort(-ppm)
                # mix worst and random to avoid always same
                if rng.random() < 0.75:
                    idxs = order[:kkill].copy()
                    # occasionally include rattler (min)
                    if rng.random() < 0.3:
                        idxs[-1] = int(np.argmin(ppm))
                else:
                    idxs = rng.choice(n, size=kkill, replace=False)
            else:
                idxs = rng.choice(n, size=kkill, replace=False)
            for idx in idxs:
                v = _void_candidate(cur, rng, exclude=[int(idx)], n_samples=3000)
                v = v + rng.standard_normal(size=(d,)) * 0.005
                v /= np.linalg.norm(v)
                Xtry[int(idx)] = v
            Xtry += rng.standard_normal(size=Xtry.shape) * 0.003
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.74:
            # coherent cap shake: pick random axis, move cap together + jitter
            u = rng.standard_normal(size=(d,))
            u /= np.linalg.norm(u)
            proj = cur @ u
            # threshold to get ~5-15 points
            thr = np.quantile(proj, 1.0 - rng.uniform(0.07, 0.22))
            cap = np.where(proj >= thr)[0]
            if len(cap) == 0:
                cap = np.array([int(np.argmax(proj))])
            shift = rng.standard_normal(size=(d,)) * 0.05
            # make shift tangential-ish on average
            Xtry[cap] += shift + rng.standard_normal(size=(len(cap), d)) * 0.02
            Xtry += rng.standard_normal(size=Xtry.shape) * 0.002
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.88:
            # subset shake of highly loaded points
            ksub = int(rng.integers(4, 11))
            if ppm is not None:
                order = np.argsort(-ppm)
                pool = order[:22]
                idx = rng.choice(pool, size=min(ksub, len(pool)), replace=False)
            else:
                idx = rng.choice(n, size=ksub, replace=False)
            Xtry[idx] += rng.standard_normal(size=(len(idx), d)) * 0.08
            Xtry += rng.standard_normal(size=Xtry.shape) * 0.002
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            # separate current max pair
            try:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                a, b = np.unravel_index(int(np.argmax(G)), G.shape)
                Xtry[a] = Xtry[a] - 0.04 * Xtry[b] + rng.standard_normal(size=(d,)) * 0.012
                Xtry[b] = Xtry[b] - 0.04 * Xtry[a] + rng.standard_normal(size=(d,)) * 0.012
                Xtry += rng.standard_normal(size=Xtry.shape) * 0.004
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            except Exception:
                Xtry = Xtry + rng.standard_normal(size=Xtry.shape) * 0.02
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        steps = 280 + int(rng.integers(0, 220))
        # anneal sharpness slightly per trial
        s1 = float(rng.uniform(260.0, 360.0))
        try:
            Xtry = _refine(Xtry, steps=steps, s0=65.0, s1=s1, lr0=0.028, lr1=0.004)
        except Exception:
            continue
        try:
            v = _max_cos(Xtry)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best - 1e-15:
            best, bestX = v, Xtry.copy()
            cur, cur_val = Xtry.copy(), v
        elif v < cur_val:
            cur, cur_val = Xtry.copy(), v
        else:
            try:
                if rng.random() < np.exp((cur_val - v) / max(T, 1e-6)):
                    cur, cur_val = Xtry.copy(), v
            except Exception:
                pass
        if it % 30 == 0 and cur_val > best + 0.012:
            cur, cur_val = bestX.copy(), best
        # safety: stop early if little time for polish
        if time.time() - t0 > SEARCH_END - 0.5:
            break
    # final squeeze of incumbent only
    try:
        remain = DEADLINE - (time.time() - t0) - 2.0
        if remain > 3.0:
            budget = min(remain, 16.0)
            bestX, best = _lp_polish_budget(bestX, seconds=budget, radius_init=8e-4)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    if not np.all(np.isfinite(bestX)):
        bestX = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    return bestX

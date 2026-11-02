import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, r0=8e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        tol = max(20.0 * radius, 1e-9)
        sel = c_all > (m - tol)
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = 0  # placeholder
        # build via concatenation (clearer)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv),
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c_sel,
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        if np.max(np.abs(step)) < 1e-14:
            radius *= 0.5
            # still try
            if radius < 1e-12:
                break
            continue
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            # early exit if clearly improved a lot? continue
        else:
            radius *= 0.5
    return x


def _refine_void(Y, z0, rng, iters=700):
    z = z0 / np.linalg.norm(z0)
    cur = float((Y @ z).max())
    # exponential decay sigma 0.12 -> 0.004
    for it in range(iters):
        frac = it / max(1, iters - 1)
        sigma = 0.12 * (0.004 / 0.12) ** frac
        r = rng.normal(size=z.shape)
        r -= (r @ z) * z
        rn = np.linalg.norm(r)
        if rn < 1e-12:
            continue
        r /= rn
        step = sigma * abs(float(rng.normal()))
        zt = z + step * r
        nrm = np.linalg.norm(zt)
        if nrm < 1e-12:
            continue
        zt /= nrm
        v = float((Y @ zt).max())
        if v < cur:
            z, cur = zt, v
    return z, cur


def _find_insertion(Y, rng):
    d = Y.shape[1]
    Ncand = 6000
    C = rng.normal(size=(Ncand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    vals = (C @ Y.T).max(axis=1)
    idx = np.argpartition(vals, 8)[:8]
    best_z = C[idx[0]]
    best_v = float(vals[idx[0]])
    for k in idx:
        zr, vr = _refine_void(Y, C[k], rng, iters=600)
        if vr < best_v:
            best_v, best_z = vr, zr
    return best_z


def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    DEADLINE = t0 + 176.0
    # load starts
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (73, 13):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append(c)
    if not cands:
        c = rng.normal(size=(73, 13))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    bestX = min(cands, key=_max_cos)
    best = _max_cos(bestX)
    # short initial squeeze if time allows (already polished parent)
    # explore until reserve for final polish
    EXPLORE_END = t0 + 148.0
    trial = 0
    sig_sub_list = [0.03, 0.08, 0.15, 0.28, 0.45]
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 10:
        trial += 1
        remaining = min(EXPLORE_END - time.time(), DEADLINE - 10 - time.time())
        if remaining < 2.0:
            break
        if trial % 4 == 0:
            # remove-relax-reinsert on worst point
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            top5 = np.argsort(-per)[:5]
            rem = int(rng.choice(top5))
            Y = np.delete(bestX, rem, axis=0)
            t1 = min(2.2, remaining * 0.35)
            Yp = _lp_polish(Y, seconds=max(0.6, t1), r0=8e-3)
            z = _find_insertion(Yp, rng)
            Xcand = np.vstack([Yp, z[None, :]])
            t2 = min(2.8, remaining - t1 - 0.2)
            if t2 < 0.6:
                continue
            Xp = _lp_polish(Xcand, seconds=t2, r0=8e-3)
            v = _max_cos(Xp)
            if v < best - 1e-13:
                best, bestX = v, Xp
        else:
            # kick subset biased to worst points
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            n_sub = int(rng.integers(2, 9))
            sigma_sub = sig_sub_list[(trial + int(rng.integers(0, 5))) % len(sig_sub_list)]
            sigma_all = 0.004 if (trial % 2 == 0) else 0.010
            X = bestX.copy()
            X += rng.normal(scale=sigma_all, size=X.shape)
            idx = set(order[:2].tolist())
            # add rattler-like (low contacts) sometimes to move rattler
            # contacts under current best
            contacts = np.count_nonzero(G > (best - 1e-9), axis=1)
            ratt = np.where(contacts == 0)[0]
            for r in ratt:
                idx.add(int(r))
            while len(idx) < n_sub + len(ratt):
                idx.add(int(rng.integers(0, 73)))
            for i in idx:
                X[i] += rng.normal(scale=sigma_sub, size=X.shape[1])
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            tpol = min(3.2, remaining - 0.2)
            Xp = _lp_polish(X, seconds=max(0.8, tpol), r0=1e-2)
            v = _max_cos(Xp)
            if v < best - 1e-13:
                best, bestX = v, Xp
    # final squeeze with remaining time
    rem = DEADLINE - time.time() - 1.0
    if rem > 1.0:
        bestX = _lp_polish(bestX, seconds=min(rem, 25.0), r0=1e-3)
        best = _max_cos(bestX)
        # tiny shrink guard (no-op, keep finite & normalized)
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(bestX)):
        bestX = cands[0]
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

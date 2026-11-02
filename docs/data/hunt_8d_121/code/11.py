import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _adam(X, steps, k0, k1, lr0, deadline):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lk0, lk1 = np.log(k0), np.log(k1)
    b1p, b2p = 1.0, 1.0
    best = X.copy()
    bestv = _max_cos(X)
    for t in range(1, steps + 1):
        if (t & 31) == 0 and time.time() > deadline:
            break
        frac = t / steps
        k = float(np.exp(lk0 + (lk1 - lk0) * frac))
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (W @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = np.clip(step, -0.05, 0.05)
        # tangent projection of step
        step = step - np.sum(step * X, axis=1, keepdims=True) * X
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        if (t % 100 == 0) or (t == steps):
            cur = _max_cos(X)
            if cur < bestv:
                bestv = cur
                best = X.copy()
    return best, bestv


def _greedy(X, rng, n_try, n_points):
    n, d = X.shape
    X = X.copy()
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cur = float(G.max())
    for _ in range(2):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        improved = False
        for idx in order[:n_points]:
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Grest = float(G[np.ix_(mask, mask)].max())
            if Grest >= cur - 1e-12 and rowmax[idx] <= Grest:
                continue
            C = rng.standard_normal((n_try, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ X[mask].T
            bv = S.max(axis=1)
            j = int(np.argmin(bv))
            nglob = bv[j] if bv[j] > Grest else Grest
            if nglob < cur - 1e-12:
                X[idx] = C[j]
                cur = float(nglob)
                g = X @ X[idx]
                G[idx, :] = g
                G[:, idx] = g
                G[idx, idx] = -2.0
                improved = True
        if not improved:
            break
    return X, cur


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gf = x @ x.T
        c = Gf[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        idx = np.where(sel)[0]
        if idx.size == 0:
            radius *= 0.5
            continue
        if idx.size > 3000:
            o = np.argsort(-c[idx])[:3000]
            idx = idx[o]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 170.0
    deadline_final = t0 + BUDGET
    np.random.seed(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (121, 8):
        cands.append(xb)
    if xr is not None and xr.shape == (121, 8):
        cands.append(xr)
    if not cands:
        r0 = np.random.default_rng(0)
        q = r0.standard_normal((121, 8))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        cands = [q]
    Xbest = None
    best = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best:
            best = v
            Xbest = c.copy()
    # quick squeeze of starting best if time allows (short)
    # leave main loop time; do 2s initial LP
    if deadline_final - time.time() > 30:
        Xbest = _lp_polish(Xbest, time.time() + 2.0)
        best = _max_cos(Xbest)

    sigmas = [0.012, 0.030, 0.070, 0.020, 0.045, 0.010]
    lrs = [0.010, 0.012, 0.008, 0.011, 0.009, 0.007]
    k1s = [8000.0, 4000.0, 12000.0, 6000.0, 15000.0, 3000.0]
    trial = 0
    stall = 0
    while True:
        rem = deadline_final - time.time()
        if rem < 14.0:
            break
        mode = trial % 6
        sigma = sigmas[mode] * (1.5 if stall > 8 else 1.0)
        lr0 = lrs[mode]
        k1 = k1s[mode]
        rng = np.random.default_rng(20000 + trial)
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)

        X = Xbest.copy()
        if mode == 0:
            # global small noise, extra kick worst 10
            X = X + rng.standard_normal(X.shape) * sigma
            X[order[:10]] += rng.standard_normal((10, 8)) * sigma * 1.5
        elif mode == 1:
            # medium global
            X = X + rng.standard_normal(X.shape) * sigma
        elif mode == 2:
            # cap randomization: pick random direction, randomize cap
            u = rng.standard_normal(8)
            u /= np.linalg.norm(u)
            proj = Xbest @ u
            mcap = 8 + int(rng.integers(0, 5))
            cap = np.argsort(-proj)[:mcap]
            X[cap] = rng.standard_normal((mcap, 8)) * 0.6 + X[cap]
        elif mode == 3:
            # worst-point full resample 2-3 points
            nw = 2 + (trial % 2)
            worst = order[:nw]
            X[worst] = rng.standard_normal((nw, 8))
            X = X + rng.standard_normal(X.shape) * 0.008
        elif mode == 4:
            # large shake + moderate
            X = X + rng.standard_normal(X.shape) * sigma
            # also randomize single worst fully
            X[order[0]] = rng.standard_normal(8)
        else:
            X = X + rng.standard_normal(X.shape) * sigma
            X[order[:6]] += rng.standard_normal((6, 8)) * sigma * 2.0
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        steps = 2800 if rem > 60 else 1800
        dline = time.time() + min(rem - 8.0, 4.0)
        X, v = _adam(X, steps=steps, k0=30.0, k1=k1, lr0=lr0, deadline=dline)

        if v < best + 0.008:
            nt = 15000 if (trial % 2 == 0) else 6000
            npnts = 12 if (trial % 2 == 0) else 7
            X, v = _greedy(X, rng, n_try=nt, n_points=npnts)

        promising = (v < best + 0.003) or (v < best + 0.008 and (trial % 4 == 0))
        if promising:
            rem2 = deadline_final - time.time()
            if rem2 > 10:
                secs = min(3.5, rem2 - 8.0)
                if secs > 0.8:
                    X = _lp_polish(X, time.time() + secs)
                    v = _max_cos(X)

        if v < best - 1e-12:
            best = v
            Xbest = X.copy()
            stall = 0
        else:
            stall += 1
        trial += 1
        if trial > 80:
            break

    rem = deadline_final - time.time()
    if rem > 2.0:
        Xbest = _lp_polish(Xbest, time.time() + max(0.5, rem - 0.7))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

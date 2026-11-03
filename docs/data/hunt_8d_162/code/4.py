import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def smooth_polish(X, steps=250, beta=120.0, lr=0.02):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # log-sum-exp weights, focus on near-max
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        s = E.sum()
        if not np.isfinite(s) or s <= 0:
            break
        P = E / s
        Grad = 2.0 * (P @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        mx = np.linalg.norm(Gt, axis=1).max()
        if mx < 1e-14:
            break
        cur_lr = lr * (1.0 - 0.5 * it / steps)
        step = Gt / mx * cur_lr
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # sharpen
        if it == steps // 2:
            beta *= 1.5
    return X


def lp_polish(X, seconds, r0=0.005, rmax=0.05):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iu_I = iu0[0]
    iu_J = iu0[1]
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        c_full = G[iu_I, iu_J]
        m = float(c_full.max())
        # keep best of this polish in sync (X is always best)
        if m < best:
            best = m
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_full > thresh
        # cap size for speed
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            # keep top 2500
            idx = np.argpartition(-c_full, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        I = iu_I[sel]
        J = iu_J[sel]
        c = c_full[sel]
        p = cnt
        nv = n * d + 1
        # A_ub * [dx; t] <= -c  where row = xJ*dxI + xI*dxJ - t
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = 0  # placeholder, will overwrite blockwise below
        # build cols vectorised
        base = np.arange(p)[:, None]
        # faster: concatenate
        cols = np.concatenate([cols_I.reshape(p, d), cols_J.reshape(p, d), np.full((p, 1), n * d)], axis=1).ravel()
        # rows already repeat
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        Y = X + dx
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = max_cos(Y)
        if v < best - 1e-15:
            X = Y
            best = v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
        # early exit if time exceeded inside linprog
        if time.time() - start >= seconds:
            break
    return X


def hole_candidates(X_keep, k, rng, n_samp=20000):
    n_keep, d = X_keep.shape
    C = rng.randn(n_samp, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ X_keep.T
    scores = S.max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > 0.55:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d)
        v /= np.linalg.norm(v)
        picked.append(v)
    return np.array(picked)


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(7)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            x_best = np.load("record.npy").astype(float)
        except Exception:
            x_best = rng.randn(162, 8)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    n, d = x_best.shape
    # ensure 162x8
    if n != 162 or d != 8:
        # resize deterministically by resampling
        tmp = rng.randn(162, 8)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        x_best = tmp
        n, d = 162, 8
    best_val = max_cos(x_best)

    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = np.sum(G > m - 1e-9, axis=1)
        return G, m, cnt

    Gbest, mbest, cntbest = contact_info(x_best)

    trial = 0
    # main basin-hopping: perturb-from-best
    while time.time() - t0 < 142.0:
        trial += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 8.0:
            break
        mode = trial % 5
        y = x_best.copy()
        if mode == 0:
            # global small shake
            sigma = 0.008 + 0.012 * rng.rand()
            Nz = rng.randn(n, d)
            Nz -= np.sum(Nz * y, axis=1, keepdims=True) * y
            y = y + sigma * Nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_polish(y, steps=200, beta=100.0, lr=0.015)
            y = lp_polish(y, seconds=min(1.8, remain - 1.0), r0=0.003, rmax=0.03)
        elif mode == 1:
            # overload cluster shake
            order = np.argsort(-cntbest)
            # random among top 6
            h = order[rng.randint(0, min(6, n))]
            # neighbors: closest dots
            row = Gbest[h].copy()
            row[h] = -2
            neigh = np.argsort(-row)[:12]
            # keep those reasonably close
            idx = [h] + [j for j in neigh if row[j] > mbest - 0.06][:10]
            sigma = 0.06 + 0.08 * rng.rand()
            for j in idx:
                nz = rng.randn(d)
                nz -= float(nz @ y[j]) * y[j]
                nz /= (np.linalg.norm(nz) + 1e-12)
                y[j] = y[j] + sigma * nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_polish(y, steps=250, beta=120.0, lr=0.025)
            y = lp_polish(y, seconds=min(2.2, remain - 1.0), r0=0.005, rmax=0.05)
        elif mode == 2:
            # low/rattler hole re-insertion
            k = 2 + int(rng.randint(0, 2))  # 2-3
            low = np.argsort(cntbest)[:k]
            # keep rest
            mask = np.ones(n, dtype=bool)
            mask[low] = False
            X_keep = y[mask]
            new_pts = hole_candidates(X_keep, k, rng, n_samp=20000)
            # slight random tangent jitter on new points
            for a in range(k):
                nz = rng.randn(d)
                nz -= float(nz @ new_pts[a]) * new_pts[a]
                new_pts[a] = new_pts[a] + 0.01 * nz
                new_pts[a] /= np.linalg.norm(new_pts[a])
            y[low] = new_pts
            y = smooth_polish(y, steps=300, beta=120.0, lr=0.03)
            y = lp_polish(y, seconds=min(2.4, remain - 1.0), r0=0.006, rmax=0.05)
        elif mode == 3:
            # worst-edge break
            # find top pairs
            iu = np.triu_indices(n, 1)
            c = Gbest[iu]
            top = np.argpartition(-c, 30)[:30]
            pick = top[rng.randint(0, len(top))]
            i = int(iu[0][pick]); j = int(iu[1][pick])
            # strong kick on i,j + light on their neighbors
            for vtx, s in [(i, 0.18), (j, 0.18)]:
                nz = rng.randn(d)
                nz -= float(nz @ y[vtx]) * y[vtx]
                nz /= (np.linalg.norm(nz) + 1e-12)
                y[vtx] = y[vtx] + s * (0.7 + 0.6 * rng.rand()) * nz
            # neighbors
            for vtx in [i, j]:
                row = Gbest[vtx].copy()
                row[vtx] = -2
                neigh = np.argsort(-row)[:8]
                for u in neigh:
                    if row[u] > mbest - 0.03:
                        nz = rng.randn(d)
                        nz -= float(nz @ y[u]) * y[u]
                        y[u] = y[u] + 0.03 * nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_polish(y, steps=250, beta=130.0, lr=0.025)
            y = lp_polish(y, seconds=min(2.2, remain - 1.0), r0=0.005, rmax=0.05)
        else:
            # random subset shake
            k = 6 + int(rng.randint(0, 5))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.10 + 0.06 * rng.rand()
            for j in idx:
                nz = rng.randn(d)
                nz -= float(nz @ y[j]) * y[j]
                nz /= (np.linalg.norm(nz) + 1e-12)
                y[j] = y[j] + sigma * nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_polish(y, steps=250, beta=110.0, lr=0.02)
            y = lp_polish(y, seconds=min(2.0, remain - 1.0), r0=0.004, rmax=0.05)

        v = max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
            Gbest, mbest, cntbest = contact_info(x_best)
        # opportunistic time check
        if time.time() - t0 > 145.0:
            break

    # final long polish on best
    remain = BUDGET - (time.time() - t0)
    if remain > 5.0:
        # a touch of smooth + long LP
        x_best = smooth_polish(x_best, steps=300, beta=150.0, lr=0.008)
        x_best = lp_polish(x_best, seconds=min(remain - 1.0, 28.0), r0=0.002, rmax=0.03)
        best_val = max_cos(x_best)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - tol, axis=1)
    return m, cnt, G

def _smooth_adam(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for it in range(1, iters + 1):
        frac = it / iters
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        s = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - s * X
        b1p *= b1
        b2p *= b2
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m1 / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _topk_repulse(X, steps, K, lr):
    n, d = X.shape
    for _ in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # find threshold for top K
        flat = G[np.triu_indices(n, 1)]
        if K >= flat.size:
            thr = -2.0
        else:
            thr = np.partition(flat, -K)[-K]
        mask = G >= thr
        # gradient: for each i sum of X[j] over j in contact
        Grad = (mask.astype(float)) @ X
        s = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - s * X
        # normalize step per point
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        X = X - lr * Gt / nrm * np.sqrt(n)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, r0=2e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cI, J[:, None] * d + cI, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        Xbest = rng.standard_normal((116, 6))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    cur = _max_cos(Xbest)
    try:
        Xr = np.load("record.npy").astype(float)
        Xr = Xr / np.linalg.norm(Xr, axis=1, keepdims=True)
        vr = _max_cos(Xr)
        if vr < cur:
            Xbest = Xr
            cur = vr
    except Exception:
        pass
    n, d = Xbest.shape
    stall = 0
    it = 0
    # search phase
    while time.time() - t0 < 118.0:
        it += 1
        mode = it % 6
        m, cnt, G = _contacts(Xbest, tol=1e-7)
        Y = Xbest.copy()
        if mode == 0:
            sigma = 0.010 + 0.012 * rng.random()
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 1:
            # hub surgery: hubs + their neighbors
            order = np.argsort(-cnt)
            k = 6
            hubs = order[:k]
            # neighbors in contact
            S = set(hubs.tolist())
            for h in hubs:
                nbrs = np.where(G[h] > m - 2e-7)[0]
                for nb in nbrs:
                    S.add(int(nb))
            S = np.array(sorted(S), dtype=int)
            sig_big = 0.08 + 0.08 * rng.random()
            Y[S] = Y[S] + sig_big * rng.standard_normal((len(S), d))
            Y = Y + 0.003 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 2:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:5]
            Y[ratt] = rng.standard_normal((len(ratt), d))
            Y = Y + 0.003 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 3:
            # hole reinsertion for low-contact points
            ratt = np.where(cnt <= 2)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:5]
            perm = ratt[rng.permutation(len(ratt))]
            for r in perm:
                others = np.delete(Y, r, axis=0)
                C = 6000
                cand = rng.standard_normal((C, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ others.T
                mx = np.max(dots, axis=1)
                Y[r] = cand[np.argmin(mx)]
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 4:
            # worst-pair kick: push worst pairs apart
            iu0 = np.triu_indices(n, 1)
            c = G[iu0]
            idx = np.argpartition(-c, 24)[:24]
            pts = np.unique(np.concatenate([iu0[0][idx], iu0[1][idx]]))
            Y[pts] = Y[pts] + 0.05 * rng.standard_normal((len(pts), d))
            # also push partners: move each point away from its worst partner
            for k2 in idx[:8]:
                i = int(iu0[0][k2]); j = int(iu0[1][k2])
                diff = Y[i] - Y[j]
                diff /= (np.linalg.norm(diff) + 1e-12)
                Y[i] = Y[i] + 0.02 * diff
                Y[j] = Y[j] - 0.02 * diff
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        else:
            sigma = 0.05 + 0.05 * rng.random()
            if stall > 12:
                sigma = 0.12
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # choose smoothing schedule
        if mode in (2, 3):
            a0 = 60.0 + 20.0 * rng.random()
            a1 = 500.0 + 400.0 * rng.random()
            Y = _smooth_adam(Y, iters=1100, a0=a0, a1=a1, lr0=0.014, lr1=0.002)
        elif mode == 1:
            a0 = 40.0 + 20.0 * rng.random()
            a1 = 400.0 + 300.0 * rng.random()
            Y = _smooth_adam(Y, iters=900, a0=a0, a1=a1, lr0=0.012, lr1=0.002)
            Y = _topk_repulse(Y, steps=120, K=560, lr=0.003)
        elif mode == 4:
            a0 = 50.0
            a1 = 600.0 + 300.0 * rng.random()
            Y = _smooth_adam(Y, iters=800, a0=a0, a1=a1, lr0=0.012, lr1=0.0015)
            Y = _topk_repulse(Y, steps=150, K=600, lr=0.002)
        else:
            a0 = 40.0 + 30.0 * rng.random()
            a1 = 450.0 + 350.0 * rng.random()
            Y = _smooth_adam(Y, iters=750, a0=a0, a1=a1, lr0=0.011, lr1=0.002)
        v = _max_cos(Y)
        improved = False
        # probe promising basins with short LP
        thresh = cur + (0.002 if stall < 10 else 0.004)
        if v < thresh:
            budget = 0.6
            # leave time for final polish
            if time.time() - t0 > 110:
                budget = 0.35
            try:
                Y2 = _lp_polish(Y, seconds=budget, r0=2e-3)
            except Exception:
                Y2 = Y
            v2 = _max_cos(Y2)
            if v2 < cur - 1e-12:
                Xbest = Y2
                cur = v2
                stall = 0
                improved = True
            elif v < cur - 1e-12:
                # smoothing alone improved (LP failed) -> keep with quick LP later
                Xbest = Y
                cur = v
                stall = 0
                improved = True
        if not improved:
            if v < cur - 1e-12:
                Xbest = Y
                cur = v
                stall = 0
            else:
                stall += 1
        if time.time() - t0 > 117.5:
            break
    # final polish
    try:
        remain = 172.0 - (time.time() - t0)
        if remain > 10:
            # high-temp smoothing to tighten
            Xbest = _smooth_adam(Xbest, iters=700, a0=500.0, a1=1400.0, lr0=0.0035, lr1=0.0004)
            Xbest = _topk_repulse(Xbest, steps=100, K=560, lr=0.001)
            remain = 172.0 - (time.time() - t0)
        if remain > 6:
            budget = min(max(remain - 1.0, 5.0), 38.0)
            Xbest = _lp_polish(Xbest, seconds=float(budget), r0=1.5e-3)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

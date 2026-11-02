import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _anneal(X, schedule, mu=0.9):
    n, d = X.shape
    v = np.zeros_like(X)
    for beta, iters, lr0, lr1 in schedule:
        v.fill(0.0)
        inv_b = 12.0 / beta
        for it in range(iters):
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            curmax = float(np.max(G))
            thr = curmax - inv_b
            cmin = curmax - 0.03
            if thr < cmin:
                thr = cmin
            mask = G >= thr
            W = np.zeros_like(G)
            Gm = G[mask]
            # beta*(Gm-curmax) in [-12,0], safe
            W[mask] = np.exp(beta * (Gm - curmax))
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            tg = grad - dot * X
            mn = float(np.max(np.sqrt(np.sum(tg * tg, axis=1))))
            if mn < 1e-14:
                continue
            tg /= mn
            v = mu * v + (1.0 - mu) * tg
            frac = it / max(1, iters - 1)
            lr = lr0 + (lr1 - lr0) * frac
            X = X - lr * v
            X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X

def _greedy_pair(X, steps=120):
    n, d = X.shape
    best = _max_cos(X)
    for _ in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        cij = float(X[i] @ X[j])
        ti = X[j] - cij * X[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            continue
        ti /= ni
        tj = X[i] - cij * X[j]
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            continue
        tj /= nj
        improved = False
        for eps in (1.2e-3, 6e-4, 3e-4, 1.5e-4):
            vi = X[i] - eps * ti
            vi /= np.linalg.norm(vi)
            vj = X[j] - eps * tj
            vj /= np.linalg.norm(vj)
            Xi_old = X[i].copy()
            Xj_old = X[j].copy()
            X[i] = vi
            X[j] = vj
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            if m2 < m - 1e-12:
                m = m2
                if m2 < best:
                    best = m2
                improved = True
                break
            else:
                X[i] = Xi_old
                X[j] = Xj_old
        # if not improved continue to next worst pair (will find same)
        if not improved:
            # tiny random tangent nudge on i to break exact tie
            pass
    return X, best

def _lp_polish(x, seconds, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius0)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iu_r = iu0[0]
    iu_c = iu0[1]
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1.0:
            break
        Gfull = x @ x.T
        c = Gfull[iu_r, iu_c]
        m = float(np.max(c))
        tol = 20.0 * radius
        if tol < 1e-9:
            tol = 1e-9
        sel = c > m - tol
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu_r[sel]
        J = iu_c[sel]
        p = cnt
        if p > 6000:
            cs = c[sel]
            kp = np.argpartition(-cs, 6000)[:6000]
            I = I[kp]
            J = J[kp]
            p = 6000
            b_ub = -cs[kp]
        else:
            b_ub = -c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + ar, J[:, None] * d + ar,
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=np.zeros(n),
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
        vv = _max_cos(y)
        if vv < best:
            x = y
            best = vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(0)
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(b)
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(r)
    except Exception:
        pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            cc = np.asarray(cc, dtype=np.float64)
            if cc.shape != (453, 5):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.normal(size=(453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    n, d = bestX.shape

    sched_short = [
        (80.0, 110, 0.0035, 0.0010),
        (200.0, 110, 0.0020, 0.0006),
        (500.0, 110, 0.0010, 0.0003),
    ]
    sched_tiny = [
        (150.0, 80, 0.0020, 0.0007),
        (500.0, 80, 0.0009, 0.0003),
    ]

    sigmas = [0.015, 0.010, 0.007, 0.005, 0.003, 0.002, 0.0012, 0.0008, 0.0005]

    it = 0
    # reserve final polish
    HOP_END = t0 + 138.0
    while time.time() - HOP_END < 0:
        it += 1
        if time.time() - t0 > TIME_LIMIT - 35.0:
            break
        sigma = sigmas[it % len(sigmas)]
        mode = it % 4
        X = bestX.copy()
        # find worst points from best
        try:
            Gtmp = X @ X.T
            np.fill_diagonal(Gtmp, -2.0)
            per = np.max(Gtmp, axis=1)
            order = np.argsort(-per)
        except Exception:
            order = rng.permutation(n)
            per = np.zeros(n)
        if mode == 0:
            # shake top-2 worst
            k = 2
            idx = order[:k]
            X[idx] += sigma * rng.standard_normal((k, d))
            # + small global jitter
            X += (sigma * 0.15) * rng.standard_normal((n, d))
        elif mode == 1:
            # shake worst-1 harder + few random
            k = 1
            idx = order[:k]
            X[idx] += sigma * 1.2 * rng.standard_normal((k, d))
            ridx = rng.integers(0, n, size=4)
            X[ridx] += (sigma * 0.5) * rng.standard_normal((4, d))
        elif mode == 2:
            # re-insert worst point from random candidates
            w = int(order[0])
            M = 1500
            cand = rng.standard_normal((M, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # score = max dot with others
            dots = cand @ X.T  # M x n
            # ignore self index by setting col w to -2
            dots[:, w] = -2.0
            scores = np.max(dots, axis=1)
            bi = int(np.argmin(scores))
            # mix: sometimes second best for diversity
            if it % 7 == 0:
                part = np.argpartition(scores, 5)[:5]
                bi = int(rng.choice(part))
            X[w] = cand[bi]
            # small shake around it
            X[w] += (sigma * 0.3) * rng.standard_normal(d)
            # shake second worst a bit
            if n > 1:
                w2 = int(order[1])
                X[w2] += (sigma * 0.5) * rng.standard_normal(d)
        else:
            # global small shake + worst medium
            X += (sigma * 0.35) * rng.standard_normal((n, d))
            k = 2
            idx = order[:k]
            X[idx] += sigma * 0.8 * rng.standard_normal((k, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # anneal
        try:
            if mode == 2:
                X = _anneal(X, sched_short)
            else:
                X = _anneal(X, sched_tiny)
        except Exception:
            continue
        # greedy pair shave
        try:
            X, gv = _greedy_pair(X, steps=80)
        except Exception:
            gv = float("inf")
            try:
                gv = _max_cos(X)
            except Exception:
                continue
        try:
            vv = _max_cos(X)
        except Exception:
            continue
        if vv < bestV and np.all(np.isfinite(X)):
            bestV = vv
            bestX = X.copy()
            # quick squeeze when clearly improved
            if vv < bestV - 1e-12 or True:
                pass
            # opportunistic short LP if we have time and good gain
            # do it rarely to save time
            if it % 12 == 0 and (time.time() - t0 < TIME_LIMIT - 50.0):
                try:
                    XL, VL = _lp_polish(bestX.copy(), seconds=2.5, radius0=8e-4)
                    if VL < bestV:
                        bestV = VL
                        bestX = XL
                except Exception:
                    pass
    # final polish: greedy + LP
    try:
        bestX, _ = _greedy_pair(bestX.copy(), steps=150)
        bestV = _max_cos(bestX)
    except Exception:
        pass
    remain = TIME_LIMIT - (time.time() - t0) - 1.0
    if remain < 3.0:
        remain = 3.0
    if remain > 32.0:
        remain = 32.0
    try:
        XP, VP = _lp_polish(bestX.copy(), seconds=float(remain), radius0=1e-3)
        if VP < bestV and np.all(np.isfinite(XP)):
            bestX, bestV = XP, VP
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

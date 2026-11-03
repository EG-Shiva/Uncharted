import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def fast_relax(X, iters, lr0=0.02, margin0=0.004):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    best = max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        # soft-max weights
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                break
            continue
        Xn = cur - (lr / gn) * Gt
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        mn = max_cos(Xn)
        if mn < m:
            cur = Xn
            if mn < best:
                best = mn
                bestX = Xn.copy()
            # keep lr, slowly tighten margin
            if it % 200 == 0:
                margin *= 0.85
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    # restart with wider step if stuck
                    margin = max(margin, 5e-7)
                    if lr < 1e-6:
                        break
        # mild decay to stabilize
        # lr stays bounded
    return bestX, best


def lp_polish(x, seconds, deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bestX = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 3500:
            k = 3500
            part = np.argpartition(-csel, k - 1)[:k]
            I, J, csel = I[part], J[part], csel[part]
            p = k
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best


def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    # load start
    X0 = None
    best_val = 1e9
    bestX = None
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (219, 6):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((219, 6))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_val = max_cos(bestX)
    # Phase A: quick greedy relax
    try:
        Xg, vg = fast_relax(bestX, iters=1500, lr0=0.02, margin0=0.004)
        if vg < best_val:
            best_val, bestX = vg, Xg
    except Exception:
        pass
    # Phase B: basin hopping on worst points
    # use time up to ~100s
    HOP_END = t0 + 100.0
    trial = 0
    n, d = bestX.shape
    while time.time() < HOP_END and time.time() - t0 < LIMIT - 55:
        trial += 1
        # choose k and sigma deterministically via rng
        if trial % 7 == 0:
            k = n
            sigma = 0.0015
            idx = np.arange(n)
        else:
            k = int(rng.integers(3, 28))
            ls = float(rng.uniform(-3.0, -1.3))  # log10 sigma
            sigma = 10.0 ** ls
            # worst vs random
            try:
                G = bestX @ bestX.T
                np.fill_diagonal(G, -2.0)
                rmax = np.max(G, axis=1)
                if rng.random() < 0.75:
                    idx = np.argsort(rmax)[-k:]
                else:
                    idx = rng.choice(n, size=k, replace=False)
            except Exception:
                idx = rng.choice(n, size=k, replace=False)
        cand = bestX.copy()
        noise = rng.standard_normal((len(idx), d)) * sigma
        cand[idx] += noise
        cand[idx] /= np.linalg.norm(cand[idx], axis=1, keepdims=True)
        # small global jitter occasionally
        if trial % 5 == 0:
            cand += rng.standard_normal((n, d)) * (sigma * 0.15)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        try:
            Xr, vr = fast_relax(cand, iters=400, lr0=0.015, margin0=0.003)
        except Exception:
            continue
        if vr < best_val - 1e-12:
            best_val, bestX = vr, Xr
    # Phase C: final LP squeeze with remaining time
    remain = LIMIT - (time.time() - t0) - 2.0
    if remain > 5:
        try:
            Xl, vl = lp_polish(bestX, seconds=min(remain, 70.0), deadline=t0 + LIMIT - 1.0)
            if vl < best_val:
                best_val, bestX = vl, Xl
        except Exception:
            pass
    # final touch: one more short greedy if time left
    if time.time() - t0 < LIMIT - 3:
        try:
            Xg2, vg2 = fast_relax(bestX, iters=300, lr0=0.005, margin0=0.001)
            if vg2 < best_val:
                best_val, bestX = vg2, Xg2
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

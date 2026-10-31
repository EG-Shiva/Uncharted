import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        # cap size for speed
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            thr = np.sort(c)[-2500]
            sel = c >= thr
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1
            ).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel],
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_repel(x, iters=500):
    n, d = x.shape
    for k in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        beta = 30.0 + (500.0 - 30.0) * (k / max(1, iters - 1))
        # exp(beta*(G-m)) in [-inf,0]
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(np.sqrt((grad ** 2).sum(axis=1)).mean()) + 1e-12
        grad /= s
        step = 0.022 - 0.020 * (k / max(1, iters - 1))
        x = x - step * grad
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def degree_vec(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G > m - tol).sum(axis=1)
    return deg, m


def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)

    cands = []
    try:
        xb = np.load("best.npy")
        cands.append(np.asarray(xb, dtype=float))
    except Exception:
        pass
    try:
        xr = np.load("record.npy")
        cands.append(np.asarray(xr, dtype=float))
    except Exception:
        pass

    best = None
    best_val = float("inf")
    for c in cands:
        try:
            c = np.asarray(c, dtype=float)
            if c.shape != (116, 6):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            v = max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        best = rng.normal(size=(116, 6))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = best.shape

    it = 0
    # main basin hopping, reserve ~15s for final polish
    while time.time() - t0 < DEADLINE - 15.0:
        it += 1
        mode = it % 6
        try:
            if mode == 0:
                cand = best + rng.normal(0, 0.015, size=(n, d))
            elif mode == 1:
                cand = best.copy()
                idx = rng.choice(n, 6, replace=False)
                cand[idx] += rng.normal(0, 0.35, size=(6, d))
            elif mode == 2:
                cand = best.copy()
                idx = rng.choice(n, 12, replace=False)
                cand[idx] += rng.normal(0, 0.6, size=(12, d))
            elif mode == 3:
                cand = best.copy()
                deg, _ = degree_vec(best, tol=1e-7)
                ratt = np.where(deg == 0)[0]
                if len(ratt) == 0:
                    ratt = np.argsort(deg)[:5]
                # limit number to avoid too much time
                if len(ratt) > 13:
                    ratt = ratt[:13]
                for r in ratt:
                    others = np.delete(cand, r, axis=0)
                    trials = rng.normal(size=(2500, d))
                    trials /= np.linalg.norm(trials, axis=1, keepdims=True)
                    dots = trials @ others.T
                    scores = dots.max(axis=1)
                    cand[r] = trials[np.argmin(scores)]
                cand += rng.normal(0, 0.005, size=(n, d))
            elif mode == 4:
                cand = best.copy()
                deg, _ = degree_vec(best, tol=1e-7)
                worst = np.argsort(deg)[-6:]
                for r in worst:
                    others = np.delete(cand, r, axis=0)
                    trials = rng.normal(size=(2000, d))
                    trials /= np.linalg.norm(trials, axis=1, keepdims=True)
                    dots = trials @ others.T
                    scores = dots.max(axis=1)
                    cand[r] = trials[np.argmin(scores)]
                mask = np.ones(n, dtype=bool)
                mask[worst] = False
                cand[mask] += rng.normal(0, 0.01, size=(int(mask.sum()), d))
            else:
                cand = best.copy()
                idx = rng.choice(n, 16, replace=False)
                cand[idx] = rng.normal(size=(16, d))
                cand += rng.normal(0, 0.008, size=(n, d))
        except Exception:
            continue
        # normalize
        try:
            nn = np.linalg.norm(cand, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            cand = cand / nn
            if not np.all(np.isfinite(cand)):
                continue
        except Exception:
            continue
        # fast global spread
        try:
            cand = smooth_repel(cand, iters=450)
        except Exception:
            pass
        remain = DEADLINE - 15.0 - (time.time() - t0)
        if remain < 0.8:
            break
        sec = min(1.8, max(0.8, remain * 0.15 + 0.6))
        try:
            cand = lp_polish(cand, seconds=sec)
        except Exception:
            continue
        try:
            v = max_cos(cand)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = cand
            best_val = v
        # stop if time low
        if time.time() - t0 >= DEADLINE - 15.0:
            break

    remain = DEADLINE - (time.time() - t0)
    if remain > 2.5:
        try:
            best = lp_polish(best, seconds=min(remain - 1.0, 12.0))
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

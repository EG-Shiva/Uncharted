import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def fast_polish(X, steps, lr0, lr1, m0, m1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def lp_polish(x, seconds, deadline=None):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > seconds:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _degrees(X, mv, tol=0.005):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    deg = np.sum(G >= (mv - tol), axis=1).astype(float)
    return deg, G


def _find_holes(Xkeep, k, rng, nsamp=8000):
    d = Xkeep.shape[1]
    Z = rng.standard_normal((nsamp, d))
    Z = _norm(Z)
    D = Z @ Xkeep.T
    worst = D.max(axis=1)
    idx = np.argpartition(worst, k)[:k]
    # order by worst ascending
    idx = idx[np.argsort(worst[idx])]
    return Z[idx]


def solve():
    t0 = time.time()
    BUDGET = 178.0
    HOP_BUDGET = 142.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    it = 0
    no_imp = 0
    while time.time() - t0 < HOP_BUDGET:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / HOP_BUDGET
        # annealing temperature for worse-accept
        T = 0.002 * (0.005 ** frac)  # 0.002 -> 1e-5
        # reset wandering cur if too bad
        if cur_val > best_val + 0.012:
            cur = best.copy()
            cur_val = best_val

        deg, Gcur = _degrees(cur, cur_val, tol=0.006)
        w = (deg + 1.0) ** 2.0
        probs = w / w.sum()
        # per-point worst cosine for hole selection
        rowmax = Gcur.max(axis=1)

        Y = cur.copy()
        r = rng.random()
        if r < 0.24:
            # global shake (activates rattlers too)
            sig = float(rng.uniform(0.05, 0.14))
            Y += sig * rng.standard_normal(Y.shape)
        elif r < 0.48:
            # hub-block kick: break over-constrained core
            k = int(rng.integers(6, 13))
            sig = float(rng.uniform(0.22, 0.42))
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
            Y += 0.015 * rng.standard_normal(Y.shape)
        elif r < 0.64:
            # random-block full reinit (topology reset)
            k = int(rng.integers(8, 16))
            idx = rng.choice(n, k, replace=False)
            Z = rng.standard_normal((k, d))
            Z = _norm(Z)
            Y[idx] = Z
            Y += 0.01 * rng.standard_normal(Y.shape)
        elif r < 0.82:
            # worst-point hole reinsertion
            k = int(rng.integers(3, 7))
            worst_idx = np.argpartition(-rowmax, k)[:k]
            keep_mask = np.ones(n, bool)
            keep_mask[worst_idx] = False
            Xkeep = Y[keep_mask]
            holes = _find_holes(Xkeep, k, rng, nsamp=8000)
            Y[worst_idx] = holes
            # kick neighbours of worst slightly to open space
            Y += 0.02 * rng.standard_normal(Y.shape)
        else:
            # single huge hub kick + global jitter
            k = int(rng.integers(1, 3))
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += 0.45 * rng.standard_normal((k, d))
            Y += 0.012 * rng.standard_normal(Y.shape)
        Y = _norm(Y)

        # two-stage wide->narrow polish (wide activates rattlers)
        s1 = int(rng.integers(600, 1000))
        lr0a = float(rng.uniform(0.02, 0.05))
        lr1a = float(rng.uniform(0.002, 0.005))
        m0a = float(rng.uniform(0.08, 0.14))
        m1a = float(rng.uniform(0.02, 0.04))
        Y = fast_polish(Y, s1, lr0a, lr1a, m0a, m1a)

        s2 = int(rng.integers(1100, 1700))
        lr0b = float(rng.uniform(0.008, 0.016))
        lr1b = float(rng.uniform(0.0002, 0.0008))
        m0b = m1a
        m1b = float(rng.uniform(0.002, 0.0045))
        Y = fast_polish(Y, s2, lr0b, lr1b, m0b, m1b)
        Y = fast_polish(Y, 300, 0.0006, 0.00015, 0.002, 0.001)

        v = _max_cos(Y)
        if v < best_val - 1e-12:
            best = Y.copy()
            best_val = v
            cur = Y.copy()
            cur_val = v
            no_imp = 0
        else:
            no_imp += 1
            if v < cur_val:
                cur = Y.copy()
                cur_val = v
            else:
                # annealed worse accept
                if T > 0 and rng.random() < np.exp((cur_val - v) / max(T, 1e-12)):
                    cur = Y.copy()
                    cur_val = v
                # periodic restart from elite to avoid drift
                if no_imp % 40 == 0:
                    cur = best.copy()
                    cur_val = best_val
                    # big escape shake from best
                    cur += 0.08 * rng.standard_normal(cur.shape)
                    cur = _norm(cur)
                    cur_val = _max_cos(cur)
        if time.time() - t0 > HOP_BUDGET:
            break

    remain = BUDGET - (time.time() - t0)
    if remain > 6:
        # final trust-region squeeze, cap ~32s
        try:
            best = lp_polish(best, min(remain - 1.5, 32))
        except Exception:
            pass
    best = _norm(best)
    return best

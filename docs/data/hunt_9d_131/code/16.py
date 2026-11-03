import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth(X, steps, T0, T1, lr0, lr1, mom=0.85):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr = Gr - dot * X
        s = float(np.sum(W)) / n + 1e-12
        Gr = Gr / s
        V = mom * V + (1.0 - mom) * Gr
        X = X - lr * V
        X = _norm(X)
    return X

def _lp(X, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _mc(X)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gf = X @ X.T
        ca = Gf[iu0]
        m = float(ca.max())
        th = m - max(20 * radius, 1e-9)
        sel = ca > th
        if int(sel.sum()) == 0:
            break
        if int(sel.sum()) > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-ca[idx])[:4500]
            mk = np.zeros_like(sel)
            mk[idx[order]] = True
            sel = mk
        I = iu0[0][sel]; J = iu0[1][sel]; c = ca[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _mc(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.4:
            break
    return X

def _degrees(X, mv):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    C = G >= (mv - 1e-7)
    return C.sum(axis=1), G

def _find_holes(Xfix, k, rng, ns=12000):
    d = Xfix.shape[1]
    Z = rng.standard_normal((ns, d))
    Z = _norm(Z)
    D = Z @ Xfix.T
    worst = D.max(axis=1)
    picked = []
    # greedy diverse picking
    cur = worst.copy()
    for _ in range(k):
        idx = int(np.argmin(cur))
        picked.append(Z[idx].copy())
        # penalize closeness to picked
        dp = Z @ picked[-1]
        # worst to picked set increases -> avoid same hole
        cur = np.maximum(cur, dp * 0.999)
        cur[idx] = 1e9
    holes = []
    for z in picked:
        # smooth refine vs fixed core
        for _ in range(120):
            dots = Xfix @ z
            m = float(dots.max())
            W = np.exp((dots - m) / 0.008)
            W = W / (W.sum() + 1e-12)
            g = W @ Xfix
            g = g - float(np.dot(g, z)) * z
            z = z - 0.02 * g
            z = z / max(np.linalg.norm(z), 1e-12)
        holes.append(z)
    return np.array(holes)

def solve():
    t0 = time.time()
    n, d = 131, 9
    rng = np.random.default_rng(0)
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                s = _mc(a)
                if s < best_v:
                    best_v = s; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_v = _mc(best_x)
    x_best = best_x.copy(); s_best = best_v

    def probs_deg(X, mv):
        deg, _ = _degrees(X, mv)
        w = (deg.astype(float) + 1.0) ** 1.5
        return w / w.sum(), deg

    probs, deg = probs_deg(x_best, s_best)
    trial = 0
    HOP_END = t0 + 130.0
    while time.time() < HOP_END:
        trial += 1
        if trial % 10 == 1:
            probs, deg = probs_deg(x_best, s_best)
        mode = trial % 8
        Y = None
        try:
            if mode == 0:
                sig = 0.012 if (trial % 3) else 0.03
                if trial % 31 == 0:
                    sig = 0.20
                Y = _norm(x_best + sig * rng.standard_normal((n, d)))
                Y = _smooth(Y, 200, 0.02, 0.005, 0.014, 0.002)
                Y = _smooth(Y, 200, 0.005, 0.0015, 0.004, 0.001)
            elif mode == 1:
                Y = _norm(x_best + 0.022 * rng.standard_normal((n, d)))
                Y = _smooth(Y, 260, 0.015, 0.002, 0.012, 0.0015)
            elif mode == 2:
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y = x_best.copy()
                Y[idx] += 0.10 * rng.standard_normal((1, d))
                Y = _norm(Y)
                Y = _smooth(Y, 260, 0.015, 0.002, 0.012, 0.0015)
            elif mode == 3:
                k = 2 if (trial % 2 == 0) else 3
                idx = rng.choice(n, k, replace=False, p=probs)
                Y = x_best.copy()
                Y[idx] += 0.16 * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth(Y, 300, 0.018, 0.002, 0.012, 0.0015)
            elif mode == 4:
                # hub hole-swap: remove worst hubs, insert optimized holes
                order = np.argsort(-deg)
                k = 1 if (trial % 3) else 2
                rem = order[:k]
                keep = np.array([i for i in range(n) if i not in set(rem.tolist())])
                Xfix = x_best[keep]
                holes = _find_holes(Xfix, k, rng, ns=10000)
                Y = x_best.copy()
                Y[rem] = holes
                Y = _smooth(Y, 260, 0.012, 0.0018, 0.010, 0.0012)
            elif mode == 5:
                # rattler relocation
                ratt = np.where(deg == 0)[0]
                if len(ratt) == 0:
                    ratt = np.argsort(deg)[:2]
                k = min(len(ratt), 2 if (trial % 2 == 0) else 1)
                rem = rng.choice(ratt, k, replace=False) if len(ratt) >= k else np.argsort(deg)[:k]
                keep = np.array([i for i in range(n) if i not in set(np.atleast_1d(rem).tolist())])
                Xfix = x_best[keep]
                holes = _find_holes(Xfix, k, rng, ns=10000)
                Y = x_best.copy()
                Y[np.atleast_1d(rem)] = holes
                Y = _smooth(Y, 220, 0.010, 0.0015, 0.008, 0.001)
            elif mode == 6:
                # random-subspace shake + hub kick
                Y = x_best.copy() + 0.012 * rng.standard_normal((n, d))
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.12 * rng.standard_normal((1, d))
                Y = _norm(Y)
                Y = _smooth(Y, 280, 0.016, 0.002, 0.012, 0.0012)
            else:
                # subset shake 12 pts
                k = 12
                idx = rng.choice(n, k, replace=False)
                Y = x_best.copy()
                Y[idx] += 0.05 * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth(Y, 260, 0.015, 0.002, 0.011, 0.0012)
            v = _mc(Y)
            if v < s_best:
                # squeeze promising basin briefly
                Y2 = _lp(Y, min(time.time() + 2.0, HOP_END + 4.0))
                v2 = _mc(Y2)
                if v2 < s_best:
                    x_best, s_best = Y2, v2
                    probs, deg = probs_deg(x_best, s_best)
                elif v < s_best:
                    # keep raw if LP failed to improve but raw is better
                    pass
            elif v < s_best + 0.002 and (trial % 13 == 0):
                # occasional lateral move to change topology reference
                Y2 = _lp(Y, time.time() + 1.0)
                v2 = _mc(Y2)
                if v2 < s_best:
                    x_best, s_best = Y2, v2
                    probs, deg = probs_deg(x_best, s_best)
        except Exception:
            continue
        if HOP_END - time.time() < 1.0:
            break
    # final squeeze
    try:
        x_best = _lp(x_best, t0 + 175.0)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

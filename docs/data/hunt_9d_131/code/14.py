import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degs(X, mv):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return (G >= (mv - 1e-7)).sum(axis=1)

def _smooth(X, steps, T0, T1, lr0, lr1, mom=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
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

def _hole_for(Xcore, rng, ns=12000):
    d = Xcore.shape[1]
    Z = rng.standard_normal((ns, d))
    Z = _norm(Z)
    D = Z @ Xcore.T
    w = D.max(axis=1)
    i = int(np.argmin(w))
    z = Z[i].copy()
    # refine hole: minimize LSE to core
    T = 0.008
    lr = 0.03
    for _ in range(150):
        dots = Xcore @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W = W / (W.sum() + 1e-12)
        g = W @ Xcore
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
        T *= 0.995
    return z

def _lp(X, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gf = X @ X.T
        call = Gf[iu0]
        m = float(call.max())
        th = m - max(20.0 * radius, 1e-9)
        sel = call > th
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-call[idx])[:4500]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[0][sel]; J = iu0[1][sel]; c = call[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.4:
            break
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 131, 9
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_val:
                    best_val = s; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    x_best = best.copy(); s_best = best_val

    # helper probs
    def probs_from(X, mv):
        deg = _degs(X, mv).astype(float)
        w = (deg + 1.0) ** 1.6
        return w / w.sum(), deg

    probs, degs = probs_from(x_best, s_best)
    rattlers = np.where(degs < 0.5)[0]

    it = 0
    HOP_END = t0 + 132.0
    noimp = 0
    while time.time() < HOP_END:
        it += 1
        if it % 12 == 1:
            probs, degs = probs_from(x_best, s_best)
            rattlers = np.where(degs < 0.5)[0]
        Y = x_best.copy()
        mode = it % 8
        boost = min(2.0, 1.0 + noimp / 25.0)
        if mode == 0:
            Y = Y + 0.010 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 260, 0.018, 0.004, 0.016, 0.003)
            Y = _smooth(Y, 200, 0.004, 0.0015, 0.004, 0.001)
        elif mode == 1:
            Y = Y + 0.025 * boost * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 300, 0.020, 0.005, 0.016, 0.003)
            Y = _smooth(Y, 200, 0.005, 0.0015, 0.004, 0.001)
        elif mode == 2:
            k = 2
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += 0.12 * boost * rng.standard_normal((k, d))
            Y = _norm(Y)
            Y = _smooth(Y, 350, 0.02, 0.003, 0.018, 0.002)
        elif mode == 3:
            k = 3
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += 0.18 * boost * rng.standard_normal((k, d))
            Y = _norm(Y)
            Y = _smooth(Y, 350, 0.022, 0.003, 0.018, 0.002)
        elif mode == 4:
            # relocate 1-2 rattlers to holes
            core_idx = np.where(degs > 0.5)[0]
            Xcore = x_best[core_idx] if len(core_idx) > 0 else x_best
            nr = 2 if len(rattlers) >= 2 else 1
            rep = rng.choice(rattlers, nr, replace=False) if len(rattlers) >= nr else rng.choice(n, nr, replace=False)
            for r in rep:
                z = _hole_for(Xcore, rng, ns=10000)
                Y[r] = z + 0.005 * rng.standard_normal(d)
            Y = _norm(Y)
            # keep relocated fixed a bit? just smooth all
            Y = _smooth(Y, 350, 0.02, 0.004, 0.015, 0.0025)
            Y = _smooth(Y, 200, 0.004, 0.0012, 0.004, 0.001)
        elif mode == 5:
            # relocate 3 rattlers to random + hole refine
            core_idx = np.where(degs > 0.5)[0]
            Xcore = x_best[core_idx] if len(core_idx) > 0 else x_best
            if len(rattlers) >= 1:
                nr = min(3, len(rattlers))
                rep = rng.choice(rattlers, nr, replace=False)
                for r in rep:
                    if rng.random() < 0.5:
                        Y[r] = rng.standard_normal(d)
                    else:
                        Y[r] = _hole_for(Xcore, rng, ns=8000)
                Y = _norm(Y)
            else:
                Y = Y + 0.015 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
            Y = _smooth(Y, 350, 0.02, 0.003, 0.016, 0.002)
        elif mode == 6:
            k = 1
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += 0.35 * boost * rng.standard_normal((k, d))
            Y = _norm(Y)
            Y = _smooth(Y, 300, 0.025, 0.005, 0.020, 0.003)
            Y = _smooth(Y, 200, 0.005, 0.0015, 0.005, 0.001)
        else:
            Y = Y + 0.012 * rng.standard_normal(Y.shape)
            if len(rattlers) > 0:
                r = rng.choice(rattlers, 1)[0]
                Y[r] += 0.10 * rng.standard_normal(d)
            idx = rng.choice(n, 1, replace=False, p=probs)
            Y[idx] += 0.10 * rng.standard_normal((1, d))
            Y = _norm(Y)
            Y = _smooth(Y, 300, 0.02, 0.004, 0.016, 0.0025)

        v = _max_cos(Y)
        # decide to LP-polish
        thresh = s_best + (0.004 if noimp < 30 else 0.008)
        if v < thresh and time.time() < HOP_END - 4:
            Y2 = _lp(Y, min(time.time() + 4.0, HOP_END))
            v2 = _max_cos(Y2)
            if v2 < s_best - 1e-12:
                x_best = Y2.copy(); s_best = v2
                probs, degs = probs_from(x_best, s_best)
                rattlers = np.where(degs < 0.5)[0]
                noimp = 0
                continue
            elif v2 < v:
                Y = Y2; v = v2
        if v < s_best - 1e-12:
            # direct win without LP (rare)
            x_best = _norm(Y); s_best = v
            noimp = 0
        else:
            noimp += 1

    # final squeeze with remaining time
    deadline = t0 + BUDGET - 1.0
    try:
        x_best = _lp(x_best, deadline)
    except Exception:
        pass
    x_best = _norm(x_best)
    # tiny shrink guard (strictly feasible, keep unit rows)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _degrees(x, tol=2e-4):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    deg = np.sum(g >= (m - tol), axis=1).astype(float)
    # strict contacts
    deg0 = np.sum(g >= (m - 1e-7), axis=1)
    return deg, deg0, m

def _smooth(x, steps, T0, T1, lr0, lr1, momentum=0.8):
    n, d = x.shape
    v = np.zeros_like(x)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights, avoid overflow
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(np.sum(W)) / n + 1e-12
        grad = grad / s
        v = momentum * v + (1.0 - momentum) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _lp_polish(x, deadline, start_radius=1e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = start_radius
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.3:
            break
    return x, best

def _find_hole(Xk, rng, n_samples=15000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=100, T=0.006, lr=0.02):
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W = W / (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
    return z

def solve():
    t0 = time.time()
    BUDGET = 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_val:
                    best_val = s
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best_x)
    x_best = best_x.copy()
    score_best = best_val

    def probs_from(X, m):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        deg = np.sum(G >= (m - 2e-4), axis=1).astype(float)
        w = (deg + 1.0) ** 1.6
        w = w / w.sum()
        # low-degree / rattler mask for reinsertion
        deg_strict = np.sum(G >= (m - 1e-7), axis=1)
        return w, deg, deg_strict

    probs, deg, deg_strict = probs_from(x_best, score_best)

    trial = 0
    lp_calls = 0
    while time.time() < HOP_END:
        trial += 1
        if trial % 12 == 1:
            probs, deg, deg_strict = probs_from(x_best, score_best)
        mode = trial % 8
        Y = x_best.copy()
        try:
            if mode == 0:
                Y = Y + 0.008 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, 350, 0.015, 0.002, 0.018, 0.003)
            elif mode == 1:
                Y = Y + 0.022 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, 400, 0.018, 0.002, 0.020, 0.003)
            elif mode == 2:
                k = 1
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += 0.12 * rng.standard_normal((k, d))
                Y += 0.006 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, 400, 0.015, 0.0018, 0.018, 0.0025)
            elif mode == 3:
                k = 2
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += 0.13 * rng.standard_normal((k, d))
                Y += 0.006 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, 400, 0.015, 0.0018, 0.018, 0.0025)
            elif mode == 4:
                k = 3
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += 0.18 * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth(Y, 450, 0.020, 0.002, 0.022, 0.003)
            elif mode == 5:
                # hole re-insertion for rattler / low-degree point
                # pick a low-degree index
                order = np.argsort(deg_strict + 0.1 * rng.random(n))
                idx = int(order[0])
                # hole w.r.t. all other points
                Xk = np.delete(x_best, idx, axis=0)
                z, _ = _find_hole(Xk, rng, n_samples=12000)
                z = _refine_hole(z, Xk, steps=80, T=0.006, lr=0.02)
                Y[idx] = z + 0.005 * rng.standard_normal(d)
                # small shake of neighbours
                Y += 0.004 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, 400, 0.012, 0.0015, 0.015, 0.002)
            elif mode == 6:
                Y = Y + 0.045 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth(Y, 500, 0.022, 0.002, 0.022, 0.003)
            else:
                k = 1
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += 0.38 * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth(Y, 450, 0.020, 0.002, 0.020, 0.003)

            v = _max_cos(Y)
            # immediate win without LP
            if v < score_best - 1e-12:
                # squeeze with LP
                dl = min(time.time() + 5.0, HOP_END + 4.0)
                if dl - time.time() > 0.6:
                    Y2, v2 = _lp_polish(Y, dl)
                    lp_calls += 1
                    if v2 < score_best:
                        x_best = Y2.copy()
                        score_best = v2
                        probs, deg, deg_strict = probs_from(x_best, score_best)
                    elif v < score_best:
                        x_best = Y.copy()
                        score_best = v
                else:
                    x_best = Y.copy()
                    score_best = v
            elif v < score_best + 0.004 and (trial % 6 == 0):
                # opportunistic short LP on promising basin
                if HOP_END - time.time() > 8.0:
                    dl = time.time() + 1.2
                    Y2, v2 = _lp_polish(Y, dl)
                    lp_calls += 1
                    if v2 < score_best - 1e-12:
                        x_best = Y2.copy()
                        score_best = v2
                        probs, deg, deg_strict = probs_from(x_best, score_best)
        except Exception:
            continue
        if BUDGET - (time.time() - t0) < 35.0:
            break

    # final squeeze
    try:
        deadline = t0 + BUDGET - 1.0
        x_best, score_best = _lp_polish(x_best, deadline, start_radius=1e-3)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

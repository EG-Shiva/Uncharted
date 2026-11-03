import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-16)


def _smooth_refine(x, steps, T0, T1, lr0, lr1, mom=0.85):
    n = x.shape[0]
    v = np.zeros_like(x)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 + (T1 - T0) * f
        if lr0 > 0 and lr1 > 0:
            lr = lr0 * ((lr1 / lr0) ** f)
        else:
            lr = lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        scale = float(np.sum(W)) / n + 1e-12
        grad /= scale
        v = mom * v + (1.0 - mom) * grad
        x = x - lr * v
        x = _norm(x)
    return x


def _refine_hole(z, Xk, steps=150, T=0.006, lr=0.03):
    for _ in range(steps):
        dots = Xk @ z
        m = float(np.max(dots))
        w = np.exp((dots - m) / T)
        w /= (np.sum(w) + 1e-12)
        g = w @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z / nz
    return z


def _find_hole(Xk, rng, n_samples=15000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = np.max(D, axis=1)
    idx = int(np.argmin(worst))
    z = Z[idx].copy()
    z = _refine_hole(z, Xk, steps=150, T=0.006, lr=0.03)
    return z


def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() > deadline - 0.3:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel)
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best_x = None
    best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_s:
                    best_s = s
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_s = _max_cos(best_x)
    x_best = best_x.copy()
    s_best = float(best_s)

    def degrees(X, mv, tol=0.002):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.sum(G >= (mv - tol), axis=1)

    def probs_from_deg(deg):
        w = (deg.astype(float) + 1.0) ** 1.6
        return w / (np.sum(w) + 1e-12)

    deg = degrees(x_best, s_best)
    probs = probs_from_deg(deg)

    HOP_END = t0 + 134.0
    trial = 0
    while time.time() < HOP_END:
        trial += 1
        if trial % 10 == 1:
            deg = degrees(x_best, s_best)
            probs = probs_from_deg(deg)
        mode = trial % 8
        Y = None
        try:
            if mode == 0:
                sig = 0.010
                Y = x_best + sig * rng.standard_normal(x_best.shape)
                Y = _norm(Y)
            elif mode == 1:
                sig = 0.024
                Y = x_best + sig * rng.standard_normal(x_best.shape)
                Y = _norm(Y)
            elif mode == 2:
                k, sig = 1, 0.12
                idx = rng.choice(n, k, replace=False, p=probs)
                Y = x_best.copy()
                Y[idx] += sig * rng.standard_normal((k, d))
                Y = _norm(Y)
            elif mode == 3:
                k, sig = 2, 0.12
                idx = rng.choice(n, k, replace=False, p=probs)
                Y = x_best.copy()
                Y[idx] += sig * rng.standard_normal((k, d))
                Y = _norm(Y)
            elif mode == 4:
                # rebirth single worst hub into largest hole
                deg2 = degrees(x_best, s_best)
                worst = int(np.argmax(deg2))
                keep = np.delete(x_best, worst, axis=0)
                z = _find_hole(keep, rng, n_samples=15000)
                z = z + 0.002 * rng.standard_normal(d)
                z /= (np.linalg.norm(z) + 1e-16)
                Y = x_best.copy()
                Y[worst] = z
            elif mode == 5:
                # rebirth 2-3 worst hubs
                deg2 = degrees(x_best, s_best)
                order = np.argsort(-deg2)
                k = 2 if (trial % 2 == 0) else 3
                victims = order[:k].copy()
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                keep = x_best[keep_mask].copy()
                Y = x_best.copy()
                for v_i in victims:
                    z = _find_hole(keep, rng, n_samples=12000)
                    z = z + 0.003 * rng.standard_normal(d)
                    z /= (np.linalg.norm(z) + 1e-16)
                    Y[v_i] = z
                    keep = np.vstack([keep, z[None, :]])
            elif mode == 6:
                k, sig = 1, 0.40
                idx = rng.choice(n, k, replace=False, p=probs)
                Y = x_best.copy()
                Y[idx] += sig * rng.standard_normal((k, d))
                Y = _norm(Y)
            else:
                Y = x_best + 0.012 * rng.standard_normal(x_best.shape)
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.14 * rng.standard_normal((1, d))
                Y = _norm(Y)

            # annealed smooth polish, randomized hypers
            T0 = float(rng.uniform(0.014, 0.026))
            T1 = float(rng.uniform(0.003, 0.007))
            lr0 = float(rng.uniform(0.010, 0.024))
            lr1 = float(rng.uniform(0.0008, 0.003))
            mom = float(rng.uniform(0.80, 0.90))
            steps1 = int(rng.integers(550, 950))
            if mode in (5, 6):
                steps1 = int(steps1 * 1.4)
            Y = _smooth_refine(Y, steps1, T0, T1, lr0, lr1, mom)
            Y = _smooth_refine(Y, 350, 0.003, 0.0012, 0.003, 0.0006, 0.85)
            v = _max_cos(Y)
            if v < s_best - 1e-12:
                # lock in with quick LP
                dl = time.time() + 1.2
                if dl < t0 + 165.0:
                    Y = _lp_polish(Y, dl)
                    v = _max_cos(Y)
                if v < s_best:
                    x_best = Y.copy()
                    s_best = float(v)
                    deg = degrees(x_best, s_best)
                    probs = probs_from_deg(deg)
            elif v < s_best + 0.002 and (trial % 3 == 0 or v < s_best + 0.0004):
                # near miss: LP may push below
                if time.time() < HOP_END - 2.0:
                    dl = time.time() + 0.7
                    Y2 = _lp_polish(Y.copy(), dl)
                    v2 = _max_cos(Y2)
                    if v2 < s_best - 1e-12:
                        x_best = Y2.copy()
                        s_best = float(v2)
                        deg = degrees(x_best, s_best)
                        probs = probs_from_deg(deg)
        except Exception:
            continue
        if time.time() > HOP_END:
            break

    remain = (t0 + 178.0) - time.time()
    if remain > 4.0:
        try:
            x_best = _lp_polish(x_best, time.time() + min(remain - 1.0, 32.0))
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

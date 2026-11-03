import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _degrees(X, mv, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    C = G >= (mv - tol)
    deg = C.sum(axis=1).astype(float)
    return deg, G

def _smooth_refine(x, steps, T0, T1, lr0, lr1, mom=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(np.sum(W)) / n + 1e-12
        grad = grad / s
        v = mom * v + (1.0 - mom) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _find_hole(Xk, rng, n_samples=12000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=100, T=0.008, lr=0.03):
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

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
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
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    HOP_END = t0 + 132.0
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
    score_best = best_s

    def probs_from(X, mv):
        deg, _ = _degrees(X, mv, tol=1e-7)
        w = (deg + 1.0) ** 1.5
        return w / w.sum(), deg

    probs, degs = probs_from(x_best, score_best)

    trial = 0
    while time.time() < HOP_END:
        trial += 1
        if trial % 10 == 1:
            probs, degs = probs_from(x_best, score_best)
        mode = trial % 8
        Y = x_best.copy()
        try:
            if mode in (0, 1):
                # hole reinsertion: replace 1-2 rattlers/loose points by optimized holes
                deg_thr, _ = _degrees(Y, score_best, tol=5e-6)
                loose = np.where(deg_thr <= 2)[0]
                if len(loose) == 0:
                    loose = np.argsort(deg_thr)[:3]
                k = 1 if mode == 0 else min(2, len(loose))
                # choose k victims: lowest degree
                order = np.argsort(deg_thr)
                victims = order[:k]
                # for each victim, find hole w.r.t. rest
                for vi in victims:
                    mask = np.ones(n, dtype=bool); mask[vi] = False
                    Xk = Y[mask]
                    z, _ = _find_hole(Xk, rng, n_samples=10000)
                    z = _refine_hole(z, Xk, steps=80, T=0.008, lr=0.03)
                    Y[vi] = z
                # small jitter to neighbours to unlock
                Y += 0.004 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth_refine(Y, 350, 0.016, 0.003, 0.016, 0.003)
                Y = _smooth_refine(Y, 250, 0.005, 0.0015, 0.005, 0.0012)
            elif mode in (2, 3):
                k = 1 if mode == 2 else 2
                sig = 0.18 if mode == 2 else 0.28
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += sig * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth_refine(Y, 400, 0.018, 0.003, 0.015, 0.0025, mom=0.85)
                Y = _smooth_refine(Y, 200, 0.004, 0.0012, 0.004, 0.001)
            elif mode == 4:
                k, sig = 3, 0.16
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += sig * rng.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth_refine(Y, 380, 0.02, 0.004, 0.014, 0.0025)
            elif mode == 5:
                Y += 0.010 * rng.standard_normal(Y.shape)
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.14 * rng.standard_normal((1, d))
                Y = _norm(Y)
                Y = _smooth_refine(Y, 400, 0.016, 0.0025, 0.014, 0.002)
            elif mode == 6:
                Y += 0.025 * rng.standard_normal(Y.shape)
                Y = _norm(Y)
                Y = _smooth_refine(Y, 350, 0.022, 0.005, 0.015, 0.003)
                Y = _smooth_refine(Y, 250, 0.005, 0.0015, 0.005, 0.0012)
            else:
                # strong global shake of a random subspace + hole tweak
                Y += 0.012 * rng.standard_normal(Y.shape)
                # random rotation in a 2D subspace for a hub
                idx = int(rng.choice(n, 1, replace=False, p=probs)[0])
                rp = rng.standard_normal(d); rp /= np.linalg.norm(rp)
                ang = float(rng.uniform(0.15, 0.5))
                # rotate Y[idx] toward rp
                v = Y[idx].copy()
                v_perp = rp - float(rp @ v) * v
                nv_ = np.linalg.norm(v_perp)
                if nv_ > 1e-9:
                    v_perp /= nv_
                    Y[idx] = np.cos(ang) * v + np.sin(ang) * v_perp
                Y = _norm(Y)
                Y = _smooth_refine(Y, 400, 0.018, 0.003, 0.013, 0.002)
        except Exception:
            continue
        try:
            s_smooth = _max_cos(Y)
        except Exception:
            continue
        # quick trigger: if promising, short LP to see true basin value
        if s_smooth < score_best + 0.002:
            # don't spend too much: 1.2s
            dl = min(time.time() + 1.2, HOP_END + 4.0)
            # only if we have time left for hops
            try:
                Y2 = _lp_polish(Y, dl)
                s2 = _max_cos(Y2)
            except Exception:
                s2 = s_smooth + 1.0
                Y2 = Y
            if s2 < score_best - 1e-12:
                x_best = Y2.copy()
                score_best = s2
                probs, degs = probs_from(x_best, score_best)
                # save progress if possible (not required)
            elif s_smooth < score_best - 1e-12:
                # smoothed alone improved (LP failed) -> keep smoothed
                x_best = Y.copy()
                score_best = s_smooth
                probs, degs = probs_from(x_best, score_best)
        # timeout guard
        if time.time() > t0 + BUDGET - 35:
            break

    # final squeeze with remaining time
    try:
        deadline = t0 + BUDGET - 1.0
        x_best = _lp_polish(x_best, deadline)
    except Exception:
        pass
    x_best = _norm(x_best)
    # safety: remove NaN
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    # renormalize after nan fix
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    x_best = x_best / nr
    return x_best

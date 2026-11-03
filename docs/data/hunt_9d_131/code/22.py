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

def _smooth_polish(X, steps, T0, T1, lr0, lr1, mom=0.82):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # exp weights, clip for stability
        E = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        A = (E @ X) / rs
        dot = np.sum(A * X, axis=1, keepdims=True)
        F = A - dot * X
        # normalize scale
        sc = float(np.mean(rs)) + 1e-12
        # F already averaged; momentum
        V = mom * V + (1.0 - mom) * F
        X = X - lr * V
        X = _norm(X)
    return X

def _find_hole(Xk, rng, n_samples=12000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=150):
    T0, T1 = 0.012, 0.002
    lr = 0.05
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        dots = Xk @ z
        m = float(dots.max())
        w = np.exp((dots - m) / max(T, 1e-12))
        w = w / (w.sum() + 1e-12)
        g = w @ Xk
        g = g - float(np.dot(g, z)) * z
        gn = np.linalg.norm(g) + 1e-12
        # normalized step
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
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.4:
            break
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(2)
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

    # cached degree probs
    probs = None
    degs = None
    def refresh_probs(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G >= (mv - 1e-7)
        deg = C.sum(axis=1).astype(float)
        w = (deg + 1.0) ** 1.5
        return w / w.sum(), deg, G

    probs, degs, _ = refresh_probs(x_best, score_best)

    HOP_END = t0 + 138.0
    trial = 0
    while time.time() < HOP_END:
        trial += 1
        if trial % 8 == 1:
            probs, degs, _ = refresh_probs(x_best, score_best)
        r = rng.random()
        Y = None
        if r < 0.32:
            # global shake
            if trial % 11 == 0:
                sigma = 0.035
            elif trial % 3 == 0:
                sigma = 0.015
            else:
                sigma = 0.008
            Y = x_best + sigma * rng.standard_normal((n, d))
            Y = _norm(Y)
        elif r < 0.62:
            # hub kick + light global noise
            Y = x_best.copy()
            k = int(rng.integers(1, 4))
            sig = float(rng.uniform(0.08, 0.28))
            idx = rng.choice(n, size=k, replace=False, p=probs)
            Y[idx] += sig * rng.standard_normal((k, d))
            Y += 0.004 * rng.standard_normal((n, d))
            Y = _norm(Y)
        else:
            # delete worst k and reinsert into holes
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            imax = G.max(axis=1)
            order = np.argsort(-imax)
            if r < 0.82:
                k = 1
            elif r < 0.95:
                k = 2
            else:
                k = 3
            # pick among top-12 worst for diversity
            top = order[:12]
            # always include the absolute worst with high prob
            sel_idx = [int(order[0])]
            if k > 1:
                cand = [t for t in top if t != int(order[0])]
                extra = rng.choice(cand, size=k-1, replace=False)
                for e in extra:
                    sel_idx.append(int(e))
            sel_idx = np.array(sel_idx, dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[sel_idx] = False
            Xkeep = x_best[keep_mask]
            new_pts = []
            Xcur_base = Xkeep
            for ti in range(k):
                if new_pts:
                    Xcur = np.vstack([Xcur_base] + new_pts)
                else:
                    Xcur = Xcur_base
                z, _ = _find_hole(Xcur, rng, n_samples=12000)
                z = _refine_hole(z, Xcur, steps=150)
                # small random jitter to avoid exact symmetry
                z = z + 0.002 * rng.standard_normal(d)
                z = z / max(np.linalg.norm(z), 1e-12)
                new_pts.append(z.reshape(1, d))
            Y = x_best.copy()
            for ii, ni in enumerate(sel_idx):
                Y[ni] = new_pts[ii][0]
            # light noise on rest to break jamming
            Y[keep_mask] += 0.003 * rng.standard_normal((n - k, d))
            Y = _norm(Y)

        # randomized two-stage polish
        T0 = float(rng.uniform(0.015, 0.028))
        T1c = float(rng.uniform(0.004, 0.008))
        T1f = float(rng.uniform(0.0012, 0.0025))
        lr0 = float(rng.uniform(0.015, 0.030))
        lr1c = float(rng.uniform(0.003, 0.007))
        lr1f = float(rng.uniform(0.0008, 0.002))
        if r >= 0.62:
            s1, s2 = 320, 320
        else:
            s1, s2 = 260, 260
        # occasionally longer
        if trial % 17 == 0:
            s1, s2 = 500, 500
        Y = _smooth_polish(Y, s1, T0, T1c, lr0, lr1c, mom=0.82)
        Y = _smooth_polish(Y, s2, T1c, T1f, lr1c, lr1f, mom=0.88)
        v = _max_cos(Y)
        if v < score_best - 1e-12:
            x_best = Y
            score_best = v
            probs, degs, _ = refresh_probs(x_best, score_best)

        if time.time() > HOP_END:
            break

    remain = 178.0 - (time.time() - t0)
    if remain > 6:
        try:
            dl = time.time() + min(remain - 1.5, 32.0)
            x_best = _lp_polish(x_best, dl)
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

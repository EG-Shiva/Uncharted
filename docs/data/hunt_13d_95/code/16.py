import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def gd_adam(X, steps, T0, T1, lr0, lr1, seed_mom=None):
    n, d = X.shape
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        Mh = M / bc1
        Vh = V / bc2
        X = X - lr * Mh / (np.sqrt(Vh) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def best_hole_for(others, rng, n_samples=3000):
    d = others.shape[1]
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return C[j]


def lp_polish(x, seconds, t_deadline=None, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if time.time() - start > seconds - 0.05:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 2000:
            idx = np.argpartition(c_all, -2000)[-2000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    search_end = t0 + 128.0
    rng = np.random.default_rng(7)
    np.random.seed(7)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        z = rng.standard_normal((95, 13))
        cands.append(z / np.linalg.norm(z, axis=1, keepdims=True))

    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = best.copy()
    cur_val = best_val
    n, d = best.shape

    trial = 0
    stagn = 0
    while time.time() < search_end and time.time() < deadline - 40:
        trial += 1
        remain = search_end - time.time()
        if remain < 0.5:
            break
        mode = trial % 7
        Xt = cur.copy()
        if mode == 0:
            sig = 0.03
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_adam(Xt, 500, 0.008, 0.0008, 0.015, 0.002)
        elif mode == 1:
            sig = 0.08
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_adam(Xt, 600, 0.010, 0.0008, 0.018, 0.002)
        elif mode == 2:
            sig = 0.25
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_adam(Xt, 700, 0.014, 0.0010, 0.020, 0.002)
        elif mode == 3:
            ppm = per_point_max(cur)
            order = np.argsort(-ppm)
            k = int(rng.integers(4, 11))
            pool = order[:max(12, k + 4)]
            idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
            sig = float(rng.uniform(0.05, 0.18))
            Xt[idx] = Xt[idx] + sig * rng.standard_normal((len(idx), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_adam(Xt, 550, 0.009, 0.0007, 0.015, 0.002)
        elif mode == 4:
            ppm = per_point_max(cur)
            order = np.argsort(-ppm)
            if rng.random() < 0.6:
                k = int(rng.integers(1, 3))
            else:
                k = int(rng.integers(3, 6))
            worst = order[:max(k + 6, 10)]
            chosen = rng.choice(worst, size=k, replace=False)
            for ix in chosen:
                others = np.delete(Xt, ix, axis=0)
                Xt[ix] = best_hole_for(others, rng, n_samples=3000)
            Xt = gd_adam(Xt, 600, 0.009, 0.0007, 0.015, 0.002)
        elif mode == 5:
            ppm = per_point_max(cur)
            order = np.argsort(ppm)
            k = int(rng.integers(1, 4))
            quiet = order[:max(k + 6, 10)]
            chosen = rng.choice(quiet, size=k, replace=False)
            for ix in chosen:
                others = np.delete(Xt, ix, axis=0)
                Xt[ix] = best_hole_for(others, rng, n_samples=3000)
            # also shake a few jammed ones lightly
            ppm2 = per_point_max(cur)
            order2 = np.argsort(-ppm2)
            js = rng.choice(order2[:12], size=int(rng.integers(2, 5)), replace=False)
            Xt[js] = Xt[js] + 0.05 * rng.standard_normal((len(js), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_adam(Xt, 600, 0.009, 0.0007, 0.015, 0.002)
        else:
            sig = float(rng.uniform(0.02, 0.10))
            ksub = int(rng.integers(6, 20))
            idx = rng.choice(n, size=ksub, replace=False)
            Xt[idx] = Xt[idx] + sig * rng.standard_normal((ksub, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_adam(Xt, 500, 0.010, 0.0009, 0.016, 0.002)

        v = max_cos(Xt)
        if v < best_val - 1e-12:
            best = Xt.copy()
            best_val = float(v)
            cur = Xt.copy()
            cur_val = float(v)
            stagn = 0
        elif v < cur_val - 1e-12:
            cur = Xt.copy()
            cur_val = float(v)
            stagn = 0
        else:
            stagn += 1
            if rng.random() < 0.08 and v < cur_val + 0.008:
                cur = Xt.copy()
                cur_val = float(v)
            if stagn > 60:
                cur = best.copy()
                cur_val = best_val
                stagn = 0
                # random kick of best to diversify
                cur = cur + 0.02 * rng.standard_normal((n, d))
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cur = gd_adam(cur, 300, 0.008, 0.001, 0.012, 0.002)
                cur_val = max_cos(cur)
                if cur_val < best_val:
                    best = cur.copy()
                    best_val = cur_val

    # final squeeze
    time_left = deadline - time.time() - 1.0
    if time_left > 3.0:
        best = lp_polish(best, time_left, t_deadline=deadline - 0.5)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

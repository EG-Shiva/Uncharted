import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    t_end = time.time() + seconds
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns: I-block, J-block, t
        base_I = I[:, None] * d + np.arange(d)
        base_J = J[:, None] * d + np.arange(d)
        cols = np.concatenate([base_I, base_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _fast_polish(x, steps):
    n, d = x.shape
    v = np.zeros_like(x)
    mu = 0.85
    for s in range(steps):
        frac = s / max(1, steps)
        alpha = 15.0 + 85.0 * frac
        lr = 0.02 * (1.0 - frac) + 0.002
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        thr = m - 8.0 / alpha
        mask = G > thr
        W = np.zeros_like(G)
        # shifted exp, safe
        W[mask] = np.exp(alpha * (G[mask] - m))
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        tang = grad - dot * x
        tn = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        direction = tang / tn
        v = mu * v - lr * direction
        # clip velocity to avoid huge jumps
        vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        maxv = 3.0 * lr
        sc = np.minimum(1.0, maxv / vn)
        v = v * sc
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _refine_hole(X, h, steps=200):
    h = h / np.linalg.norm(h)
    for s in range(steps):
        dots = X @ h
        j = int(np.argmax(dots))
        dm = float(dots[j])
        gj = X[j] - dm * h
        lr = 0.1 * (1.0 - s / steps) + 0.01
        h = h - lr * gj
        nh = np.linalg.norm(h)
        if nh == 0 or not np.isfinite(nh):
            break
        h /= nh
    return h


def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 168.0
    EXPLORE_END = t0 + 120.0

    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    n, d = x_best.shape
    best_val = _max_cos(x_best)

    rng = np.random.default_rng(12345)

    def pick_worst(xc):
        G = xc @ xc.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        row_max = np.max(G, axis=1)
        contacts = np.sum(G > m - 1e-9, axis=1)
        order = np.lexsort((contacts, row_max))
        top = order[-10:]
        return int(rng.choice(top)), float(m)

    it = 0
    while time.time() < EXPLORE_END:
        it += 1
        mod = it % 7
        xc = x_best.copy()
        steps = 800
        if mod == 1:
            xc = xc + 0.008 * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 600
        elif mod == 2:
            xc = xc + 0.025 * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 800
        elif mod == 3:
            xc = xc + 0.07 * rng.standard_normal(xc.shape)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            steps = 1200
        elif mod == 4:
            idx, _ = pick_worst(xc)
            rv = rng.standard_normal(d)
            rv /= np.linalg.norm(rv)
            xc[idx] = rv
            steps = 800
        elif mod == 5:
            idx, _ = pick_worst(xc)
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xr = xc[mask]
            # sample candidates for hole
            S = rng.standard_normal((5000, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            dots = S @ Xr.T
            rowm = np.max(dots, axis=1)
            bi = int(np.argmin(rowm))
            h = _refine_hole(Xr, S[bi], steps=200)
            xc[idx] = h
            steps = 800
        else:  # mod 0 or 6 : subset shake
            k = 12
            sel = rng.choice(n, size=k, replace=False)
            xc[sel] = xc[sel] + 0.05 * rng.standard_normal((k, d))
            xc[sel] /= np.linalg.norm(xc[sel], axis=1, keepdims=True)
            steps = 800

        xc = _fast_polish(xc, steps)
        val_fast = _max_cos(xc)
        # screen: only LP-refine promising candidates
        if val_fast < best_val + 0.005:
            if time.time() >= EXPLORE_END:
                break
            sec = min(2.5, max(0.5, EXPLORE_END - (time.time())))
            if sec < 0.5:
                break
            xr = _lp_polish(xc, seconds=sec, r0=1.5e-3)
            vr = _max_cos(xr)
            if vr < best_val - 1e-12:
                x_best = xr
                best_val = vr

    remain = BUDGET - (time.time() - t0)
    if remain > 5:
        x_best = _lp_polish(x_best, seconds=remain - 2.0, r0=1e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

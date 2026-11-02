import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def refine_soft(x, steps, lr, T0, T1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lT = np.log(T1 / T0)
    for t in range(1, steps + 1):
        frac = t / steps
        T = T0 * np.exp(lT * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp((G - gmax) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        r = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - r * x
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * (grad * grad)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = mhat / (np.sqrt(vhat) + eps)
        rs = np.sum(step * x, axis=1, keepdims=True)
        step = step - rs * x
        cur_lr = lr * (1.0 - 0.5 * frac)
        x = x - cur_lr * step
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    t0 = time.time()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        mval = c.max()
        thr = mval - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            # keep top 2500
            csel_all = c[sel]
            kth = np.sort(csel_all)[-2500]
            sel = c >= kth
            # if still too many due to ties, subsample
            idx = np.where(sel)[0]
            if len(idx) > 3000:
                take = np.argsort(-c[idx])[:3000]
                mask = np.zeros_like(c, dtype=bool)
                mask[idx[take]] = True
                sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv),
            ).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(
                obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t_start = time.time()
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    # fallback if shapes wrong
    if xb.shape != (121, 8):
        # try record
        try:
            xb2 = np.load("record.npy").astype(float)
            if xb2.shape == (121, 8):
                xb = xb2
        except Exception:
            pass
    # if still wrong, random start
    if xb.shape != (121, 8):
        r = rng.standard_normal(size=(121, 8))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        xb = r
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    best_val = max_cos(xb)
    x_best = xb.copy()

    sigmas = [0.004, 0.007, 0.012, 0.02, 0.035, 0.006, 0.01, 0.016, 0.028, 0.05]
    it = 0
    # search phase until ~135s
    while time.time() - t_start < 135.0:
        it += 1
        if it % 6 == 0:
            # largest-hole reinsertion of a worst point
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            k = int(order[int(rng.integers(0, min(8, n)))])
            M = 3000
            cand = rng.standard_normal(size=(M, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ x_best.T
            dots[:, k] = -2.0
            scores = dots.max(axis=1)
            best_c = cand[int(np.argmin(scores))]
            y = x_best.copy()
            y[k] = best_c
            y = y + 0.002 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, lr, T0, T1 = 1800, 0.0025, 0.015, 0.0025
        else:
            sigma = float(sigmas[(it - 1) % len(sigmas)])
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            y = x_best.copy()
            noise = rng.standard_normal(size=y.shape)
            amp = np.ones((n, 1)) * (sigma * 0.5)
            amp[order[:10]] = sigma * 1.5
            y = y + amp * noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if sigma > 0.025:
                steps, lr, T0, T1 = 2200, 0.002, 0.02, 0.002
            else:
                steps, lr, T0, T1 = 1500, 0.002, 0.02, 0.002
        y = refine_soft(y, steps, lr, T0, T1)
        v = max_cos(y)
        if v < best_val - 1e-12:
            # squeeze new record a bit
            if time.time() - t_start < 120.0:
                y2 = lp_polish(y, seconds=1.2)
                v2 = max_cos(y2)
                if v2 < v:
                    y, v = y2, v2
            x_best, best_val = y, float(v)
        # safety break near budget
        if 178.0 - (time.time() - t_start) < 5.0:
            break

    remain = 178.0 - (time.time() - t_start)
    if remain > 3.0:
        x_best = lp_polish(x_best, seconds=remain - 1.0)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best = np.nan_to_num(x_best, nan=1.0 / np.sqrt(8), posinf=1.0, neginf=-1.0)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

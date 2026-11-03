import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_opt(x, iters=600, beta0=100.0, beta1=450.0, lr0=0.02, lr1=0.0015):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for k in range(iters):
        frac = k / (iters - 1) if iters > 1 else 1.0
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights focused on near-max pairs
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        tang = grad - dot * x
        tn = np.linalg.norm(tang, axis=1, keepdims=True)
        med = float(np.median(tn)) + 1e-12
        alpha = 0.5 * med
        step = lr * tang / (tn + alpha)
        x = x - step
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def best_hole(Y, rng, samples=3000):
    m, d = Y.shape
    C = rng.standard_normal((samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = Y @ C.T  # m x samples
    worst = dots.max(axis=0)
    return C[int(np.argmin(worst))]


def solve():
    t0 = time.time()
    # load starting point
    cand_list = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (123, 13):
                cand_list.append(a.astype(float))
        except Exception:
            pass
    rng_init = np.random.default_rng(1)
    if len(cand_list) == 0:
        X0 = rng_init.standard_normal((123, 13))
    else:
        vals = []
        for a in cand_list:
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            vals.append(max_cos(a))
        X0 = cand_list[int(np.argmin(vals))]
    best = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    best_val = max_cos(best)
    rng = np.random.default_rng(12345)
    n, d = best.shape

    def elapsed():
        return time.time() - t0

    it = 0
    # explore different basins, leave time for final polish
    while elapsed() < 135 and it < 2000:
        it += 1
        mode = it % 7
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        per_worst = G.max(axis=1)
        order = np.argsort(per_worst)  # ascending
        cur = best.copy()
        if mode == 0:
            sigma = 0.04
            cur = cur + sigma * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.10
            cur = cur + sigma * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 2:
            # shake worst hubs hard, rest light
            k = 8
            idx = order[-k:]
            noise = 0.015 * rng.standard_normal(cur.shape)
            noise[idx] = 0.25 * rng.standard_normal((k, d))
            cur = cur + noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 3:
            # randomize rattlers / low points
            k = 2
            idx = order[:k]
            rand_pts = rng.standard_normal((k, d))
            rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
            cur[idx] = rand_pts
            cur = cur + 0.02 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 4:
            # void reinsert 2 worst
            k = 2
            idx_rm = order[-k:]
            mask = np.ones(n, dtype=bool)
            mask[idx_rm] = False
            Y = cur[mask]
            new_pts = []
            for _ in range(k):
                v = best_hole(Y, rng, samples=3000)
                new_pts.append(v)
                Y = np.vstack([Y, v[None, :]])
            cur = Y  # n x d (same order approx)
            # shuffle back to original size: just use Y as candidate
            cur = cur + 0.01 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 5:
            k = 1 + int(rng.integers(0, 2))  # 1-2
            idx_rm = order[-3 - int(rng.integers(0, 3)):][:k] if n > 5 else order[-k:]
            # simpler: take k worst
            idx_rm = order[-k:]
            mask = np.ones(n, dtype=bool)
            mask[idx_rm] = False
            Y = cur[mask]
            for _ in range(k):
                v = best_hole(Y, rng, samples=3000)
                Y = np.vstack([Y, v[None, :]])
            cur = Y + 0.015 * rng.standard_normal(Y.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        else:
            sigma = 0.22
            cur = cur + sigma * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)

        # fast soft-max repulsion smoothing
        cur_s = smooth_opt(cur, iters=600, beta0=120.0, beta1=450.0, lr0=0.02, lr1=0.0012)
        v = max_cos(cur_s)
        if v < best_val - 1e-12:
            best = cur_s
            best_val = v
            # squeeze immediately
            if elapsed() < 128:
                lp_x = lp_polish(cur_s, 2.0)
                vlp = max_cos(lp_x)
                if vlp < best_val:
                    best, best_val = lp_x, vlp
            continue
        # promising -> short LP to reach true local optimum
        thresh = 0.0008 if (it % 2 == 0) else 0.0004
        if v < best_val + thresh and elapsed() < 132:
            # limit LP frequency slightly
            if (it % 3 == 0) or (v < best_val + 0.0004):
                lp_x = lp_polish(cur_s, 1.4)
                vlp = max_cos(lp_x)
                if vlp < best_val - 1e-12:
                    best, best_val = lp_x, vlp

    # final polish with remaining time
    remain = 180 - elapsed() - 3.0
    if remain > 1.0:
        remain = min(remain, 25.0)
        best = lp_polish(best, remain)
        best_val = max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

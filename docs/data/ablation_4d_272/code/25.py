import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def row_info(x, m=None):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    rm = G.max(axis=1)
    # contacts within 1e-4 of max
    cnt = np.sum(G > (m - 1e-4), axis=1)
    return rm, m, cnt, G


def lp_polish(x, seconds, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
        if time.time() - start >= seconds:
            break
    return x


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def subset_smooth(x, movable, steps, a0, a1, s0, s1, mu=0.7):
    movable = np.asarray(movable, dtype=int)
    M = len(movable)
    vel = np.zeros((M, 4))
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        Xm = x[movable]
        D = Xm @ x.T  # M x N
        # kill self dots
        # for each k, D[k, movable[k]] = -2
        D[np.arange(M), movable] = -2.0
        m = D.max()
        W = np.exp(alpha * (D - m))
        grad = W @ x  # M x 4
        dot = np.sum(grad * Xm, axis=1, keepdims=True)
        gt = grad - dot * Xm
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            Xn = Xm - (smax / nrm) * vel
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            x[movable] = Xn
    return x


def largest_hole_candidate(x, rng, n_cand=15000):
    C = rng.standard_normal((n_cand, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # chunk to save memory
    best_i = -1
    best_v = 1e9
    chunk = 3000
    dots_max = np.empty(n_cand)
    for s in range(0, n_cand, chunk):
        e = min(s + chunk, n_cand)
        D = C[s:e] @ x.T
        dots_max[s:e] = D.max(axis=1)
    i = int(np.argmin(dots_max))
    c = C[i].copy()
    # refine single point against fixed set
    Y = x
    vel = np.zeros(4)
    alpha = 120.0
    for it in range(150):
        dots = Y @ c
        m = dots.max()
        W = np.exp(alpha * (dots - m))
        g = W @ Y
        gt = g - np.dot(g, c) * c
        nn = np.linalg.norm(gt)
        if nn < 1e-14:
            break
        gt /= nn
        vel = 0.6 * vel + gt
        nn2 = np.linalg.norm(vel)
        if nn2 < 1e-14:
            break
        step = 0.02 * (1.0 - it / 150.0) + 0.002
        c = c - (step / nn2) * vel
        c /= np.linalg.norm(c)
    return c


def greedy_removal_index(x):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    order = np.argsort(-rm)[:min(24, n)]
    best_i = int(order[0])
    best_v = 1e9
    idx_all = np.arange(n)
    for cand in order:
        cand = int(cand)
        mask = idx_all != cand
        v = float(G[np.ix_(mask, mask)].max())
        if v < best_v:
            best_v = v
            best_i = cand
    return best_i, best_v


def optimize_small(x, seconds, t_end):
    # aggressive smooth + lp alternation respecting global deadline
    # two rounds
    if time.time() >= t_end:
        return x
    x = smooth_gd(x, steps=350, a0=60.0, a1=500.0, s0=0.06, s1=0.006)
    if time.time() >= t_end:
        return x
    x = lp_polish(x, seconds=min(seconds, max(0.5, t_end - time.time())))
    if time.time() >= t_end:
        return x
    x = smooth_gd(x, steps=250, a0=120.0, a1=700.0, s0=0.03, s1=0.004)
    if time.time() >= t_end:
        return x
    x = lp_polish(x, seconds=min(2.0, max(0.5, t_end - time.time())))
    return x


def grow_back(x_small, target_n, rng, t_end, best_x, best_v):
    x = x_small.copy()
    while x.shape[0] < target_n and time.time() < t_end - 1.0:
        c = largest_hole_candidate(x, rng)
        x = np.vstack([x, c.reshape(1, 4)])
        n = x.shape[0]
        new_idx = n - 1
        # 20 nearest neighbours (largest dots)
        dots = x[:-1] @ x[-1]
        nn = min(20, n - 1)
        neigh = np.argsort(-dots)[:nn]
        movable = np.concatenate([neigh, np.array([new_idx])])
        x = subset_smooth(x, movable, steps=90, a0=80.0, a1=500.0, s0=0.05, s1=0.005)
        # brief global smoothing
        x = smooth_gd(x, steps=60, a0=150.0, a1=500.0, s0=0.015, s1=0.003)
        x = lp_polish(x, seconds=min(0.9, max(0.3, (t_end - time.time()) / max(1, target_n - x.shape[0] + 1))))
        if x.shape[0] == target_n:
            v = max_cos(x)
            if v < best_v:
                best_v = v
                best_x = x.copy()
    return x, best_x, best_v


def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(7)

    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    cands = []
    if bx is not None:
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (272, 4):
            bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
            cands.append(bx)
    if rx is not None:
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (272, 4):
            rx = rx / np.linalg.norm(rx, axis=1, keepdims=True)
            cands.append(rx)
    if not cands:
        r = rng.standard_normal((272, 4))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        cands.append(r)
    best_x = min(cands, key=lambda z: max_cos(z)).copy()
    best_v = max_cos(best_x)
    base = best_x.copy()

    # ---- Path A1: delete 30 (mix high-contact + random) ----
    try:
        rm, m, cnt, G = row_info(base)
        n = 272
        K = 30
        n_high = 15
        order_high = np.argsort(-cnt - 1e-6 * rm)[:n_high]
        # random among remainder
        remaining = np.array([i for i in range(n) if i not in set(order_high.tolist())])
        rsel = rng.choice(remaining, size=K - n_high, replace=False)
        del_idx = np.concatenate([order_high, rsel])
        keep = np.array([i for i in range(n) if i not in set(del_idx.tolist())])
        x_small = base[keep].copy()
        # jitter small to leave basin
        x_small = x_small + 0.004 * rng.standard_normal(x_small.shape)
        x_small /= np.linalg.norm(x_small, axis=1, keepdims=True)
        x_small = optimize_small(x_small, seconds=3.0, t_end=t_end - 40)
        x_full, best_x, best_v = grow_back(x_small, 272, rng, t_end - 45, best_x, best_v)
    except Exception:
        pass

    # ---- Path A2: cap deletion (geographic void) K=25 ----
    try:
        if time.time() < t_end - 50:
            rm2, m2, cnt2, G2 = row_info(base)
            direction = rng.standard_normal(4)
            direction /= np.linalg.norm(direction)
            dots_dir = base @ direction
            K2 = 25
            del_idx2 = np.argsort(-dots_dir)[:K2]
            keep2 = np.array([i for i in range(272) if i not in set(del_idx2.tolist())])
            y_small = base[keep2].copy()
            y_small = y_small + 0.006 * rng.standard_normal(y_small.shape)
            y_small /= np.linalg.norm(y_small, axis=1, keepdims=True)
            y_small = optimize_small(y_small, seconds=3.0, t_end=t_end - 25)
            _, best_x, best_v = grow_back(y_small, 272, rng, t_end - 22, best_x, best_v)
    except Exception:
        pass

    # ---- Path B: reverse — add 12 dummies, optimise N>272, prune ----
    try:
        if time.time() < t_end - 20:
            z = best_x.copy() if max_cos(best_x) < max_cos(base) + 1e-12 else base.copy()
            D = 12
            for _ in range(D):
                c = largest_hole_candidate(z, rng, n_cand=8000)
                z = np.vstack([z, c.reshape(1, 4)])
            z = z + 0.003 * rng.standard_normal(z.shape)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            z = smooth_gd(z, steps=300, a0=60.0, a1=500.0, s0=0.05, s1=0.005)
            z = lp_polish(z, seconds=min(3.0, max(0.5, t_end - time.time() - 12)))
            # greedy prune back to 272
            while z.shape[0] > 272 and time.time() < t_end - 10:
                bi, _ = greedy_removal_index(z)
                z = np.delete(z, bi, axis=0)
            z = smooth_gd(z, steps=120, a0=150.0, a1=600.0, s0=0.02, s1=0.004)
            z = lp_polish(z, seconds=min(4.0, max(0.5, t_end - time.time() - 2)))
            v = max_cos(z)
            if v < best_v and z.shape[0] == 272:
                best_v = v
                best_x = z.copy()
    except Exception:
        pass

    # ---- final polish of best with remaining time ----
    try:
        rem = t_end - time.time()
        if rem > 1.0:
            best_x = lp_polish(best_x, seconds=min(rem - 0.5, 8.0))
            best_v = max_cos(best_x)
    except Exception:
        pass

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

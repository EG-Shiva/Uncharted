import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

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
        m = c_all.max()
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

def optimize_single(Y, c0, steps=120, s=0.03, mu=0.5):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        if np.count_nonzero(sel) == 0:
            g = Y[np.argmax(dots)]
        else:
            g = Y[sel].sum(axis=0)
        gt = g - np.dot(g, c) * c
        n = np.linalg.norm(gt)
        if n < 1e-14:
            break
        gt = gt / n
        vel = mu * vel + gt
        vn = np.linalg.norm(vel)
        if vn > 1e-14:
            c = c - (s / vn) * vel
            c = c / np.linalg.norm(c)
            s *= 0.997
    return c

def hole_move(x, rng, n_samples=8000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    i, j = divmod(k, n)
    Gi = G[i].copy(); Gi[j] = -2
    Gj = G[j].copy(); Gj[i] = -2
    move = i if Gi.max() > Gj.max() else j
    if rng.random() < 0.2:
        # random worst-ish point instead
        rm, _ = row_max(x)
        order = np.argsort(-rm)
        move = int(order[rng.integers(0, min(20, n))])
    Y = np.delete(x, move, axis=0)
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_c = None
    best_s = 2.0
    bs = 2000
    for a in range(0, n_samples, bs):
        Cb = C[a:a+bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = int(np.argmin(sm))
        if sm[k2] < best_s:
            best_s = float(sm[k2])
            best_c = Cb[k2].copy()
    best_c = optimize_single(Y, best_c, steps=150, s=0.03)
    y = x.copy()
    y[move] = best_c / np.linalg.norm(best_c)
    return y

def refine_worst(x, rng, num_points=4, steps=80):
    n, d = x.shape
    rm, _ = row_max(x)
    order = np.argsort(-rm)[:num_points]
    y = x.copy()
    for w in order:
        Y = np.delete(y, w, axis=0)
        c0 = y[w].copy()
        # small random tilt to break ties
        c0 = c0 + 0.002 * rng.standard_normal(4)
        c0 /= np.linalg.norm(c0)
        c0 = optimize_single(Y, c0, steps=steps, s=0.02)
        y[w] = c0 / np.linalg.norm(c0)
    return y

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    n, d = 272, 4
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
        try:
            bx = np.asarray(bx, dtype=np.float64)
            if bx.shape == (n, d):
                bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
                cands.append(bx)
        except Exception:
            pass
    if rx is not None:
        try:
            rx = np.asarray(rx, dtype=np.float64)
            if rx.shape == (n, d):
                rx = rx / np.linalg.norm(rx, axis=1, keepdims=True)
                cands.append(rx)
        except Exception:
            pass
    if len(cands) == 0:
        bx = rng.standard_normal((n, d))
        bx /= np.linalg.norm(bx, axis=1, keepdims=True)
        cands.append(bx)
    best_x = None
    best_v = 2.0
    for c in cands:
        v = max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    # quick initial short polish if time allows (already polished, keep tiny)
    # main basin-hopping
    it = 0
    sigmas_global = [0.004, 0.009, 0.018, 0.035, 0.07, 0.12]
    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        remain = BUDGET - 13.0 - (time.time() - t0)
        if remain <= 0.5:
            break
        # choose kick type
        mode = it % 6
        x0 = None
        if it % 23 == 0:
            # fresh random start for diversity (rare)
            x0 = rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = 500
            a0, a1 = 12.0, 180.0
            s0, s1 = 0.05, 0.004
        elif mode == 3 or mode == 5:
            # hole reinsertion from best
            try:
                x0 = hole_move(best_x, rng, n_samples=7000)
            except Exception:
                x0 = best_x.copy()
            steps = 260
            a0 = float(rng.uniform(15, 30))
            a1 = float(rng.uniform(100, 220))
            s0 = float(rng.uniform(0.02, 0.04))
            s1 = float(rng.uniform(0.003, 0.008))
        elif mode == 1:
            # worst-subset strong kick
            try:
                rm, mcur = row_max(best_x)
                order = np.argsort(-rm)
                k = int(rng.integers(6, 28))
                worst = order[:k]
            except Exception:
                worst = np.array([], dtype=int)
            sigma_big = float(rng.uniform(0.05, 0.16))
            sigma_small = float(rng.uniform(0.002, 0.008))
            noise = rng.standard_normal((n, d)) * sigma_small
            if len(worst) > 0:
                noise[worst] = rng.standard_normal((len(worst), d)) * sigma_big
            x0 = best_x + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(220, 380))
            a0 = float(rng.uniform(12, 30))
            a1 = float(rng.uniform(90, 220))
            s0 = float(rng.uniform(0.02, 0.05))
            s1 = float(rng.uniform(0.003, 0.01))
        else:
            # global kick
            sg = sigmas_global[(it - 1) % len(sigmas_global)] * float(rng.uniform(0.7, 1.3))
            # rattler extra
            try:
                rm, mcur = row_max(best_x)
                ratt = np.where(rm < mcur - 5e-4)[0]
            except Exception:
                ratt = np.array([], dtype=int)
            noise = rng.standard_normal((n, d)) * sg
            if len(ratt) > 0 and rng.random() < 0.6:
                # boost rattlers
                noise[ratt] += rng.standard_normal((len(ratt), d)) * (sg * 1.5)
            x0 = best_x + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(200, 350))
            a0 = float(rng.uniform(12, 30))
            a1 = float(rng.uniform(80, 200))
            s0 = float(rng.uniform(0.015, 0.045))
            s1 = float(rng.uniform(0.002, 0.009))
        mu = float(rng.uniform(0.6, 0.9))
        # smooth repulsion
        try:
            x1 = smooth_gd(x0, steps, a0, a1, s0, s1, mu=mu)
        except Exception:
            continue
        # worst-point local refinement
        try:
            if rng.random() < 0.7:
                x1 = refine_worst(x1, rng, num_points=int(rng.integers(2, 6)), steps=60)
        except Exception:
            pass
        try:
            v1 = max_cos(x1)
        except Exception:
            continue
        # decide to LP polish
        # threshold: allow worse but not too worse
        thresh = best_v + (0.010 if it < 30 else 0.006)
        # fresh starts need stricter
        if it % 23 == 0:
            thresh = best_v + 0.004
        if v1 < thresh:
            # allocate LP time: more if promising
            if v1 < best_v:
                lp_sec = min(4.0, max(1.5, remain * 0.15))
            elif v1 < best_v + 0.002:
                lp_sec = min(2.5, max(1.0, remain * 0.08))
            else:
                lp_sec = 1.2
            # don't LP if almost no time left for loop
            if remain - lp_sec < 2.0 and v1 >= best_v:
                continue
            try:
                x2 = lp_polish(x1, lp_sec)
                v2 = max_cos(x2)
            except Exception:
                continue
            if v2 < best_v - 1e-12:
                best_x = x2.copy()
                best_v = float(v2)
                # early exit if big win? keep going to improve further
            elif v2 < v1 - 1e-9 and rng.random() < 0.1:
                # occasionally keep non-improving but improved-by-LP as new base for diversity?
                pass
        # opportunistic: if smooth alone beats best (rare without LP), keep
        if v1 < best_v - 1e-12:
            # quick LP to lock it
            try:
                x2 = lp_polish(x1, 2.0)
                v2 = max_cos(x2)
                if v2 < best_v:
                    best_x = x2.copy()
                    best_v = float(v2)
                else:
                    best_x = x1.copy()
                    best_v = float(v1)
            except Exception:
                best_x = x1.copy()
                best_v = float(v1)
        if time.time() - t0 >= BUDGET - 13.0:
            break
    # final long polish of best
    try:
        remain = BUDGET - (time.time() - t0) + 13.0  # ~13s left + margin to 180
        # leave ~1s margin
        final_sec = max(2.0, min(12.0, 178.0 - (time.time() - t0) - 1.0))
        if final_sec > 0.5:
            best_x = lp_polish(best_x, final_sec)
            best_v = max_cos(best_x)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

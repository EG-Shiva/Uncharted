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


def lp_polish(x, seconds):
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
        if np.count_nonzero(sel) > 5000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 5000 - 1)[:5000]
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


def optimize_single(Y, c0, steps=150, s=0.02, mu=0.5):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 1e-3
        if np.count_nonzero(sel) == 0:
            sel = np.argmax(dots)
            g = Y[sel]
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
            c /= np.linalg.norm(c)
            s *= 0.995
    return c


def hole_reinsert(x, rng, n_samples=12000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    # worst pair
    k = np.argmax(G)
    i, j = divmod(k, n)
    # choose which endpoint to move: larger second-max stays
    Gi = G[i].copy()
    Gj = G[j].copy()
    # second max
    Gi[j] = -2
    Gj[i] = -2
    move = i if Gi.max() > Gj.max() else j
    if rng.random() < 0.25:
        move = i if rng.random() < 0.5 else j
    Y = np.delete(x, move, axis=0)
    # sample holes
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # batch to save memory
    best_c = None
    best_s = 2.0
    bs = 3000
    for a in range(0, n_samples, bs):
        Cb = C[a:a + bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = np.argmin(sm)
        if sm[k2] < best_s:
            best_s = sm[k2]
            best_c = Cb[k2].copy()
    # local refine single point
    best_c = optimize_single(Y, best_c, steps=200, s=0.03)
    y = x.copy()
    y[move] = best_c / np.linalg.norm(best_c)
    return y


def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(7)
    n, d = 272, 4
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    cands0 = []
    if bx is not None:
        try:
            bx = np.asarray(bx, dtype=np.float64)
            if bx.shape == (n, d):
                bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
                cands0.append(bx)
        except Exception:
            pass
    if rx is not None:
        try:
            rx = np.asarray(rx, dtype=np.float64)
            if rx.shape == (n, d):
                rx = rx / np.linalg.norm(rx, axis=1, keepdims=True)
                cands0.append(rx)
        except Exception:
            pass
    if not cands0:
        bx = rng.standard_normal((n, d))
        bx /= np.linalg.norm(bx, axis=1, keepdims=True)
        cands0.append(bx)
    best_x = min(cands0, key=max_cos).copy()
    best_v = max_cos(best_x)

    # quick LP touch on start if time allows (very short)
    # skip heavy polish now; do hopping first

    sigmas = [0.004, 0.010, 0.022, 0.045, 0.090, 0.16, 0.30, 0.015]
    gd_presets = [
        (400, 30.0, 250.0, 0.020, 0.003, 0.80),
        (350, 80.0, 500.0, 0.012, 0.002, 0.85),
        (500, 15.0, 120.0, 0.025, 0.004, 0.75),
        (300, 200.0, 800.0, 0.008, 0.0015, 0.85),
    ]
    it = 0
    # main basin hopping; reserve 16s for final polish
    while time.time() - t0 < BUDGET - 18.0:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        r = rng.random()
        base = best_x
        cand = base.copy()
        if it % 6 == 0:
            # hole reinsertion move
            try:
                cand = hole_reinsert(base, rng, n_samples=12000)
            except Exception:
                cand = base.copy()
            # small jitter others
            noise = rng.standard_normal(cand.shape) * 0.003
            cand = cand + noise
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        elif it % 6 == 5:
            # contact breaking: pick active pair, push apart
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m = G.max()
                iu = np.triu_indices(n, 1)
                act = np.where(G[iu] > m - 2e-4)[0]
                if len(act) > 0:
                    pick = act[rng.integers(len(act))]
                    a = iu[0][pick]; b = iu[1][pick]
                    s = sigma * 2.0
                    cand[a] = cand[a] - s * base[b]
                    cand[b] = cand[b] - s * base[a]
                    # jitter rest
                    cand += rng.standard_normal(cand.shape) * (sigma * 0.3)
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                else:
                    cand = base + rng.standard_normal(base.shape) * sigma
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            except Exception:
                cand = base + rng.standard_normal(base.shape) * sigma
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        else:
            # Gaussian kick, rattler-targeted every other
            try:
                rm, mcur = row_max(base)
                is_rat = rm < (mcur - 5e-4)
            except Exception:
                is_rat = np.zeros(n, dtype=bool)
            s = sigma * (0.7 + 0.6 * rng.random())
            noise = rng.standard_normal(base.shape) * s
            if np.any(is_rat) and (r < 0.6):
                # big kick to rattlers
                big = 0.25 + 0.35 * rng.random()
                rr = rng.standard_normal(base.shape) * big
                noise[is_rat] = rr[is_rat]
                if r < 0.25:
                    # fully randomize a few rattlers
                    idx = np.where(is_rat)[0]
                    k = min(len(idx), 1 + int(rng.integers(3)))
                    pick = rng.choice(idx, size=k, replace=False)
                    nr = rng.standard_normal((k, d))
                    nr /= np.linalg.norm(nr, axis=1, keepdims=True)
                    cand[pick] = nr
                    cand = cand + noise * 0.3
                else:
                    cand = base + noise
            else:
                cand = base + noise
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)

        # choose GD preset
        pr = gd_presets[(it - 1) % len(gd_presets)]
        # vary slightly
        steps, a0, a1, s0, s1, mu = pr
        if rng.random() < 0.3:
            a1 = a1 * (0.7 + 0.6 * rng.random())
        try:
            cand = smooth_gd(cand, steps, a0, a1, s0, s1, mu)
        except Exception:
            continue
        v = max_cos(cand)
        if v < best_v + 0.004 and v >= best_v - 1e-12:
            # extra refine promising near-best
            try:
                cand = smooth_gd(cand, 250, a1, a1 * 2.0, s1, s1 * 0.4, 0.85)
                v = max_cos(cand)
            except Exception:
                pass
        if v < best_v:
            best_x = cand
            best_v = v
            # quick lock with tiny LP if improvement and time permits
            if time.time() - t0 < BUDGET - 30.0 and (best_v < 0.8981 or it % 3 == 0):
                try:
                    rem = min(3.0, max(0.5, (BUDGET - 18.0 - (time.time() - t0)) / 10.0))
                    locked = lp_polish(best_x, rem)
                    lv = max_cos(locked)
                    if lv < best_v:
                        best_x, best_v = locked, lv
                except Exception:
                    pass
        # early exit if big win
        if best_v < 0.895 and time.time() - t0 > 60:
            break

    # final thorough polish of best
    try:
        # annealed GD from high to very high alpha
        best_x = smooth_gd(best_x, 600, 40.0, 600.0, 0.012, 0.001, 0.85)
    except Exception:
        pass
    try:
        left = BUDGET - (time.time() - t0) - 1.0
        if left > 2.0:
            best_x = lp_polish(best_x, min(left, 14.0))
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

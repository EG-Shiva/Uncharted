import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    """Trust-region LP polish with cap on constraints for speed."""
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
        # cap number of constraints for speed
        if np.count_nonzero(sel) > 4000:
            # keep top 4000
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 4000 - 1)[:4000]
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
        # stop if out of time (linprog may have consumed it)
        if time.time() - start >= seconds:
            break
    return x


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.8):
    """Annealed softmax repulsion with trust-region-normalized momentum."""
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        smax = s0 * (s1 / s0) ** frac if (s0 > 0 and s1 > 0 and s0 != s1) else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # exp(alpha*(G-m)); G-m <= 0 so no overflow
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


def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G.max()


def solve():
    t0 = time.time()
    BUDGET = 168.0  # leave margin to 180
    rng = np.random.default_rng(0)

    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            # fallback random
            bx = rng.standard_normal((272, 4))
            bx /= np.linalg.norm(bx, axis=1, keepdims=True)
    bx = np.asarray(bx, dtype=np.float64)
    if bx.shape != (272, 4):
        # try to reshape / truncate
        bx = np.resize(bx, (272, 4))
    bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
    best_x = bx.copy()
    best_v = max_cos(best_x)

    # quick evaluation of record too
    try:
        rx = np.load("record.npy")
        rx = rx / np.linalg.norm(rx, axis=1, keepdims=True)
        rv = max_cos(rx)
        if rv < best_v:
            best_x, best_v = rx.copy(), rv
    except Exception:
        pass

    sigmas = [0.004, 0.009, 0.020, 0.045, 0.090, 0.015]
    it = 0
    # reserve ~12s for final polish
    while time.time() - t0 < BUDGET - 14.0:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        # identify rattlers of current best
        try:
            rowmax, mcur = per_point_max(best_x)
            is_rattler = rowmax < (mcur - 5e-4)
        except Exception:
            is_rattler = np.zeros(272, dtype=bool)
        # number of candidates per outer iter (GD is cheap)
        n_cand = 3 if time.time() - t0 < BUDGET - 40 else 2
        cands = []
        for k in range(n_cand):
            # vary sigma slightly per candidate
            s = sigma * (0.7 + 0.6 * rng.random())
            noise = rng.standard_normal(best_x.shape) * s
            # boost rattlers
            if np.any(is_rattler):
                noise[is_rattler] *= 3.0
            # occasional contact-break push on top pair
            xp = best_x + noise
            if rng.random() < 0.5:
                # pick a near-max pair and push apart
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                # find one of top pairs randomly
                iu = np.triu_indices(272, 1)
                cg = G[iu]
                # top 20 pairs
                top = np.argpartition(-cg, min(20, len(cg) - 1))[:20]
                choice = top[rng.integers(len(top))]
                i = int(iu[0][choice]); j = int(iu[1][choice])
                push = 0.5 * s + 0.005
                xp[i] += push * (best_x[i] - best_x[j])
                xp[j] += push * (best_x[j] - best_x[i])
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            # GD params jitter for diversity
            a0 = 22.0 + 8.0 * rng.random()
            a1 = 150.0 + 80.0 * rng.random()
            s0 = 0.008 + 0.006 * rng.random()
            s1 = 0.0008 + 0.0008 * rng.random()
            steps = 900 + int(rng.integers(0, 500))
            # reduce steps if low on time
            remain = BUDGET - 14.0 - (time.time() - t0)
            if remain < 30:
                steps = min(steps, 600)
            xp = smooth_gd(xp, steps, a0, a1, s0, s1, mu=0.8)
            v = max_cos(xp)
            cands.append((v, xp))
        cands.sort(key=lambda t: t[0])
        # LP-polish only the most promising (and maybe second if close)
        to_polish = [cands[0][1]]
        if len(cands) > 1 and cands[1][0] < cands[0][0] + 0.002 and rng.random() < 0.4:
            to_polish = [cands[1][1]]
        # alternate which to polish to save time
        xp0 = to_polish[0]
        # time for this LP polish
        remain = BUDGET - 14.0 - (time.time() - t0)
        if remain < 4:
            break
        lp_time = min(6.0, max(2.5, remain / 6.0))
        yp = lp_polish(xp0, lp_time)
        vp = max_cos(yp)
        if vp < best_v - 1e-12:
            best_x, best_v = yp, vp
        else:
            # also try polishing best GD candidate briefly if first failed?
            # opportunistic: if GD alone beats best, keep it and polish later
            if cands[0][0] < best_v - 5e-5:
                # promising basin, polish it more next round by making it best
                # do a short extra polish now if time allows
                remain2 = BUDGET - 14.0 - (time.time() - t0)
                if remain2 > 5:
                    yp2 = lp_polish(cands[0][1], min(4.0, remain2 - 1.0))
                    vp2 = max_cos(yp2)
                    if vp2 < best_v - 1e-12:
                        best_x, best_v = yp2, vp2
        # early exit if time low
        if time.time() - t0 >= BUDGET - 14.0:
            break

    # final squeeze on best
    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        best_x = lp_polish(best_x, min(remain - 1.0, 11.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

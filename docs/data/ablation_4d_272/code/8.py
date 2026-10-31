import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def rowmax_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    m = float(rm.max())
    return rm, m, G


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
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > 5000:
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


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        if s0 != s1:
            smax = s0 * (s1 / s0) ** frac
        else:
            smax = s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def hard_descent(x, steps, step0, step1, tol0=2e-3, tol1=2e-4):
    # directly push apart near-max pairs
    n, d = x.shape
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        step = step0 * (step1 / step0) ** frac if step0 != step1 else step0
        tol = tol0 * (tol1 / tol0) ** frac if tol0 != tol1 else tol0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = G > (m - tol)
        # symmetric mask, no diagonal
        grad = M.astype(np.float64) @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = float(np.linalg.norm(gt, axis=1).max())
        if nrm < 1e-14:
            continue
        # per-point normalize to step to avoid one crowded point dominating
        pn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-14
        # scale so max move == step
        gt = gt / pn * np.minimum(pn / (pn.mean() + 1e-14), 1.0)
        # simpler: global normalize
        x = x - step * gt / (float(np.linalg.norm(gt, axis=1).max()) + 1e-14)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def find_hole(x, rng, n_sample=24000):
    n, d = x.shape
    # random samples on S^3
    S = rng.standard_normal((n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    # max dot to code: smaller is larger hole
    # chunk to save memory
    best_val = np.empty(n_sample, dtype=np.float64)
    chunk = 4000
    for a in range(0, n_sample, chunk):
        b = min(a + chunk, n_sample)
        Gm = S[a:b] @ x.T
        best_val[a:b] = Gm.max(axis=1)
    k = int(np.argmin(best_val))
    h = S[k].copy()
    # refine hole by hill climbing: minimize max dot
    for _ in range(120):
        dots = x @ h  # (n,)
        m = float(dots.max())
        # neighbours within 0.02 of max influence gradient
        sel = dots > (m - 0.03)
        g = np.sum(x[sel] - dots[sel, None] * h[None, :], axis=0)
        # g points toward increasing max-dot; move opposite, tangential
        g = g - np.dot(g, h) * h
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        st = 0.01
        hn = h - st * g / ng
        hn /= np.linalg.norm(hn)
        # accept if improves
        mn = float((x @ hn).max())
        if mn < m:
            h = hn
        else:
            # smaller step
            hn2 = h - 0.002 * g / ng
            hn2 /= np.linalg.norm(hn2)
            mn2 = float((x @ hn2).max())
            if mn2 < m:
                h = hn2
            else:
                break
    return h


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(42)

    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.standard_normal((272, 4))
    bx = np.asarray(bx, dtype=np.float64)
    if bx.shape != (272, 4):
        bx = np.resize(bx, (272, 4))
    bx /= np.linalg.norm(bx, axis=1, keepdims=True)
    best_x = bx.copy()
    best_v = max_cos(best_x)
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64) / np.linalg.norm(np.asarray(rx, dtype=np.float64), axis=1, keepdims=True)
        rv = max_cos(rx)
        if rv < best_v:
            best_x, best_v = rx.copy(), rv
    except Exception:
        pass

    sigmas = [0.004, 0.009, 0.020, 0.045, 0.085, 0.014]
    it = 0
    # cache hole periodically
    cached_hole = None
    hole_age = 100

    while time.time() - t0 < BUDGET - 14.0:
        it += 1
        remain_total = BUDGET - 14.0 - (time.time() - t0)
        if remain_total < 4:
            break
        sigma = sigmas[(it - 1) % len(sigmas)]
        try:
            rowmax, mcur, Gcur = rowmax_info(best_x)
            is_rattler = rowmax < (mcur - 5e-4)
            # contact degree
            deg = np.count_nonzero(Gcur > (mcur - 1e-3), axis=1)
        except Exception:
            rowmax = None
            is_rattler = np.zeros(272, dtype=bool)
            deg = np.zeros(272, dtype=int)

        n_cand = 3 if remain_total > 40 else 2
        cands = []
        # refresh hole every few iters or when best changes
        if cached_hole is None or hole_age > 3:
            try:
                cached_hole = find_hole(best_x, rng, n_sample=20000)
            except Exception:
                cached_hole = None
            hole_age = 0

        for k in range(n_cand):
            mode = rng.random()
            xp = None
            if mode < 0.35 and cached_hole is not None:
                # transplant victim into hole
                xp = best_x.copy()
                # victim: among worst 12 by rowmax/degree
                order = np.argsort(-(rowmax + 1e-3 * deg))
                pool = order[:12]
                victim = int(pool[rng.integers(len(pool))])
                s = sigma * (0.7 + 0.6 * rng.random())
                noise = rng.standard_normal(4) * (0.002 + s * 0.3)
                xp[victim] = cached_hole + noise
                xp[victim] /= np.linalg.norm(xp[victim])
                # small global jitter to break symmetry
                xp = xp + rng.standard_normal(xp.shape) * (s * 0.35)
                # boost rattlers a bit
                if np.any(is_rattler):
                    xp[is_rattler] += rng.standard_normal((int(is_rattler.sum()), 4)) * (s * 0.5)
                xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            else:
                s = sigma * (0.7 + 0.6 * rng.random())
                noise = rng.standard_normal(best_x.shape) * s
                if np.any(is_rattler):
                    noise[is_rattler] *= 3.0
                xp = best_x + noise
                if rng.random() < 0.5:
                    G = best_x @ best_x.T
                    np.fill_diagonal(G, -2.0)
                    iu = np.triu_indices(272, 1)
                    cg = G[iu]
                    top = np.argpartition(-cg, 19)[:20]
                    choice = top[rng.integers(len(top))]
                    i = int(iu[0][choice]); j = int(iu[1][choice])
                    push = 0.5 * s + 0.005
                    xp[i] += push * (best_x[i] - best_x[j])
                    xp[j] += push * (best_x[j] - best_x[i])
                xp /= np.linalg.norm(xp, axis=1, keepdims=True)

            a0 = 20.0 + 10.0 * rng.random()
            a1 = 140.0 + 100.0 * rng.random()
            s0 = 0.008 + 0.006 * rng.random()
            s1 = 0.0007 + 0.0009 * rng.random()
            steps = 750 + int(rng.integers(0, 450))
            if remain_total < 30:
                steps = min(steps, 550)
            xp = smooth_gd(xp, steps, a0, a1, s0, s1, mu=0.85)
            # hard sharpening phase
            xp = hard_descent(xp, 120, 0.003, 0.0004)
            v = max_cos(xp)
            cands.append((v, xp))

        cands.sort(key=lambda t: t[0])
        # pick which to LP-polish: best, but occasionally second if close
        pick = cands[0][1]
        if len(cands) > 1 and cands[1][0] < cands[0][0] + 0.0015 and rng.random() < 0.3:
            pick = cands[1][1]
        remain = BUDGET - 14.0 - (time.time() - t0)
        if remain < 3:
            # keep GD gain if any
            if cands[0][0] < best_v:
                best_x, best_v = cands[0][1], cands[0][0]
                hole_age = 100
            break
        lp_time = min(6.0, max(2.5, remain / 6.0))
        yp = lp_polish(pick, lp_time)
        vp = max_cos(yp)
        improved = False
        if vp < best_v - 1e-12:
            best_x, best_v = yp, vp
            improved = True
            hole_age = 100
        else:
            if cands[0][0] < best_v - 5e-5:
                remain2 = BUDGET - 14.0 - (time.time() - t0)
                if remain2 > 5:
                    yp2 = lp_polish(cands[0][1], min(4.0, remain2 - 1.0))
                    vp2 = max_cos(yp2)
                    if vp2 < best_v - 1e-12:
                        best_x, best_v = yp2, vp2
                        improved = True
                        hole_age = 100
        if not improved:
            hole_age += 1
        else:
            hole_age = 100  # force hole refresh

        if time.time() - t0 >= BUDGET - 14.0:
            break

    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        try:
            best_x = lp_polish(best_x, min(remain - 1.0, 11.0))
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

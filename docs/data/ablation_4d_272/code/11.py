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
    rm = G.max(axis=1)
    return rm, float(rm.max())


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
        if np.count_nonzero(sel) > 4500:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 4500 - 1)[:4500]
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


def repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85, long_w=0.005, long_p=6.0):
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
        if long_w > 0:
            Gl = np.clip(G, 0.0, None)
            # (G clipped) power, only where G>0 to keep cheap
            W = W + long_w * np.power(Gl, long_p) * np.exp(alpha * 0.0)  # small background
            # fix diagonal back to 0
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


def refine_single(pt, others, steps=150, lr=0.02):
    # maximise minimal angle to others: minimise max dot
    pt = pt / np.linalg.norm(pt)
    for _ in range(steps):
        d = others @ pt  # (m,)
        m = d.max()
        # softmax weights to worst neighbours
        alpha = 60.0
        w = np.exp(alpha * (d - m))
        g = w @ others  # gradient of weighted dot
        # project to tangent and descend
        gt = g - np.dot(g, pt) * pt
        n = np.linalg.norm(gt)
        if n < 1e-14:
            break
        pt = pt - (lr / max(1.0, n * 5.0)) * gt * 5.0
        # simpler fixed trust step
        pt /= np.linalg.norm(pt)
        lr *= 0.995
    return pt / np.linalg.norm(pt)


def reinsert_rattlers(x, rng, n_replace=3, n_sample=12000):
    n, d = x.shape
    rm, m = row_max(x)
    order = np.argsort(rm)  # lowest first = most rattler
    # only replace among rattlers if exist, else lowest ones
    is_r = rm < (m - 4e-4)
    cand_idx = order[:max(n_replace * 3, 8)]
    # pick n_replace among them randomly (bias lowest)
    if len(cand_idx) < n_replace:
        chosen = cand_idx
    else:
        # take lowest n_replace with some randomness
        chosen = cand_idx[:n_replace]
        if rng.random() < 0.5 and len(cand_idx) > n_replace:
            # swap one with a random other low one
            chosen = chosen.copy()
            chosen[-1] = cand_idx[rng.integers(n_replace, len(cand_idx))]
    y = x.copy()
    for idx in chosen:
        others = np.delete(y, idx, axis=0)
        # sample candidates
        best_pt = None
        best_v = 2.0
        # chunk sampling to bound memory
        chunk = 4000
        remaining = n_sample
        while remaining > 0:
            cs = min(chunk, remaining)
            S = rng.standard_normal((cs, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            D = S @ others.T
            vm = D.max(axis=1)
            k = int(np.argmin(vm))
            if float(vm[k]) < best_v:
                best_v = float(vm[k])
                best_pt = S[k].copy()
            remaining -= cs
        best_pt = refine_single(best_pt, others, steps=120, lr=0.03)
        y[idx] = best_pt
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
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
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands0.append(z)
    # pick best of loaded
    vals = [(max_cos(c), c) for c in cands0]
    vals.sort(key=lambda t: t[0])
    best_v, best_x = vals[0][0], vals[0][1].copy()

    pool = [(best_v, best_x.copy())]  # elite archive (v,x)

    def pool_base():
        # bias to best but sometimes pick others
        if len(pool) == 1 or rng.random() < 0.7:
            return pool[0][1]
        else:
            k = rng.integers(len(pool))
            return pool[k][1]

    def add_to_pool(v, x):
        nonlocal pool
        # keep up to 4 diverse entries
        for vv, xx in pool:
            if np.linalg.norm(xx - x) < 1e-6 or abs(vv - v) < 1e-12 and vv <= v:
                pass
        pool.append((v, x.copy()))
        pool.sort(key=lambda t: t[0])
        pool = pool[:4]

    sigmas = [0.004, 0.010, 0.022, 0.05, 0.09, 0.015, 0.035]
    it = 0
    iu_all = np.triu_indices(n, 1)

    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        remain = BUDGET - 13.0 - (time.time() - t0)
        if remain < 3:
            break
        base = pool_base()
        rm, mcur = row_max(base)
        is_rattler = rm < (mcur - 5e-4)
        u = rng.random()
        # build 2-3 GD candidates with different moves
        n_cand = 2
        if remain > 60:
            n_cand = 3
        gd_cands = []
        for k in range(n_cand):
            if u < 0.50:
                # classic rattler-boosted kick
                sigma = sigmas[(it + k) % len(sigmas)] * (0.7 + 0.6 * rng.random())
                noise = rng.standard_normal(base.shape) * sigma
                if np.any(is_rattler):
                    noise[is_rattler] *= 3.0
                xp = base + noise
                if rng.random() < 0.5:
                    G = base @ base.T
                    np.fill_diagonal(G, -2.0)
                    cg = G[iu_all]
                    top = np.argpartition(-cg, 19)[:20]
                    choice = top[rng.integers(len(top))]
                    i = int(iu_all[0][choice]); j = int(iu_all[1][choice])
                    push = 0.5 * sigma + 0.005
                    xp[i] += push * (base[i] - base[j])
                    xp[j] += push * (base[j] - base[i])
                xp /= np.linalg.norm(xp, axis=1, keepdims=True)
                a0 = 20.0 + 10.0 * rng.random()
                a1 = 140.0 + 100.0 * rng.random()
                s0 = 0.007 + 0.007 * rng.random()
                s1 = 0.0006 + 0.0012 * rng.random()
                steps = 800 + int(rng.integers(0, 500))
                if remain < 30:
                    steps = min(steps, 550)
                mu = 0.8 + 0.1 * rng.random()
                lw = float(rng.choice([0.0, 0.003, 0.008]))
                xp = repulse_gd(xp, steps, a0, a1, s0, s1, mu=mu, long_w=lw)
            elif u < 0.75:
                # rattler reinsertion + short GD
                nr = int(rng.integers(1, 5))
                xp = reinsert_rattlers(base, rng, n_replace=nr, n_sample=8000)
                # small shake of the rest
                xp = xp + rng.standard_normal(xp.shape) * 0.003
                xp /= np.linalg.norm(xp, axis=1, keepdims=True)
                a0 = 25.0 + 10.0 * rng.random()
                a1 = 160.0 + 80.0 * rng.random()
                steps = 500 + int(rng.integers(0, 300))
                xp = repulse_gd(xp, steps, a0, a1, 0.006, 0.0008, mu=0.85, long_w=0.003)
            elif u < 0.90:
                # worst-subset focused kick
                K = 24 + int(rng.integers(0, 24))
                worst = np.argsort(-rm)[:K]
                sigma_big = 0.02 + 0.03 * rng.random()
                sigma_small = 0.002
                noise = rng.standard_normal(base.shape) * sigma_small
                noise[worst] = rng.standard_normal((K, d)) * sigma_big
                xp = base + noise
                xp /= np.linalg.norm(xp, axis=1, keepdims=True)
                a0 = 22.0 + 8.0 * rng.random()
                a1 = 180.0 + 80.0 * rng.random()
                steps = 600 + int(rng.integers(0, 400))
                xp = repulse_gd(xp, steps, a0, a1, 0.008, 0.0007, mu=0.85, long_w=0.005)
            else:
                # fresh random basin with Riesz-like spread then focus
                xp = rng.standard_normal(base.shape)
                xp /= np.linalg.norm(xp, axis=1, keepdims=True)
                # two-stage: low alpha spread then high alpha focus
                xp = repulse_gd(xp, 500, 8.0, 30.0, 0.02, 0.005, mu=0.8, long_w=0.01, long_p=4.0)
                xp = repulse_gd(xp, 600, 30.0, 200.0, 0.008, 0.0008, mu=0.85, long_w=0.0)
            gd_cands.append((max_cos(xp), xp))
        gd_cands.sort(key=lambda t: t[0])
        # update pool with good GD finds even before LP
        for v, xp in gd_cands:
            if v < pool[-1][0] + 0.004 and v < best_v + 0.004:
                add_to_pool(v, xp)
        # decide LP polish: polish best gd cand if promising
        best_gd_v = gd_cands[0][0]
        do_polish = False
        if best_gd_v < best_v - 2e-6:
            do_polish = True
            to_polish = gd_cands[0][1]
        elif best_gd_v < best_v + 0.0015 and rng.random() < 0.55:
            do_polish = True
            to_polish = gd_cands[0][1]
        elif len(gd_cands) > 1 and gd_cands[1][0] < best_v + 0.0005 and rng.random() < 0.25:
            do_polish = True
            to_polish = gd_cands[1][1]
        if do_polish:
            remain2 = BUDGET - 13.0 - (time.time() - t0)
            if remain2 < 2.5:
                break
            lp_time = min(6.0, max(2.5, remain2 / 7.0))
            yp = lp_polish(to_polish, lp_time)
            vp = max_cos(yp)
            if vp < best_v - 1e-12:
                best_x, best_v = yp, vp
                add_to_pool(vp, yp)
            else:
                # if GD alone beat best, keep it
                if best_gd_v < best_v - 1e-12:
                    best_x, best_v = gd_cands[0][1], best_gd_v
                    add_to_pool(best_v, best_x)
        else:
            if best_gd_v < best_v - 1e-12:
                best_x, best_v = gd_cands[0][1], best_gd_v
                add_to_pool(best_v, best_x)

    remain = BUDGET - (time.time() - t0)
    if remain > 3.5:
        best_x = lp_polish(best_x, min(remain - 1.0, 11.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

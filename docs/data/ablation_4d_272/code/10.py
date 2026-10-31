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
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        if np.count_nonzero(sel) > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
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


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.82):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        smax = s0 * (s1 / s0) ** frac if (s0 > 0 and s1 > 0 and s0 != s1) else s0
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


def hard_refine(x, steps=180, s0=0.003, tol=0.0025):
    for it in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = (G > m - tol).astype(np.float64)
        # symmetrize already symmetric
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = np.linalg.norm(gt, axis=1).max()
        if nrm < 1e-14:
            continue
        s = s0 * (1.0 - it / steps) + 1e-5
        x = x - (s / nrm) * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def find_holes(X_keep, k, rng, nsamp=25000):
    # largest-hole search: minimize max dot
    cand = rng.standard_normal((nsamp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X_keep.T
    scores = dots.max(axis=1)
    order = np.argsort(scores)
    # diverse selection
    sel = []
    for idx in order:
        c = cand[idx]
        ok = True
        for s in sel:
            if float(c @ cand[s]) > 0.93:
                ok = False
                break
        if ok:
            sel.append(int(idx))
            if len(sel) >= k:
                break
    holes = cand[np.array(sel)]
    # local refine each hole: subgradient descent on max-dot
    for hi in range(len(holes)):
        y = holes[hi].copy()
        step = 0.02
        for _ in range(250):
            d = X_keep @ y
            m = float(d.max())
            mask = d > m - 1e-3
            g = X_keep[mask].sum(axis=0)
            gt = g - float(g @ y) * y
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-12:
                break
            gt /= nrm
            y = y - step * gt
            y /= np.linalg.norm(y)
            step *= 0.996
        holes[hi] = y
    return holes


def contact_components(G, m, delta):
    n = G.shape[0]
    adj = [np.where(G[i] > m - delta)[0].tolist() for i in range(n)]
    # remove self
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v == u:
                    continue
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)

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
        rx = np.asarray(rx, dtype=np.float64)
        rx /= np.linalg.norm(rx, axis=1, keepdims=True)
        rv = max_cos(rx)
        if rv < best_v:
            best_x, best_v = rx.copy(), rv
    except Exception:
        pass

    sigmas = [0.004, 0.009, 0.020, 0.045, 0.090, 0.015]
    it = 0
    stall = 0
    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        remain = BUDGET - 13.0 - (time.time() - t0)
        if remain < 3.0:
            break
        rtype = rng.random()
        # boost hole moves if stalled
        if stall > 12 and rtype < 0.55:
            rtype = 0.65  # force hole move

        if rtype < 0.52:
            # --- standard rattler kick + softmax GD ---
            sigma = sigmas[(it - 1) % len(sigmas)] * (0.7 + 0.6 * rng.random())
            if stall > 8:
                sigma *= 1.6
            try:
                rm, mc = row_max(best_x)
                is_r = rm < (mc - 5e-4)
            except Exception:
                is_r = np.zeros(272, dtype=bool)
            noise = rng.standard_normal(best_x.shape) * sigma
            if np.any(is_r):
                noise[is_r] *= 3.0
            xp = best_x + noise
            if rng.random() < 0.5:
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(272, 1)
                cg = G[iu]
                top = np.argpartition(-cg, 19)[:20]
                ch = top[rng.integers(len(top))]
                i = int(iu[0][ch]); j = int(iu[1][ch])
                push = 0.5 * sigma + 0.005
                xp[i] += push * (best_x[i] - best_x[j])
                xp[j] += push * (best_x[j] - best_x[i])
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            a0 = 22.0 + 8.0 * rng.random()
            a1 = 150.0 + 90.0 * rng.random()
            s0 = 0.008 + 0.006 * rng.random()
            s1 = 0.0008 + 0.0008 * rng.random()
            steps = 850 + int(rng.integers(0, 500))
            if remain < 30:
                steps = min(steps, 550)
            xp = smooth_gd(xp, steps, a0, a1, s0, s1, mu=0.82)
            if rng.random() < 0.7:
                xp = hard_refine(xp, steps=120, s0=0.002)
            v = max_cos(xp)
        elif rtype < 0.80:
            # --- hole respawn of worst points ---
            try:
                rm, mc = row_max(best_x)
            except Exception:
                rm = np.zeros(272); mc = best_v
            kr = rng.random()
            if kr < 0.4:
                k = 1
            elif kr < 0.7:
                k = 2
            elif kr < 0.9:
                k = 3
            else:
                k = 4
            worst = np.argsort(rm)[-k:]
            mask = np.ones(272, dtype=bool)
            mask[worst] = False
            X_keep = best_x[mask]
            try:
                holes = find_holes(X_keep, k, rng, nsamp=25000)
            except Exception:
                holes = rng.standard_normal((k, 4))
                holes /= np.linalg.norm(holes, axis=1, keepdims=True)
            xp = best_x.copy()
            xp[worst] = holes
            # tiny jitter to break symmetry
            xp[worst] += rng.standard_normal((k, 4)) * 0.002
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            a0 = 25.0 + 10.0 * rng.random()
            a1 = 160.0 + 80.0 * rng.random()
            s0 = 0.006 + 0.004 * rng.random()
            s1 = 0.0007 + 0.0006 * rng.random()
            steps = 750 + int(rng.integers(0, 400))
            if remain < 30:
                steps = min(steps, 550)
            xp = smooth_gd(xp, steps, a0, a1, s0, s1, mu=0.82)
            xp = hard_refine(xp, steps=100, s0=0.0015)
            v = max_cos(xp)
        else:
            # --- contact-component rigid move ---
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            delta = 0.006 + 0.006 * rng.random()
            try:
                comps = contact_components(G, m, delta)
            except Exception:
                comps = [list(range(272))]
            # pick a non-trivial component, else random subset
            big = [c for c in comps if 2 <= len(c) <= 120]
            if len(big) > 0 and rng.random() < 0.85:
                comp = big[rng.integers(len(big))]
            else:
                # random block of worst points
                rm, _ = row_max(best_x)
                comp = np.argsort(rm)[-int(10 + rng.integers(40)):].tolist()
            sigma = 0.01 + 0.03 * rng.random()
            common = rng.standard_normal(4) * sigma
            # project common to be roughly tangential on average? just add
            xp = best_x.copy()
            noise = rng.standard_normal((len(comp), 4)) * (sigma * 0.4)
            xp[np.array(comp)] += common + noise
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            a0 = 20.0 + 10.0 * rng.random()
            a1 = 140.0 + 80.0 * rng.random()
            s0 = 0.007 + 0.005 * rng.random()
            s1 = 0.0008 + 0.0007 * rng.random()
            steps = 800 + int(rng.integers(0, 400))
            if remain < 30:
                steps = min(steps, 550)
            xp = smooth_gd(xp, steps, a0, a1, s0, s1, mu=0.8)
            v = max_cos(xp)

        # decide LP polish
        if v < best_v + 0.004:
            rem2 = BUDGET - 13.0 - (time.time() - t0)
            if rem2 < 2.5:
                # no time for LP, just compare GD value
                if v < best_v - 1e-12:
                    best_x, best_v = xp, v
                    stall = 0
                else:
                    stall += 1
                continue
            lp_time = min(5.0, max(2.2, rem2 / 6.0))
            yp = lp_polish(xp, lp_time)
            vp = max_cos(yp)
            if vp < best_v - 1e-12:
                best_x, best_v = yp, vp
                stall = 0
            else:
                if v < best_v - 5e-5:
                    # promising but LP didn't win; keep trying nearby next time?
                    stall += 1
                else:
                    stall += 1
        else:
            stall += 1

    remain = BUDGET - (time.time() - t0)
    if remain > 3.0:
        best_x = lp_polish(best_x, min(remain - 1.0, 11.0))
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

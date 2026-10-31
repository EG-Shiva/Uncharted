import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, max_pairs=5000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 2e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        tol = max(20 * radius, 1e-9)
        sel = c > m - tol
        if int(np.count_nonzero(sel)) == 0:
            break
        # keep only largest pairs if too many
        p0 = int(np.count_nonzero(sel))
        if p0 > max_pairs:
            idx = np.where(sel)[0]
            part = np.argpartition(-c[idx], max_pairs - 1)[:max_pairs]
            keep = idx[part]
            mask = np.zeros_like(sel)
            mask[keep] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def softmax_refine(x_init, steps, a0, a1, s0, s1, n0, n1, mu, window, rng):
    x = x_init.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    v = np.zeros_like(x)
    for it in range(steps):
        frac = it / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** frac if a0 > 0 and a1 > 0 else a0
        step = s0 * (s1 / s0) ** frac if s0 > 0 and s1 > 0 else s0 + (s1 - s0) * frac
        if n0 > 0:
            noise = n0 * (n1 / n0) ** frac if n1 > 0 else n0 * (1.0 - frac)
        else:
            noise = 0.0
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        M = float(np.max(G))
        np.exp(alpha * (G - M), out=G)
        if window < np.inf:
            try:
                thr = np.exp(-alpha * window)
                G[G < thr] = 0.0
            except Exception:
                pass
        sumU = float(G.sum() * 0.5)
        if not np.isfinite(sumU) or sumU <= 0:
            continue
        G /= sumU
        grad = G @ x
        dots = np.sum(grad * x, axis=1, keepdims=True)
        Gt = grad - dots * x
        rms = float(np.sqrt(np.mean(Gt * Gt) + 1e-24))
        direction = Gt / rms
        v = mu * v + (1.0 - mu) * direction
        if noise > 1e-13:
            Zn = rng.standard_normal((n, d))
            Zdot = np.sum(Zn * x, axis=1, keepdims=True)
            Zt = Zn - Zdot * x
            zrms = float(np.sqrt(np.mean(Zt * Zt) + 1e-24))
            Zt /= zrms
            x = x - step * v + noise * Zt
        else:
            x = x - step * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def riesz_refine(x_init, steps, e0, e1, s0, s1, n0, n1, mu, rng):
    x = x_init.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    v = np.zeros_like(x)
    for it in range(steps):
        frac = it / (steps - 1) if steps > 1 else 0.0
        es = e0 * (e1 / e0) ** frac if e0 > 0 and e1 > 0 else e0 + (e1 - e0) * frac
        step = s0 * (s1 / s0) ** frac if s0 > 0 and s1 > 0 else s0 + (s1 - s0) * frac
        if n0 > 0:
            noise = n0 * (n1 / n0) ** frac if n1 > 0 else n0 * (1.0 - frac)
        else:
            noise = 0.0
        G = x @ x.T
        # r^2 = 2-2g
        R2 = 2.0 - 2.0 * G
        np.fill_diagonal(R2, 1.0)
        np.clip(R2, 1e-12, None, out=R2)
        pw = -es / 2.0 - 1.0
        # W = R2^pw, avoid overflow
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            W = np.power(R2, pw)
        W[~np.isfinite(W)] = 1e12
        np.fill_diagonal(W, 0.0)
        ssum = float(W.sum() * 0.5)
        if not np.isfinite(ssum) or ssum <= 0:
            continue
        W /= ssum
        grad = W @ x
        dots = np.sum(grad * x, axis=1, keepdims=True)
        Gt = grad - dots * x
        rms = float(np.sqrt(np.mean(Gt * Gt) + 1e-24))
        direction = Gt / rms
        v = mu * v + (1.0 - mu) * direction
        if noise > 1e-13:
            Zn = rng.standard_normal((n, d))
            Zdot = np.sum(Zn * x, axis=1, keepdims=True)
            Zt = Zn - Zdot * x
            zrms = float(np.sqrt(np.mean(Zt * Zt) + 1e-24))
            Zt /= zrms
            x = x - step * v + noise * Zt
        else:
            x = x - step * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def greedy_respawn(x, idx, rng, n_cand=400):
    n, d = x.shape
    y = x.copy()
    others_mask = np.ones(n, dtype=bool)
    others_mask[np.asarray(idx)] = False
    fixed = y[others_mask]
    for k in idx:
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # score = max dot to fixed + already placed respawned
        # build pool of blockers
        pool = fixed
        # include other respawned that are already updated (excluding current)
        # simple: use y with current row zeroed? just use y of idx != k
        other_idx = [t for t in idx if t != k]
        if len(other_idx) > 0:
            pool = np.vstack([fixed, y[np.asarray(other_idx)]])
        scores = cand @ pool.T
        m = scores.max(axis=1)
        best_c = int(np.argmin(m))
        y[k] = cand[best_c]
    return y


def solve():
    t_start = time.time()
    HARD = t_start + 171.0
    n, d = 153, 4
    rng = np.random.default_rng(7)
    np.random.seed(7)

    def normed(a):
        a = np.asarray(a, dtype=float)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a

    best = None
    best_val = 2.0
    for f in ("best.npy", "record.npy"):
        try:
            xx = normed(np.load(f))
            if xx.shape == (n, d) and np.all(np.isfinite(xx)):
                vv = max_cos(xx)
                if vv < best_val:
                    best = xx
                    best_val = vv
        except Exception:
            pass
    if best is None:
        best = normed(rng.standard_normal((n, d)))
        best_val = max_cos(best)

    # quick initial polish of loaded best with tiny budget to get true value
    # (do not spend much)
    try:
        if HARD - time.time() > 40:
            bp = lp_polish(best, seconds=2.0)
            vp = max_cos(bp)
            if vp < best_val:
                best, best_val = bp, vp
    except Exception:
        pass

    trial = 0
    # reserve ~28s for final
    while True:
        now = time.time()
        if now > HARD - 28:
            break
        if trial >= 200:
            break
        mode = trial % 8
        xtry = None
        use_riesz = False
        # default softmax params placeholders
        if mode == 0:
            sig = 0.010 + 0.008 * rng.random()
            xtry = best + rng.standard_normal((n, d)) * sig
            xtry = normed(xtry)
            pars = (2200, 15.0, 200.0, 0.012, 0.0010, 0.003, 0.0, 0.86, 0.25)
        elif mode == 1:
            # worst-pair kick + cluster noise
            xtry = best + rng.standard_normal((n, d)) * 0.004
            G = xtry @ xtry.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii, jj = divmod(flat, n)
            beta = 0.03 + 0.03 * rng.random()
            xtry[ii] -= beta * xtry[jj]
            xtry[jj] -= beta * xtry[ii]
            # noisy cluster around ii
            dots = xtry @ xtry[ii]
            nn = np.argsort(-dots)[:8]
            xtry[nn] += rng.standard_normal((len(nn), d)) * 0.05
            xtry = normed(xtry)
            pars = (2200, 25.0, 280.0, 0.010, 0.0009, 0.002, 0.0, 0.90, 0.15)
        elif mode == 2:
            # rattler respawn with greedy hole filling
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            M = float(G.max())
            contact = G > M - 0.015
            deg = contact.sum(axis=1)
            ratt = np.where(deg <= 1)[0]
            if len(ratt) == 0:
                ratt = np.argsort(deg)[:4]
            k = min(len(ratt), 3 + int(rng.integers(1, 5)))
            sel = rng.choice(ratt, size=k, replace=False) if len(ratt) >= k else np.arange(n)[rng.choice(n, k, replace=False)]
            tmp = best + rng.standard_normal((n, d)) * 0.003
            tmp = normed(tmp)
            xtry = greedy_respawn(tmp, sel, rng, n_cand=500)
            xtry = normed(xtry)
            use_riesz = True
            rpars = (1800, 4.0, 14.0, 0.014, 0.0012, 0.004, 0.0, 0.82)
        elif mode == 3:
            sig = 0.028 + 0.015 * rng.random()
            xtry = best + rng.standard_normal((n, d)) * sig
            xtry = normed(xtry)
            use_riesz = True
            e1 = 10.0 + 8.0 * rng.random()
            rpars = (2000, 2.0, e1, 0.015, 0.0011, 0.005, 0.0, 0.80)
        elif mode == 4:
            # subset replacement
            k = int(rng.integers(5, 13))
            idx = rng.choice(n, k, replace=False)
            tmp = best + rng.standard_normal((n, d)) * 0.004
            tmp = normed(tmp)
            xtry = greedy_respawn(tmp, idx, rng, n_cand=400)
            xtry = normed(xtry)
            pars = (2200, 12.0, 170.0, 0.013, 0.0010, 0.0035, 0.0, 0.85, 0.30)
        elif mode == 5:
            sig = 0.05 + 0.03 * rng.random()
            xtry = best + rng.standard_normal((n, d)) * sig
            xtry = normed(xtry)
            use_riesz = True
            rpars = (1600, 3.0, 12.0, 0.014, 0.0010, 0.004, 0.0, 0.84)
        elif mode == 6:
            # fresh random start with long anneal
            xtry = normed(rng.standard_normal((n, d)))
            use_riesz = True
            rpars = (2500, 1.5, 16.0, 0.018, 0.0010, 0.006, 0.0, 0.78)
        else:
            sig = 0.012 + 0.010 * rng.random()
            xtry = best + rng.standard_normal((n, d)) * sig
            # extra kick to 6 random points
            kk = 6
            idx = rng.choice(n, kk, replace=False)
            xtry[idx] += rng.standard_normal((kk, d)) * 0.10
            xtry = normed(xtry)
            # sparse window softmax
            w = 0.08 + 0.12 * rng.random()
            pars = (2100, 20.0, 260.0, 0.011, 0.0010, 0.0025, 0.0, 0.88, w)

        if use_riesz:
            steps, e0, e1, s0, s1, n0, n1, mu = rpars
            # occasionally chain softmax after riesz for sharp finish
            xopt = riesz_refine(xtry, steps, e0, e1, s0, s1, n0, n1, mu, rng)
            # short sharp softmax tail
            try:
                xopt = softmax_refine(xopt, 500, 60.0, 220.0, 0.004, 0.0008, 0.0, 0.0, 0.9, 0.2, rng)
            except Exception:
                pass
        else:
            steps, a0, a1, s0, s1, n0, n1, mu, window = pars
            xopt = softmax_refine(xtry, steps, a0, a1, s0, s1, n0, n1, mu, window, rng)
        v = max_cos(xopt)
        # decide to LP polish if promising
        thresh = best_val + 0.004
        if mode == 6:
            thresh = best_val + 0.002  # random starts usually worse
        if v < thresh:
            tleft = HARD - time.time()
            if tleft < 30:
                # no time for intermediate polish
                if v < best_val - 1e-12:
                    best, best_val = xopt, v
            else:
                alloc = 1.6 if v < best_val + 0.001 else 1.0
                alloc = min(alloc, tleft - 29.0)
                if alloc < 0.4:
                    if v < best_val - 1e-12:
                        best, best_val = xopt, v
                else:
                    try:
                        xpol = lp_polish(xopt, seconds=alloc)
                    except Exception:
                        xpol = xopt
                    vp = max_cos(xpol)
                    bv = min(v, vp)
                    bx = xopt if v <= vp else xpol
                    if bv < best_val - 1e-12:
                        best, best_val = bx, bv
                    # also keep near-best diverse pool? just continue
        else:
            pass
        trial += 1

    # final intensive polish + micro kicks
    tleft = HARD - time.time() - 0.5
    if tleft > 3:
        try:
            best = lp_polish(best, seconds=min(tleft * 0.6, 14.0))
            best_val = max_cos(best)
        except Exception:
            pass
    # micro-perturb repolish loop
    for _ in range(6):
        rem = HARD - time.time() - 0.5
        if rem < 2.5:
            break
        try:
            pert = best + rng.standard_normal((n, d)) * 2e-7
            pert = normed(pert)
            cand = lp_polish(pert, seconds=min(2.0, rem - 0.5))
            vv = max_cos(cand)
            if vv < best_val - 1e-14:
                best, best_val = cand, vv
            else:
                # try slightly larger kick once
                pert2 = best + rng.standard_normal((n, d)) * 1e-5
                pert2 = normed(pert2)
                cand2 = lp_polish(pert2, seconds=min(1.5, rem - 0.5))
                vv2 = max_cos(cand2)
                if vv2 < best_val - 1e-14:
                    best, best_val = cand2, vv2
                else:
                    break
        except Exception:
            break
    best = normed(best)
    return best

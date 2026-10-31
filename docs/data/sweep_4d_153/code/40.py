import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 else k1
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        # exp weights, max=1 for stability
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        # bias correction
        if mu > 0:
            vc = v / (1.0 - mu ** (t + 1))
        else:
            vc = v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x = x / nrm
    return x


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        # keep LP small
        if np.count_nonzero(sel) > 2500:
            # keep tightest 2500
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        c = c_all[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def farthest_candidates(Xkeep, rng, n_new, n_samp=2500):
    out = []
    cur = Xkeep
    for _ in range(n_new):
        cand = rng.standard_normal((n_samp, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ cur.T
        md = dots.max(axis=1)
        best_i = int(np.argmin(md))
        best_pt = cand[best_i]
        out.append(best_pt)
        cur = np.vstack([cur, best_pt[None, :]])
    return np.array(out)


def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(42)
    n, d = 153, 4
    try:
        base = np.load("best.npy")
    except Exception:
        try:
            base = np.load("record.npy")
        except Exception:
            base = rng.standard_normal((n, d))
    base = np.asarray(base, dtype=float)
    if base.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        # try to salvage
        m = min(base.shape[0], n)
        tmp[:m] = base[:m, :d] if base.shape[1] >= d else base[:m]
        base = tmp
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    incumbent = base.copy()
    inc_val = max_cos(incumbent)

    # quick ensure local optimum briefly (2s) if time allows
    # we skip long initial polish to save time for exploration;
    # do 2s polish
    if time.time() - t0 < BUDGET - 30:
        incumbent = lp_polish(incumbent, 2.0)
        inc_val = max_cos(incumbent)

    FINAL_RESERVE = 22.0
    SEARCH_END = t0 + BUDGET - FINAL_RESERVE
    trial = 0
    # candidate pool for diversity
    pool = [incumbent.copy()]

    while time.time() < SEARCH_END:
        trial += 1
        time_left = SEARCH_END - time.time()
        if time_left < 1.0:
            break
        # choose base: mostly incumbent
        rb = rng.random()
        if rb < 0.70:
            b = incumbent
        elif rb < 0.90:
            b = pool[int(rng.integers(len(pool)))]
        else:
            b = base
        x = b.copy()
        mode = rng.random()
        if mode < 0.30:
            # global jitter
            sigma = float(rng.choice(np.array([0.015, 0.04, 0.07, 0.11, 0.18, 0.28])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            iters = int(rng.integers(500, 1100))
            k0 = float(rng.choice(np.array([8.0, 15.0, 25.0])))
            k1 = float(rng.choice(np.array([250.0, 400.0, 600.0])))
            lr0 = float(rng.choice(np.array([0.03, 0.05, 0.08])))
            lr1 = float(rng.choice(np.array([0.003, 0.006, 0.01])))
            mu = float(rng.choice(np.array([0.70, 0.85, 0.90])))
            noise = float(rng.choice(np.array([0.0, 0.005, 0.015])))
            x = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        elif mode < 0.60:
            # worst-point removal + farthest-void reinsertion
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            k = int(rng.integers(2, 9))
            idx = np.argsort(-maxper)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            # occasionally also drop random rattler-like low-contact point? keep as is
            newpts = farthest_candidates(Xkeep, rng, k, n_samp=2500)
            xnew = np.vstack([Xkeep, newpts])
            # small jitter
            xnew = xnew + 0.012 * rng.standard_normal(xnew.shape)
            xnew /= np.linalg.norm(xnew, axis=1, keepdims=True)
            x = xnew
            iters = int(rng.integers(600, 1200))
            k0 = float(rng.choice(np.array([12.0, 20.0])))
            k1 = float(rng.choice(np.array([300.0, 500.0])))
            lr0, lr1, mu, noise = 0.05, 0.005, 0.85, 0.005
            x = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        elif mode < 0.85:
            # coherent patch shift: move a cap together
            center = rng.standard_normal(4)
            center /= np.linalg.norm(center)
            rad = float(rng.uniform(0.45, 0.95))  # radians
            cosr = float(np.cos(rad))
            dots_c = x @ center
            mask = dots_c > cosr
            cnt = int(mask.sum())
            if cnt < 3 or cnt > 35:
                # fallback to jitter
                sigma = 0.08
                x = x + sigma * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                # random tangent direction at center
                r = rng.standard_normal(4)
                r -= (r @ center) * center
                rn = np.linalg.norm(r)
                if rn < 1e-9:
                    r = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    r /= rn
                step = float(rng.uniform(0.06, 0.28))
                x[mask] = x[mask] + step * r[None, :]
                # individual jitter inside patch + outside tiny
                x[mask] += 0.02 * rng.standard_normal((cnt, 4))
                x[~mask] += 0.005 * rng.standard_normal(((n - cnt), 4))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            iters = int(rng.integers(600, 1100))
            k0 = 15.0
            k1 = float(rng.choice(np.array([300.0, 500.0])))
            x = smooth_relax(x, iters, k0, k1, 0.05, 0.005, 0.85, 0.008, rng)
        else:
            # strong shake high-temp anneal
            sigma = float(rng.uniform(0.20, 0.35))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            iters = int(rng.integers(900, 1400))
            x = smooth_relax(x, iters, 8.0, 350.0, 0.08, 0.004, 0.88, 0.02, rng)

        v = max_cos(x)
        # decide to LP polish: if promising
        # threshold adapts: always polish if better than incumbent, else sometimes
        do_polish = False
        if v < inc_val:
            do_polish = True
        elif v < inc_val + 0.008 and (rng.random() < 0.25 or trial % 7 == 0):
            do_polish = True
        elif rng.random() < 0.03:
            do_polish = True
        if do_polish:
            # short polish 1.2s
            remain = SEARCH_END - time.time()
            if remain < 1.5:
                # no time for polish, just check
                if v < inc_val:
                    incumbent = x / np.linalg.norm(x, axis=1, keepdims=True)
                    inc_val = v
                    pool.append(incumbent.copy())
                    if len(pool) > 6:
                        pool.pop(0)
                continue
            xp = lp_polish(x, min(1.5, remain - 0.2))
            vp = max_cos(xp)
            if vp < inc_val:
                incumbent = xp
                inc_val = vp
                pool.append(incumbent.copy())
                if len(pool) > 6:
                    pool.pop(1 if len(pool) > 2 else 0)
            elif vp < inc_val + 0.003 and len(pool) < 6 and rng.random() < 0.3:
                pool.append(xp.copy())
                if len(pool) > 6:
                    pool.pop(0)
            # also keep smooth best if LP failed but smooth good?
            if v < inc_val:
                incumbent = x / np.linalg.norm(x, axis=1, keepdims=True)
                inc_val = v
                pool.append(incumbent.copy())
                if len(pool) > 6:
                    pool.pop(0)
        else:
            if v < inc_val:
                incumbent = x / np.linalg.norm(x, axis=1, keepdims=True)
                inc_val = v
                pool.append(incumbent.copy())
                if len(pool) > 6:
                    pool.pop(0)

    # final long polish of incumbent
    time_left = (t0 + BUDGET) - time.time() - 1.0
    if time_left > 2.0:
        incumbent = lp_polish(incumbent, min(time_left, 25.0))
        inc_val = max_cos(incumbent)
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    # tiny shrink guard not needed; ensure finite
    incumbent = np.nan_to_num(incumbent, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(incumbent, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    incumbent = incumbent / nr
    return incumbent

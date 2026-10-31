import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _degrees(x, tol=1e-9):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    mask = g > m - tol
    deg = np.sum(mask, axis=1)
    maxper = np.max(g, axis=1)
    return deg, maxper, m


def _smooth(x, iters, rng, k0=15.0, k1=250.0, lr0=0.04, lr1=0.002, mu=0.85, noise=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(np.max(G))
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
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
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if seconds <= 0.01:
        return x
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while (time.time() - start) < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(nrm)) or np.any(nrm < 1e-12):
            radius *= 0.5
            continue
        y = y / nrm
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _farthest_insert_candidates(Xkeep, rng, nc=20000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((nc, d))
    nrm = np.linalg.norm(cand, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    cand = cand / nrm
    dots = cand @ Xkeep.T
    md = np.max(dots, axis=1)
    pick = int(np.argmin(md))
    return cand[pick]


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 142.0
    rng_master = np.random.default_rng(12345)
    n_target, d = 153, 4

    try:
        base0 = np.load("best.npy")
    except Exception:
        try:
            base0 = np.load("record.npy")
        except Exception:
            base0 = rng_master.standard_normal((n_target, d))
    base0 = np.asarray(base0, dtype=float)
    if base0.shape != (n_target, d):
        tmp = rng_master.standard_normal((n_target, d))
        # try to salvage
        try:
            m = min(base0.shape[0], n_target)
            tmp[:m] = base0.flat[:m * d].reshape(m, d) if base0.size >= m * d else tmp[:m]
        except Exception:
            pass
        base0 = tmp
    base0 = base0 / np.linalg.norm(base0, axis=1, keepdims=True)
    # clean non-finite
    if not np.all(np.isfinite(base0)):
        trng = np.random.default_rng(0)
        base0 = trng.standard_normal((n_target, d))
        base0 /= np.linalg.norm(base0, axis=1, keepdims=True)
    try:
        base_val = _max_cos(base0)
    except Exception:
        base_val = 1.0

    incumbent = base0.copy()
    inc_val = base_val
    results = [(base_val, base0.copy())]

    def time_left():
        return DEADLINE - time.time()

    def strip_and_rebuild(start_x, target_low, remove_mode, seed, overshoot_to=None):
        nonlocal incumbent, inc_val, results
        rng = np.random.default_rng(seed)
        x = start_x.copy()
        # strip down
        while x.shape[0] > target_low:
            if time.time() > SEARCH_END:
                break
            try:
                deg, maxper, _ = _degrees(x)
            except Exception:
                break
            ncur = x.shape[0]
            if remove_mode == "max":
                # highest contact; tie-break by maxper
                order = np.lexsort((-maxper, -deg))
                rm = int(order[0])
            else:
                # random among high-contact top 12
                order = np.lexsort((-maxper, -deg))
                K = min(12, ncur)
                top = order[:K]
                # weight by deg
                rm = int(rng.choice(top))
            keep = np.ones(ncur, dtype=bool)
            keep[rm] = False
            x = x[keep]
            try:
                x = _smooth(x, 250, rng)
            except Exception:
                pass
            try:
                x = _lp_polish(x, 0.5)
            except Exception:
                pass
        # optionally overshoot handled by caller via build_up
        goal = overshoot_to if overshoot_to is not None else n_target
        while x.shape[0] < goal:
            if time.time() > SEARCH_END:
                break
            try:
                new_pt = _farthest_insert_candidates(x, rng, nc=20000)
            except Exception:
                # fallback smaller batch
                try:
                    new_pt = _farthest_insert_candidates(x, rng, nc=5000)
                except Exception:
                    break
            x = np.vstack([x, new_pt.reshape(1, -1)])
            try:
                x = _smooth(x, 350, rng)
            except Exception:
                pass
            try:
                x = _lp_polish(x, 0.5)
            except Exception:
                pass
        return x

    def prune_to_target(x, target, seed):
        rng = np.random.default_rng(seed + 999)
        x = x.copy()
        while x.shape[0] > target:
            if time.time() > SEARCH_END:
                break
            ncur = x.shape[0]
            # find removal giving lowest max after removal (quick eval, then brief polish)
            best_rm = -1
            best_m = 1e9
            # fast scan of max without i
            for i in range(ncur):
                keep = np.ones(ncur, dtype=bool)
                keep[i] = False
                xs = x[keep]
                try:
                    v = _max_cos(xs)
                except Exception:
                    v = 1e9
                if v < best_m:
                    best_m = v
                    best_rm = i
            keep = np.ones(ncur, dtype=bool)
            keep[best_rm] = False
            x = x[keep]
            try:
                x = _smooth(x, 250, rng)
            except Exception:
                pass
            try:
                x = _lp_polish(x, 0.5)
            except Exception:
                pass
        return x

    # plan branches: (target_low, remove_mode, seed, overshoot_to)
    branches = [
        (150, "max", 101, None),
        (149, "max", 102, None),
        (148, "max", 103, None),
        (150, "rand", 201, None),
        (149, "rand", 202, None),
        (151, "rand", 203, None),
        (152, "max", 104, None),
        (153, "max", 301, 155),  # overshoot to 155 then prune
        (153, "max", 302, 156),
        (153, "max", 303, 154),
        (150, "rand", 204, None),
        (148, "rand", 205, None),
    ]

    for (tlow, rmode, seed, over) in branches:
        if time.time() > SEARCH_END - 3.0:
            break
        if time_left() < 8.0:
            break
        try:
            # vary start: mostly incumbent, sometimes base0
            if len(results) >= 2 and (seed % 3 == 0):
                # start from second best for diversity
                srt = sorted(results, key=lambda z: z[0])
                start = srt[1][1].copy() if len(srt) > 1 else incumbent.copy()
            elif seed % 4 == 0:
                start = base0.copy()
            else:
                start = incumbent.copy()
            x = strip_and_rebuild(start, tlow, rmode, seed, overshoot_to=over)
            # if overshoot, prune back to 153
            if over is not None and x.shape[0] >= over:
                x = prune_to_target(x, n_target, seed)
            # if still below 153 (early stop), build up remaining
            while x.shape[0] < n_target and time.time() < SEARCH_END:
                rngb = np.random.default_rng(seed + x.shape[0] * 77)
                try:
                    new_pt = _farthest_insert_candidates(x, rngb, nc=20000)
                except Exception:
                    break
                x = np.vstack([x, new_pt.reshape(1, -1)])
                try:
                    x = _smooth(x, 350, rngb)
                except Exception:
                    pass
                try:
                    x = _lp_polish(x, 0.4)
                except Exception:
                    pass
            if x.shape[0] != n_target:
                continue
            # final touch for this branch: short LP
            try:
                x = _lp_polish(x, 0.8)
            except Exception:
                pass
            try:
                v = _max_cos(x)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            results.append((v, x.copy()))
            results = sorted(results, key=lambda z: z[0])[:6]
            if v < inc_val - 1e-15:
                inc_val = v
                incumbent = x.copy()
        except Exception:
            continue

    # retain every intermediate 153 result: results already holds them
    results = sorted(results, key=lambda z: z[0])[:6]
    best153 = results[0][1].copy()
    best153_val = results[0][0]

    # reserve final 25-30s for polishing best one
    remaining = DEADLINE - time.time() - 1.0
    if remaining < 5.0:
        remaining = min(5.0, max(1.0, DEADLINE - time.time() - 0.5))
    # polish best longest; if time permits also try second best briefly
    try:
        budget_best = min(remaining, 28.0)
        # if we have plenty, split: 75% best, 25% second
        if len(results) >= 2 and remaining > 12.0:
            b2 = min(6.0, remaining * 0.25)
            budget_best = remaining - b2
            try:
                x2 = _lp_polish(results[1][1].copy(), b2)
                v2 = _max_cos(x2)
                if np.isfinite(v2):
                    results.append((v2, x2.copy()))
            except Exception:
                pass
        best153 = _lp_polish(best153, budget_best)
        try:
            best153_val = _max_cos(best153)
            results.append((best153_val, best153.copy()))
        except Exception:
            pass
    except Exception:
        pass

    results = sorted([r for r in results if np.all(np.isfinite(r[1]))], key=lambda z: z[0])
    final = None
    for (v, xc) in results:
        try:
            if xc.shape == (n_target, d) and np.all(np.isfinite(xc)):
                nrm = np.linalg.norm(xc, axis=1, keepdims=True)
                if np.min(nrm) > 1e-9:
                    final = xc / nrm
                    break
        except Exception:
            continue
    if final is None:
        final = base0.copy()
    # fallback to loaded record if branch fails / not improving? keep best found
    try:
        fv = _max_cos(final)
        if not np.isfinite(fv) or fv > base_val + 1e-12:
            # still return best found (fallback to record guarantees validity)
            # prefer the better of final and base0
            if base_val <= fv:
                final = base0.copy()
    except Exception:
        final = base0.copy()
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    return final

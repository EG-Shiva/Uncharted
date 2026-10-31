import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _degrees(x, window=0.002):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = np.count_nonzero(G > m - window, axis=1)
    return deg, m, G


def _smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    x = _norm_rows(x.copy())
    n = x.shape[0]
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        import math
        alpha = float(math.exp(math.log(a0) + (math.log(a1) - math.log(a0)) * frac))
        smax = float(s0 * (s1 / s0) ** frac) if s0 != s1 else float(s0)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights, shift for stability
        W = np.exp(np.clip(alpha * (G - m), -50.0, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) * (1.0 / (0.5 * S + 1e-300))
        # project
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if not np.isfinite(nrm) or nrm < 1e-14:
            continue
        # cap huge steps
        x = x - (smax / nrm) * vel
        x = _norm_rows(x)
    return x


def _refine_hole(Y, c0, beta=60.0, steps=220, s0=0.06, s1=0.005):
    c = c0 / (np.linalg.norm(c0) + 1e-300)
    vel = np.zeros(4)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        s = s0 * (s1 / s0) ** frac
        dots = Y @ c
        m = float(dots.max())
        w = np.exp(np.clip(beta * (dots - m), -50.0, 0.0))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w = w / sw
        g = w @ Y
        gt = g - float(np.dot(g, c)) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = 0.5 * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        c = c - (s / vn) * vel
        n = float(np.linalg.norm(c))
        if n == 0 or not np.isfinite(n):
            break
        c = c / n
    return c / np.linalg.norm(c)


def _largest_hole(Y, rng, n_samples=50000, beta=60.0):
    d = Y.shape[1]
    best_s = 2.0
    best_c = None
    bs = 10000
    # batched search to bound memory
    for a in range(0, n_samples, bs):
        nb = min(bs, n_samples - a)
        C = rng.standard_normal((nb, d))
        C = _norm_rows(C)
        S = C @ Y.T
        sm = S.max(axis=1)
        k = int(np.argmin(sm))
        if float(sm[k]) < best_s:
            best_s = float(sm[k])
            best_c = C[k].copy()
    if best_c is None:
        best_c = rng.standard_normal(4)
        best_c /= np.linalg.norm(best_c)
    best_c = _refine_hole(Y, best_c, beta=beta, steps=220)
    return best_c


def _subset_last_opt(Y, steps=150, alpha=60.0, smax=0.05, mu=0.5):
    # optimize only last row, others frozen
    Y = _norm_rows(Y.copy())
    n = Y.shape[0]
    last = n - 1
    vel = np.zeros(4)
    F = Y[:-1]
    c = Y[last].copy()
    for it in range(steps):
        dots = F @ c
        m = float(dots.max())
        w = np.exp(np.clip(alpha * (dots - m), -50.0, 0.0))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w = w / sw
        g = w @ F
        gt = g - float(np.dot(g, c)) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = mu * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        # decay smax slightly
        s = smax * (0.995 ** it)
        c = c - (s / vn) * vel
        c = c / np.linalg.norm(c)
    Y[last] = c
    return _norm_rows(Y)


def _lp_polish(x, deadline):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline - 0.05 and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    rng_master = np.random.default_rng(20261031)

    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (272, 4):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    X_best = a.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _norm_rows(rng_master.standard_normal((272, 4)))
        best_val = _max_cos(X_best)
    X0 = X_best.copy()

    # build deletion sets
    deg, m0, G0 = _degrees(X0, window=0.002)
    order_deg = np.argsort(-deg, kind="stable")
    seed_c = int(np.argmax(deg))
    dots_seed = G0[seed_c].copy()
    dots_seed[seed_c] = 2.0
    order_near_seed = np.argsort(-dots_seed, kind="stable")

    rng_tmp = np.random.default_rng(777)
    rand32 = rng_tmp.choice(272, size=32, replace=False)
    rng_tmp2 = np.random.default_rng(778)
    rand28 = rng_tmp2.choice(272, size=28, replace=False)

    trials = []
    trials.append(("deg32", order_deg[:32].copy()))
    trials.append(("cluster32", np.concatenate([[seed_c], order_near_seed[:31]]).copy()))
    trials.append(("rand32", rand32.copy()))
    trials.append(("deg24", order_deg[:24].copy()))
    trials.append(("deg40", order_deg[:40].copy()))
    trials.append(("rand28", rand28.copy()))

    # time slices: allocate roughly equal, leave 25s for final polish
    # estimate per-trial budget
    n_trials = len(trials)
    # we will iterate and skip if not enough time
    for ti, (name, del_idx) in enumerate(trials):
        if time.time() > T_END - 28.0:
            break
        # per-trial deadline (leave time for remaining trials + final)
        remaining_trials = n_trials - ti
        time_left = T_END - time.time() - 22.0  # reserve final
        if time_left < 5.0:
            break
        trial_budget = time_left / remaining_trials
        trial_budget = min(max(trial_budget, 8.0), 28.0)
        T_TRIAL_END = time.time() + trial_budget
        try:
            D = len(del_idx)
            mask = np.ones(272, dtype=bool)
            mask[np.asarray(del_idx, dtype=int)] = False
            Y = X0[mask].copy()  # Nrem x 4  (but use current X_best? use X0 for diversity, occasionally X_best)
            # every other trial start from current best topology
            if ti % 2 == 1 and X_best.shape == (272, 4):
                # recompute deletion on X_best for variety: delete same rank positions on X_best degrees
                deg2, _, G2 = _degrees(X_best, window=0.002)
                if name.startswith("deg"):
                    od2 = np.argsort(-deg2, kind="stable")
                    mask2 = np.ones(272, dtype=bool)
                    mask2[od2[:D]] = False
                    Y = X_best[mask2].copy()
                elif name.startswith("cluster"):
                    s2 = int(np.argmax(deg2))
                    ds = (X_best @ X_best[s2]).copy()
                    ds[s2] = 2.0
                    on2 = np.argsort(-ds, kind="stable")
                    del2 = np.concatenate([[s2], on2[:D - 1]])
                    mask2 = np.ones(272, dtype=bool)
                    mask2[del2] = False
                    Y = X_best[mask2].copy()
                else:
                    # random: use X_best with same random idx
                    mask2 = np.ones(272, dtype=bool)
                    mask2[np.asarray(del_idx, dtype=int)] = False
                    Y = X_best[mask2].copy()

            # Stage 1: anneal remaining code well below record
            # use trial-specific seed
            rng = np.random.default_rng(9000 + ti * 131)
            # light jitter before anneal to leave basin
            Y = Y + 0.002 * rng.standard_normal(Y.shape)
            Y = _norm_rows(Y)
            Y = _smooth_gd(Y, steps=600, a0=12.0, a1=400.0, s0=0.030, s1=0.003, mu=0.85)
            if time.time() > T_TRIAL_END - 6.0:
                # still need regrow; skip to regrow with fewer steps
                pass
            # optional quick LP on reduced set if very high? skip to save time
            # Stage 2: reinsert one by one
            for k in range(D):
                if time.time() > T_TRIAL_END - 8.0:
                    break
                c = _largest_hole(Y, rng, n_samples=50000, beta=60.0)
                Y = np.vstack([Y, c[None, :]])
                Y = _subset_last_opt(Y, steps=120, alpha=60.0, smax=0.05, mu=0.5)
                if ((k + 1) % 6 == 0) or ((k + 1) == D):
                    # fully re-anneal after every 6
                    if (k + 1) == D:
                        Y = _smooth_gd(Y, steps=500, a0=14.0, a1=500.0, s0=0.020, s1=0.002, mu=0.86)
                    else:
                        Y = _smooth_gd(Y, steps=350, a0=14.0, a1=350.0, s0=0.020, s1=0.003, mu=0.85)
                if Y.shape[0] != (272 - D + k + 1):
                    break
            if Y.shape[0] != 272:
                continue
            # final anneal sharpen
            if time.time() < T_TRIAL_END - 5.0:
                Y = _smooth_gd(Y, steps=450, a0=10.0, a1=800.0, s0=0.014, s1=0.0015, mu=0.88)
            v = _max_cos(Y)
            # short LP polish within trial budget
            lp_dl = min(T_TRIAL_END, time.time() + 4.0)
            if lp_dl > time.time() + 0.8 and v < best_val + 0.003:
                Y2 = _lp_polish(Y, lp_dl)
                v2 = _max_cos(Y2)
                if v2 < v:
                    Y, v = Y2, v2
            if v < best_val:
                best_val = v
                X_best = Y.copy()
                # also update X0 to new best for subsequent trials to exploit? keep X0 as new best partially
                # keep original X0 for diversity, but update a copy
                # do not overwrite X0 entirely to preserve diversity; instead set X0 = X_best for next cluster trials
                X0 = X_best.copy()
                deg, m0, G0 = _degrees(X0, window=0.002)
        except Exception:
            continue

    # Final: LP polish best with remaining time
    try:
        if time.time() < T_END - 2.0:
            dl = T_END - 0.5
            # small final jitter + anneal if stuck? just LP
            X_best = _lp_polish(X_best, dl)
            best_val = _max_cos(X_best)
    except Exception:
        pass

    # safety: if somehow worse than loaded best, reload
    try:
        for fn in ("best.npy", "record.npy"):
            try:
                a = _norm_rows(np.load(fn).astype(float))
                if a.shape == (272, 4) and _max_cos(a) < _max_cos(X_best):
                    X_best = a.copy()
            except Exception:
                pass
    except Exception:
        pass

    X_best = _norm_rows(np.asarray(X_best, dtype=np.float64))
    # tiny shrink not needed; ensure unit
    return X_best

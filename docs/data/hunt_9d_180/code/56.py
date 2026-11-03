import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def get_marginals(res, p):
    try:
        m = res.ineqlin.marginals
        m = np.asarray(m, dtype=float).ravel()
        if m.size == p:
            return m
    except Exception:
        pass
    try:
        # dict-like access
        m = res.ineqlin["marginals"]
        m = np.asarray(m, dtype=float).ravel()
        if m.size == p:
            return m
    except Exception:
        pass
    try:
        m = getattr(res, "marginals", None)
        if m is not None:
            m = np.asarray(m, dtype=float).ravel()
            if m.size == p:
                return m
    except Exception:
        pass
    return None


def lp_polish_capped(x, seconds, cap=3000, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_full = G[iu0, iu1]
        m = float(c_full.max())
        sel = c_full > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]
        J = iu1[sel]
        c = c_full[sel]
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
        try:
            step = np.asarray(res.x[:-1], dtype=float).reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def lp_polish_with_duals(x, seconds, cap=3000, radius_init=1e-3):
    """Polish and also return per-point dual stress from last successful LP."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    last_marg = None
    last_I = None
    last_J = None
    last_c = None
    last_m = best
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_full = G[iu0, iu1]
        m = float(c_full.max())
        sel = c_full > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]
        J = iu1[sel]
        c = c_full[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        marg = get_marginals(res, p)
        if marg is not None:
            last_marg = marg.copy()
            last_I = I.copy()
            last_J = J.copy()
            last_c = c.copy()
            last_m = m
        try:
            step = np.asarray(res.x[:-1], dtype=float).reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    # aggregate dual weight per point
    w = np.zeros(n)
    if last_marg is not None:
        am = np.abs(last_marg)
        # only slack < 1e-7
        slack = last_m - last_c
        for k in range(len(am)):
            if slack[k] < 1e-7:
                a = am[k]
                if a != 0.0 and np.isfinite(a):
                    w[last_I[k]] += a
                    w[last_J[k]] += a
    return x, w


def refine_candidates(C, F, steps=300):
    k = C.shape[0]
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = 20.0 + (260.0 - 20.0) * frac
        lr = 0.06 - 0.03 * frac
        D = C @ F.T
        Dm = D.max(axis=1, keepdims=True)
        E = np.exp(alpha * (D - Dm))
        S = E.sum(axis=1, keepdims=True) + 1e-30
        W = E / S
        Grad = W @ F
        dot = (Grad * C).sum(axis=1, keepdims=True)
        Gt = Grad - dot * C
        # clip extreme steps
        C = C - lr * Gt
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    return C


def surgery_reinsert(frozen, n_add, rng, n_rand=60000, n_best=32, steps=300):
    F = frozen.copy()
    d = F.shape[1]
    for _ in range(n_add):
        m = F.shape[0]
        cand = rng.randn(n_rand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30
        # vectorial scoring
        dots = cand @ F.T
        md = dots.max(axis=1)
        if n_best >= n_rand:
            sel_idx = np.argsort(md)[:n_best]
        else:
            part = np.argpartition(md, n_best)[:n_best]
            # sort them for determinism
            order = np.argsort(md[part])
            sel_idx = part[order]
        C = cand[sel_idx].copy()
        try:
            C = refine_candidates(C, F, steps=steps)
        except Exception:
            pass
        # true max dot
        d2 = C @ F.T
        m2 = d2.max(axis=1)
        bi = int(np.argmin(m2))
        F = np.vstack([F, C[bi:bi + 1] / np.linalg.norm(C[bi:bi + 1], axis=1, keepdims=True)])
    return F


def solve():
    np.random.seed(0)
    t0 = time.time()
    deadline = t0 + 176.0
    explore_deadline = t0 + 140.0
    n, d = 180, 9
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    # 1) polish once with cap 3000, extract duals
    try:
        remain = explore_deadline - time.time()
        secs0 = min(6.0, max(2.0, remain - 100.0)) if remain > 10 else min(3.0, max(1.0, remain / 4))
        # ensure at least small polish for duals
        secs0 = float(np.clip(secs0, 1.5, 6.0))
        polished, dual_w = lp_polish_with_duals(best, secs0, cap=3000, radius_init=1e-3)
        vpol = max_cos(polished)
        if vpol < best_val and np.isfinite(vpol):
            best = polished
            best_val = vpol
        else:
            # keep polished for dual topology even if not better? use best for duals
            # if polish didn't improve, still use its duals but keep best points
            pass
    except Exception:
        dual_w = np.zeros(n)
    # Fallback contact-based stress if duals all zero
    try:
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        contact_mask = G > gm - 1e-9
        # count per point (upper+lower double counts, halve)
        cnt = contact_mask.sum(axis=1).astype(float)
    except Exception:
        cnt = np.zeros(n)
        gm = best_val
    # rattlers: 0 contacts
    try:
        # per-point max for rattler detect
        G2 = best @ best.T
        np.fill_diagonal(G2, -2.0)
        ppm = G2.max(axis=1)
        # rattler if ppm < gm - 1e-9
        rattler_idx = np.where(ppm < gm - 1e-9)[0]
        if len(rattler_idx) < 2:
            order_c = np.argsort(cnt)  # fewest contacts first
            rattler_idx = order_c[:2]
        elif len(rattler_idx) > 2:
            # take 2 with smallest ppm
            o = np.argsort(ppm[rattler_idx])[:2]
            rattler_idx = rattler_idx[o]
    except Exception:
        rattler_idx = np.argsort(cnt)[:2]
    rattler_idx = np.asarray(rattler_idx, dtype=int).ravel()[:2]
    # if dual_w all zero, use cnt as stress
    if np.all(dual_w == 0) or not np.all(np.isfinite(dual_w)):
        dual_w = cnt.astype(float)
    # hub order excluding rattlers
    hub_order = np.argsort(-dual_w)
    hub_order = np.array([i for i in hub_order if i not in set(rattler_idx.tolist())], dtype=int)
    # configs: (hub_count, mode, seed)
    configs = [
        (4, "top", 100),
        (3, "top", 101),
        (5, "top", 102),
        (7, "top", 103),
        (4, "q90", 104),
        (5, "q75", 105),
        (3, "q90", 106),
        (7, "q90", 107),
    ]
    # quantile pools
    try:
        q90 = float(np.quantile(dual_w, 0.90))
        q75 = float(np.quantile(dual_w, 0.75))
    except Exception:
        q90 = float(np.sort(dual_w)[int(0.9 * n)])
        q75 = float(np.sort(dual_w)[int(0.75 * n)])
    for (hk, mode, seed) in configs:
        if time.time() >= explore_deadline:
            break
        # select hubs
        try:
            if mode == "top":
                hubs = hub_order[:hk]
            elif mode == "q90":
                pool = np.where(dual_w >= q90)[0]
                pool = np.array([i for i in pool if i not in set(rattler_idx.tolist())], dtype=int)
                if len(pool) < hk:
                    pool = hub_order[:max(hk, 18)]
                rng_sel = np.random.RandomState(seed)
                # deterministic choice
                perm = rng_sel.permutation(len(pool))
                # prefer high stress: sort pool by dual desc then take permuted among top?
                # simple: take random hk from pool
                hubs = np.sort(pool[perm[:hk]])
            elif mode == "q75":
                pool = np.where(dual_w >= q75)[0]
                pool = np.array([i for i in pool if i not in set(rattler_idx.tolist())], dtype=int)
                if len(pool) < hk:
                    pool = hub_order[:max(hk, 30)]
                rng_sel = np.random.RandomState(seed)
                perm = rng_sel.permutation(len(pool))
                hubs = np.sort(pool[perm[:hk]])
            else:
                hubs = hub_order[:hk]
            hubs = np.asarray(hubs, dtype=int).ravel()
            remove = np.unique(np.concatenate([hubs, rattler_idx]))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[remove] = False
            frozen = best[keep_mask].copy()
            n_add = int(n - frozen.shape[0])
            if n_add <= 0:
                continue
            rng = np.random.RandomState(seed + 1000)
            # reinsert one at a time
            # adapt n_rand if time short
            time_left = explore_deadline - time.time()
            if time_left < 12.0:
                n_rand_use = 30000
                steps_use = 200
            else:
                n_rand_use = 60000
                steps_use = 300
            full = surgery_reinsert(frozen, n_add, rng, n_rand=n_rand_use, n_best=32, steps=steps_use)
            # full has 180 points
            v_before = max_cos(full)
            # capped LP polish at most 8s
            time_left2 = explore_deadline - time.time()
            if time_left2 < 2.0:
                break
            secs = min(8.0, max(2.0, time_left2 - 1.0))
            # if clearly worse, still give short polish only
            if v_before > best_val + 0.004:
                secs = min(secs, 3.0)
            try:
                full2 = lp_polish_capped(full, secs, cap=3000, radius_init=0.004)
            except Exception:
                full2 = full
            v_after = max_cos(full2)
            if np.isfinite(v_after) and v_after < best_val:
                best = full2 / np.linalg.norm(full2, axis=1, keepdims=True)
                best_val = float(v_after)
                # refresh duals/cnts/rattlers for subsequent surgeries? keep original topology per spec?
                # update hub order to new best's contacts for diversity but keep rattlers updated
                try:
                    G = best @ best.T
                    np.fill_diagonal(G, -2.0)
                    gm = float(G.max())
                    cnt = (G > gm - 1e-9).sum(axis=1).astype(float)
                    ppm = G.max(axis=1)
                    ri = np.where(ppm < gm - 1e-9)[0]
                    if len(ri) < 2:
                        ri = np.argsort(cnt)[:2]
                    elif len(ri) > 2:
                        ri = ri[np.argsort(ppm[ri])[:2]]
                    rattler_idx = np.asarray(ri, dtype=int).ravel()[:2]
                    # update dual_w partially: keep old but boost? recompute quick contact-based?
                    # keep dual_w as cnt for next iters to reflect new best
                    # but spec says extract once; we blend: keep original dual order for hubs except refresh
                    # use cnt as new dual_w for remaining configs
                    dual_w = cnt.astype(float)
                    hub_order = np.argsort(-dual_w)
                    hub_order = np.array([i for i in hub_order if i not in set(rattler_idx.tolist())], dtype=int)
                    try:
                        q90 = float(np.quantile(dual_w, 0.90))
                        q75 = float(np.quantile(dual_w, 0.75))
                    except Exception:
                        pass
                except Exception:
                    pass
            if time.time() >= explore_deadline:
                break
        except Exception:
            continue
    # final polish with remaining time (keep total < deadline)
    remain = deadline - time.time() - 0.8
    if remain > 2.0:
        try:
            secs_f = min(remain, 20.0)
            best = lp_polish_capped(best, secs_f, cap=3000 if remain > 8 else 6000, radius_init=8e-4)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # shrink by tiny margin not needed; ensure finite
    best[~np.isfinite(best)] = 0.0
    # renormalize after fix
    nr = np.linalg.norm(best, axis=1, keepdims=True) + 1e-30
    best = best / nr
    return best

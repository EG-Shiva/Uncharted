import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish_bd(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    t_start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        now = time.time()
        if now - t_start > seconds:
            break
        if now > deadline - 0.8:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2600:
            idx = np.where(sel)[0]
            part = np.argpartition(c_all[idx], -2600)[-2600:]
            sel[:] = False
            sel[idx[part]] = True
            cnt = 2600
        I = iu[0][sel]
        J = iu[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_b = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_b[:, :d] = I[:, None] * d + ar[None, :]
        cols_b[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_b[:, 2 * d] = n * d
        vals_b = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_b[:, :d] = x[J]
        vals_b[:, d:2 * d] = x[I]
        vals_b[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals_b.ravel(), (rows, cols_b.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        b_ub = -c_all[sel]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
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
        nrm[nrm == 0] = 1.0
        y /= nrm
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def best_void_20k(X_fixed, rng, n_cand=20000):
    d = X_fixed.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    dots = cand @ X_fixed.T
    scores = dots.max(axis=1)
    i = int(np.argmin(scores))
    return cand[i], float(scores[i])


def refine_void(v0, X_fixed, rng, iters=700):
    best = v0 / (np.linalg.norm(v0) + 1e-30)
    try:
        best_s = float((X_fixed @ best).max())
    except Exception:
        return best
    for k in range(iters):
        frac = k / max(1, iters - 1)
        sigma = 0.08 * (1.0 - frac) + 0.002 * frac
        prop = best + rng.standard_normal(size=best.shape) * sigma
        n = float(np.linalg.norm(prop))
        if n < 1e-12 or not np.isfinite(n):
            continue
        prop /= n
        s = float((X_fixed @ prop).max())
        if s < best_s:
            best, best_s = prop, s
    return best


def local_adam(X_all, active, steps=250, T=0.05, lr=0.005):
    # fixed number of tangential Adam steps on active subset only
    Xcur = X_all.copy()
    m = len(active)
    if m == 0:
        return Xcur
    active = np.asarray(active, dtype=np.int64)
    m1 = np.zeros((m, Xcur.shape[1]), dtype=np.float64)
    m2 = np.zeros((m, Xcur.shape[1]), dtype=np.float64)
    b1, b2, eps = 0.9, 0.999, 1e-8
    N = Xcur.shape[0]
    for s in range(steps):
        Xa = Xcur[active]
        G = Xa @ Xcur.T
        # exclude self-dots
        G[np.arange(m), active] = -2.0
        # smooth max weights, bounded (G-1)/T in [-60,0]
        E = np.exp((G - 1.0) / T)
        E[~np.isfinite(E)] = 0.0
        grad = E @ Xcur
        dot = (grad * Xa).sum(axis=1, keepdims=True)
        grad -= dot * Xa
        m1 = b1 * m1 + (1.0 - b1) * grad
        m2 = b2 * m2 + (1.0 - b2) * (grad * grad)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        Xa_new = Xa - lr * step
        nrm = np.linalg.norm(Xa_new, axis=1, keepdims=True) + 1e-30
        Xa_new /= nrm
        Xcur[active] = Xa_new
    return Xcur


def contact_degrees(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol).sum(axis=1)
    return C, m


def solve():
    t0 = time.time()
    SURGERY_END = t0 + 110.0
    FINAL_END = t0 + 165.0
    HARD_END = t0 + 175.0
    rng = np.random.RandomState(0)

    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal(size=(180, 9))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (180, 9):
        flat = xb.ravel()
        need = 180 * 9
        if flat.size < need:
            rep = np.tile(flat, int(np.ceil(need / max(1, flat.size))))[:need]
            xb = rep.reshape(180, 9)
        else:
            xb = flat[:need].reshape(180, 9)
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-30)
    try:
        best = max_cos(xb)
    except Exception:
        best = 1.0
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=np.float64)
        if xr.shape == (180, 9):
            xr = xr / (np.linalg.norm(xr, axis=1, keepdims=True) + 1e-30)
            vr = max_cos(xr)
            if vr < best:
                best = vr
                xb = xr
    except Exception:
        pass
    x_best = xb.copy()

    cycle = 0
    n = 180
    while True:
        now = time.time()
        if now > SURGERY_END - 14.0:
            break
        if now > HARD_END - 20.0:
            break
        if cycle >= 25:
            break
        cycle += 1

        try:
            C, mval = contact_degrees(x_best, tol=1e-7)
        except Exception:
            C = np.zeros(n, dtype=int)
        order_asc = np.argsort(C)
        order_desc = order_asc[::-1]

        # two rattlers / lowest-degree
        rattlers = list(order_asc[:2].copy())
        K = 8 + (cycle % 8)  # 8..15 hubs/random
        # choose hub/random part
        mode = cycle % 3
        candidates = [i for i in range(n) if i not in rattlers]
        del_set = list(rattlers)
        if mode == 0:
            # highest-degree hubs
            for idx in order_desc:
                if idx in del_set:
                    continue
                del_set.append(int(idx))
                if len(del_set) >= 2 + K:
                    break
        elif mode == 1:
            # half hubs + half random
            nh = K // 2
            cnt = 0
            for idx in order_desc:
                if idx in del_set:
                    continue
                del_set.append(int(idx))
                cnt += 1
                if cnt >= nh:
                    break
            # random fill
            perm = rng.permutation(candidates)
            for idx in perm:
                if int(idx) in del_set:
                    continue
                del_set.append(int(idx))
                if len(del_set) >= 2 + K:
                    break
        else:
            # random points
            perm = rng.permutation(candidates)
            for idx in perm:
                del_set.append(int(idx))
                if len(del_set) >= 2 + K:
                    break
        del_set = np.array(sorted(set(del_set)), dtype=int)
        # ensure size 10..17
        if len(del_set) < 10:
            # pad random
            perm = rng.permutation(n)
            for idx in perm:
                if int(idx) not in del_set:
                    del_set = np.append(del_set, int(idx))
                if len(del_set) >= 10:
                    break
        if len(del_set) > 17:
            del_set = del_set[:17]

        mask = np.ones(n, dtype=bool)
        mask[del_set] = False
        X_core = x_best[mask].copy()
        k = X_core.shape[0]

        # LP-polish remaining core until internal max drops
        core_time = min(4.0, max(1.0, SURGERY_END - time.time() - 10.0))
        if core_time < 0.8:
            break
        try:
            X_core_pol = lp_polish_bd(X_core, seconds=core_time, deadline=SURGERY_END)
        except Exception:
            continue
        # rebuild one at a time
        try:
            X_cur = X_core_pol.copy()
        except Exception:
            continue
        # random order of reinsertion
        reins_order = rng.permutation(len(del_set))
        ok = True
        for _ii in reins_order:
            if time.time() > SURGERY_END - 8.0:
                ok = False
                break
            if time.time() > HARD_END - 12.0:
                ok = False
                break
            try:
                v0, _s0 = best_void_20k(X_cur, rng, n_cand=20000)
                v0 = refine_void(v0, X_cur, rng, iters=600)
            except Exception:
                continue
            # append
            try:
                X_cur = np.vstack([X_cur, v0.reshape(1, -1)])
            except Exception:
                continue
            new_idx = X_cur.shape[0] - 1
            try:
                dots = X_cur[:-1] @ X_cur[new_idx]
                neigh = np.where(dots > 0.25)[0]
                active = np.append(neigh, new_idx).astype(int)
                # limit active size to avoid blowup (keep closest 60 if too many)
                if len(active) > 70:
                    # keep new + closest 69
                    dsel = dots[neigh]
                    keep = np.argsort(dsel)[-69:]
                    active = np.append(neigh[keep], new_idx).astype(int)
                X_cur = local_adam(X_cur, active, steps=250, T=0.05, lr=0.005)
            except Exception:
                continue
        if not ok:
            break
        if X_cur.shape[0] != 180:
            continue
        # full LP polish only after all restored
        try:
            v_cur = max_cos(X_cur)
        except Exception:
            continue
        # budget full polish depending on promise
        t_rem_surg = SURGERY_END - time.time()
        if t_rem_surg < 2.0:
            break
        if v_cur > best + 0.02:
            full_time = min(2.0, t_rem_surg - 0.5)
        else:
            full_time = min(6.0, t_rem_surg - 0.5)
        if full_time < 0.8:
            continue
        try:
            X_pol = lp_polish_bd(X_cur, seconds=full_time, deadline=SURGERY_END)
        except Exception:
            continue
        try:
            v = max_cos(X_pol)
        except Exception:
            continue
        if np.all(np.isfinite(X_pol)) and v < best - 1e-12:
            best = v
            x_best = (X_pol / (np.linalg.norm(X_pol, axis=1, keepdims=True) + 1e-30)).copy()

    # final polish by 165s, always retain best fallback
    try:
        if time.time() < FINAL_END - 1.5:
            secs = min(FINAL_END - time.time() - 0.5, HARD_END - time.time() - 0.8)
            if secs > 1.0:
                x_best = lp_polish_bd(x_best, seconds=float(secs), deadline=FINAL_END)
    except Exception:
        pass
    x_best = x_best / (np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-30)
    x_best[~np.isfinite(x_best)] = 0.0
    nrm = np.linalg.norm(x_best, axis=1)
    for i in np.where(nrm < 1e-12)[0]:
        v = rng.standard_normal(size=x_best.shape[1])
        v /= (np.linalg.norm(v) + 1e-30)
        x_best[i] = v
    x_best = x_best / (np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-30)
    return x_best

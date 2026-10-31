import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _smooth_global(x, iters, T0, T1, lr0, lr1):
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _relax_moving(base, mov, iters, T0, T1, lr0, lr1):
    # base frozen (Nb,4), mov (K,4) optimized
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        Db = mov @ base.T
        Dm = mov @ mov.T
        np.fill_diagonal(Dm, -2.0)
        m = float(max(Db.max(), Dm.max()))
        Wb = np.exp((Db - m) / T)
        Wm = np.exp((Dm - m) / T)
        np.fill_diagonal(Wm, 0.0)
        grad = Wb @ base + Wm @ mov
        dot = np.sum(grad * mov, axis=1, keepdims=True)
        gt = grad - dot * mov
        rs = (Wb.sum(axis=1, keepdims=True) + Wm.sum(axis=1, keepdims=True))
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        mov = mov - lr * step
        mov /= np.linalg.norm(mov, axis=1, keepdims=True)
    return mov

def _refine_voids_vec(base, pool, steps=20, lr=0.08, sharp=60.0):
    # pool (P,4), base (Nb,4)
    for _ in range(steps):
        D = pool @ base.T  # P x Nb
        M = D.max(axis=1, keepdims=True)
        W = np.exp(sharp * (D - M))
        S = W.sum(axis=1, keepdims=True)
        S = np.maximum(S, 1e-300)
        Gv = (W @ base) / S
        dot = np.sum(Gv * pool, axis=1, keepdims=True)
        Gt = Gv - dot * pool
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        pool = pool - lr * Gt / nrm
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    return pool

def _find_voids(base, rng, k, n_samp=6000):
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ base.T).max(axis=1)
    take = min(n_samp, max(k * 8, k + 12))
    order = np.argpartition(sc, take - 1)[:take]
    pool = cand[order]
    pool = _refine_voids_vec(base, pool, steps=20, lr=0.08, sharp=60.0)
    sc2 = (pool @ base.T).max(axis=1)
    o2 = np.argsort(sc2)
    pool = pool[o2]
    sc2 = sc2[o2]
    if k >= len(pool):
        return pool
    if k == 1:
        return pool[:1]
    sel = [0]
    while len(sel) < k:
        rem = [i for i in range(len(pool)) if i not in sel]
        S = pool[sel] @ pool[rem].T
        mm = S.max(axis=0)
        crit = sc2[rem] + 0.30 * mm
        j = int(np.argmin(crit))
        sel.append(rem[j])
    return pool[np.array(sel)]

def _lp_polish(x, seconds, r0=2e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 500:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = 168.0
    rng = np.random.default_rng(777123)
    n = 163
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (n, 4):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                nr = np.linalg.norm(c, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                c = c / nr
                cands.append(c)
        except Exception:
            pass
    if len(cands) == 0:
        x0 = rng.normal(size=(n, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        x0 = cands[int(np.argmin(vals))].copy()
    x_best = x0.copy()
    best = _max_cos(x_best)
    x_cur = x_best.copy()
    cur = best
    HOP = 108.0
    it = 0
    # precompute random centers buffer
    while time.time() - t0 < HOP and time.time() - t0 < DEAD - 55:
        it += 1
        elapsed = time.time() - t0
        frac = min(1.0, elapsed / HOP)
        x = x_cur.copy()
        mode = it % 7
        if mode in (0, 1, 2, 5):
            # determine K and center
            if mode == 0:
                K = 3
            elif mode == 1:
                K = 5
            elif mode == 2:
                K = 8
            else:
                K = 6
            rm, Gfull = _row_max(x)
            worst = int(np.argmax(rm))
            m = float(rm.max())
            if mode == 5:
                deg = (Gfull > m - 5e-4).sum(axis=1)
                seed = int(np.argmax(deg + 0.1 * rm))
            elif mode == 0 and (it % 14 == 0):
                # random cap to diversify
                rc = rng.normal(size=(4,))
                rc /= np.linalg.norm(rc)
                dots_c = x @ rc
                victims = np.argsort(-dots_c)[:K]
                # jump to reinsert
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                keep_idx = np.where(keep_mask)[0]
                base = x[keep_idx]
                voids = _find_voids(base, rng, K, n_samp=5000)
                voids = voids + rng.normal(scale=6e-4, size=voids.shape)
                voids /= np.linalg.norm(voids, axis=1, keepdims=True)
                mov = voids.copy()
                mov = _relax_moving(base, mov, 70, 0.020, 0.002, 0.09, 0.02)
                x[victims] = mov
                sig = 0.004 * (1.0 - frac) + 0.001
                x[keep_idx] += rng.normal(scale=sig, size=(len(keep_idx), 4))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                T0 = 0.018
                x = _smooth_global(x, 110, T0, 0.0016, 0.05, 0.010)
                v = _max_cos(x)
                tol = 0.0015 * (1.0 - frac)
                if v < cur + tol:
                    x_cur, cur = x, v
                if v < best:
                    x_best, best = x.copy(), v
                if it % 15 == 0 and cur > best + 0.002:
                    x_cur, cur = x_best.copy(), best
                continue
            else:
                seed = worst
                if mode == 2 and (it % 3 == 0):
                    rc = rng.normal(size=(4,))
                    rc /= np.linalg.norm(rc)
                    dots_c = x @ rc
                    # mix: half nearest to random, half worst
                    victims_r = np.argsort(-dots_c)[:K]
                    victims = victims_r
                    keep_mask = np.ones(n, dtype=bool)
                    keep_mask[victims] = False
                    keep_idx = np.where(keep_mask)[0]
                    base = x[keep_idx]
                    voids = _find_voids(base, rng, K, n_samp=5000)
                    voids /= np.linalg.norm(voids, axis=1, keepdims=True)
                    mov = voids + rng.normal(scale=6e-4, size=voids.shape)
                    mov /= np.linalg.norm(mov, axis=1, keepdims=True)
                    mov = _relax_moving(base, mov, 70, 0.020, 0.002, 0.09, 0.02)
                    x[victims] = mov
                    x[keep_idx] += rng.normal(scale=0.003, size=(len(keep_idx), 4))
                    x /= np.linalg.norm(x, axis=1, keepdims=True)
                    x = _smooth_global(x, 110, 0.020, 0.0016, 0.05, 0.010)
                    v = _max_cos(x)
                    tol = 0.0015 * (1.0 - frac)
                    if v < cur + tol:
                        x_cur, cur = x, v
                    if v < best:
                        x_best, best = x.copy(), v
                    if it % 15 == 0 and cur > best + 0.002:
                        x_cur, cur = x_best.copy(), best
                    continue
            # cap around seed
            dseed = x @ x[seed]
            victims = np.argsort(-dseed)[:K]
            # for K=3 also force a low-degree rattler in if exists
            if K == 3:
                deg = (Gfull > m - 5e-4).sum(axis=1)
                ratt = np.where(deg <= 1)[0]
                for rr in ratt:
                    if int(rr) not in victims:
                        victims[-1] = int(rr)
                        break
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            keep_idx = np.where(keep_mask)[0]
            base = x[keep_idx]
            voids = _find_voids(base, rng, K, n_samp=6000)
            voids = voids + rng.normal(scale=6e-4, size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            mov = voids.copy()
            mov = _relax_moving(base, mov, 80, 0.022, 0.002, 0.09, 0.02)
            x[victims] = mov
            sig = (0.006 if K <= 5 else 0.004) * (1.0 - frac) + 0.001
            x[keep_idx] = x[keep_idx] + rng.normal(scale=sig, size=(len(keep_idx), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            T0 = 0.020 if (it % 2 == 0) else 0.013
            x = _smooth_global(x, 130 if K <= 5 else 110, T0, 0.0016, 0.055, 0.010)
            v = _max_cos(x)
            tol = 0.0015 * (1.0 - frac)
            if v < cur + tol:
                x_cur, cur = x, v
            if v < best:
                x_best, best = x.copy(), v
        elif mode == 3:
            K = 12
            # random cap excision large jump
            rc = rng.normal(size=(4,))
            rc /= np.linalg.norm(rc)
            dots_c = x @ rc
            victims = np.argsort(-dots_c)[:K]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            keep_idx = np.where(keep_mask)[0]
            base = x[keep_idx]
            voids = _find_voids(base, rng, K, n_samp=6000)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            mov = voids + rng.normal(scale=8e-4, size=voids.shape)
            mov /= np.linalg.norm(mov, axis=1, keepdims=True)
            mov = _relax_moving(base, mov, 90, 0.025, 0.0025, 0.10, 0.02)
            x[victims] = mov
            sig = 0.005 * (1.0 - frac) + 0.0015
            x[keep_idx] += rng.normal(scale=sig, size=(len(keep_idx), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth_global(x, 130, 0.024, 0.0018, 0.06, 0.011)
            v = _max_cos(x)
            tol = 0.0020 * (1.0 - frac)
            if v < cur + tol:
                x_cur, cur = x, v
            if v < best:
                x_best, best = x.copy(), v
        elif mode == 4:
            sig = 0.018 * (1.0 - frac) + 0.003
            x = x + rng.normal(scale=sig, size=x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth_global(x, 120, 0.016, 0.0015, 0.05, 0.010)
            v = _max_cos(x)
            tol = 0.0012 * (1.0 - frac)
            if v < cur + tol:
                x_cur, cur = x, v
            if v < best:
                x_best, best = x.copy(), v
        else:  # mode 6: targeted shake of worst neighbourhood
            rm, _ = _row_max(x)
            w0 = int(np.argmax(rm))
            d0 = x @ x[w0]
            nbr = np.argsort(-d0)[:14]
            sig = 0.05 * (1.0 - frac) + 0.008
            x[nbr] = x[nbr] + rng.normal(scale=sig, size=(len(nbr), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth_global(x, 110, 0.020, 0.0016, 0.05, 0.010)
            v = _max_cos(x)
            tol = 0.0015 * (1.0 - frac)
            if v < cur + tol:
                x_cur, cur = x, v
            if v < best:
                x_best, best = x.copy(), v
        if it % 15 == 0 and cur > best + 0.0025:
            x_cur, cur = x_best.copy(), best
        if best < 0.8535:
            break
        if DEAD - (time.time() - t0) < 58:
            break
    remain = DEAD - (time.time() - t0)
    lp_time = float(min(38.0, max(8.0, remain - 5.0)))
    if lp_time > 3 and DEAD - (time.time() - t0) > 7:
        x_best = _lp_polish(x_best, lp_time, r0=2e-3)
        best = _max_cos(x_best)
    if DEAD - (time.time() - t0) > 14:
        for _ in range(3):
            if DEAD - (time.time() - t0) < 11:
                break
            y = x_best + rng.normal(scale=0.0025, size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_global(y, 90, 0.007, 0.0014, 0.028, 0.007)
            v = _max_cos(y)
            if v < best:
                x_best, best = y, v
        r2 = float(min(9.0, DEAD - (time.time() - t0) - 1.0))
        if r2 > 2:
            x_best = _lp_polish(x_best, r2, r0=7e-4)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

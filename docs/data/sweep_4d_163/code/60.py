import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees(x, thr=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    # contact if m - g < thr
    C = (m - G) < thr
    np.fill_diagonal(C, False)
    deg = C.sum(axis=1).astype(int) // 1
    # C is symmetric, sum counts both? row sum counts neighbours
    return deg, m

def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    x = _norm(x)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
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
            if sel.size > cap + 400:
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
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return x

def _local_opt_free(Y, Z, iters=300, T0=0.02, T1=0.0015, lr0=0.08, lr1=0.012):
    Y = _norm(Y)
    k = Y.shape[0]
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        Gfb = Y @ Z.T
        Gff = Y @ Y.T
        # ignore diag
        m = Gfb.max() if Gfb.size else -2
        # max of off-diag Gff
        off = Gff.copy()
        np.fill_diagonal(off, -2.0)
        m2 = float(off.max())
        if m2 > m:
            m = m2
        Wfb = np.exp((Gfb - m) / T)
        Wff = np.exp((off - m) / T)
        np.fill_diagonal(Wff, 0.0)
        grad = Wfb @ Z + Wff @ Y
        rs = (Wfb.sum(axis=1) + Wff.sum(axis=1))[:, None]
        rs = np.maximum(rs, 1e-12)
        dot = np.sum(grad * Y, axis=1, keepdims=True)
        gt = (grad - dot * Y) / rs
        sn = np.linalg.norm(gt, axis=1, keepdims=True)
        gt = gt / np.maximum(sn, 1.0)
        Y = Y - lr * gt
        Y = _norm(Y)
    return Y

def _void_reinit(centre, Z, k, rng, extra_jitter=0.0):
    d = 4
    # threshold from cap: we will sample near centre
    # generate pool: mix of uniform and concentrated
    P = 4000
    # uniform part
    uni = rng.normal(size=(P, d))
    uni = uni / np.linalg.norm(uni, axis=1, keepdims=True)
    # concentrated part around centre
    sigs = np.array([0.25, 0.35, 0.5])
    sig = sigs[rng.integers(0, len(sigs))]
    conc = centre[None, :] + sig * rng.normal(size=(P, d))
    conc = conc / np.linalg.norm(conc, axis=1, keepdims=True)
    pool = np.vstack([uni, conc])
    # score by distance to frozen: max dot (smaller better)
    # also prefer inside cap: dot with centre large
    dots_c = pool @ centre
    # keep only near cap: top 2500 by dots_c
    order_c = np.argsort(-dots_c)
    pool = pool[order_c[:3000]]
    # compute frozen proximity
    # chunked for speed
    # Z is (Nz,4)
    S = pool @ Z.T  # 3000 x Nz
    smax = S.max(axis=1)
    # greedy diverse pick
    picked_idx = []
    picked = []
    # start from best void
    i0 = int(np.argmin(smax))
    picked_idx.append(i0)
    picked.append(pool[i0])
    # precompute pool-pool dots on the fly
    for _ in range(1, k):
        cur = np.stack(picked, axis=0)  # m x 4
        # max dot to already picked
        dpick = pool @ cur.T  # 3000 x m
        dmax_pick = dpick.max(axis=1)
        # combined cost: max(smax, dmax_pick)
        cost = np.maximum(smax, dmax_pick)
        # mask out already picked (set inf)
        for pi in picked_idx:
            cost[pi] = 2.0
        # add tiny random tie-break for diversity
        cost = cost + 1e-6 * rng.random(cost.shape)
        ib = int(np.argmin(cost))
        picked_idx.append(ib)
        picked.append(pool[ib])
    Y0 = np.stack(picked, axis=0)
    if extra_jitter > 0:
        Y0 = Y0 + extra_jitter * rng.normal(size=Y0.shape)
        Y0 = _norm(Y0)
    return Y0

def solve():
    t0 = time.time()
    HARD = 150.0
    FINAL_DEAD = 172.0
    rng = np.random.default_rng(777)
    n, d = 163, 4
    inc = None
    inc_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f)
            c = np.asarray(c, dtype=float)
            if c.shape == (163, 4):
                c = _norm(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
                if np.all(np.isfinite(c)):
                    v = _max_cos(c)
                    if v < inc_val:
                        inc_val = v
                        inc = c.copy()
        except Exception:
            pass
    if inc is None:
        inc = _norm(rng.normal(size=(163, 4)))
        inc_val = _max_cos(inc)
    best = inc.copy()
    best_val = inc_val

    cap_cycle = [10, 12, 14, 11, 13, 10, 14, 12]
    attempt = 0
    # precompute degrees
    deg, _ = _degrees(best, thr=1e-7)
    while time.time() - t0 < HARD:
        attempt += 1
        remain = HARD - (time.time() - t0)
        if remain < 4:
            break
        # rotate centre type: 0 rattler, 1 hub, 2 low, 3 random
        ctype = attempt % 4
        # refresh degrees every few attempts or after improvement
        if attempt % 5 == 1:
            deg, _ = _degrees(best, thr=1e-7)
        rattlers = np.where(deg == 0)[0]
        hubs = np.where(deg >= 8)[0]
        lows = np.where((deg >= 1) & (deg <= 4))[0]
        if ctype == 0 and len(rattlers) > 0:
            c_idx = int(rattlers[rng.integers(0, len(rattlers))])
        elif ctype == 1 and len(hubs) > 0:
            c_idx = int(hubs[rng.integers(0, len(hubs))])
        elif ctype == 2 and len(lows) > 0:
            c_idx = int(lows[rng.integers(0, len(lows))])
        else:
            c_idx = int(rng.integers(0, n))
        k = int(cap_cycle[attempt % len(cap_cycle)])
        # occasionally vary k by +-1
        if rng.random() < 0.25:
            k = int(np.clip(k + int(rng.integers(-1, 2)), 10, 14))
        centre = best[c_idx].copy()
        dots = best @ centre
        free = np.argsort(-dots)[:k]
        frozen_mask = np.ones(n, dtype=bool)
        frozen_mask[free] = False
        frozen_idx = np.where(frozen_mask)[0]
        Z = best[frozen_idx].copy()
        # two diverse restarts, keep better
        best_Y = None
        best_full = 1e9
        best_Xcand = None
        n_restart = 2
        # vary iters to stay cheap
        for r in range(n_restart):
            if time.time() - t0 > HARD - 2:
                break
            Y0 = _void_reinit(centre, Z, k, rng, extra_jitter=0.02 * r)
            # anneal params slightly varied
            T0 = 0.02 if (r == 0) else 0.03
            Yopt = _local_opt_free(Y0, Z, iters=280, T0=T0, T1=0.0015, lr0=0.08, lr1=0.012)
            Xcand = best.copy()
            Xcand[free] = Yopt
            fv = _max_cos(Xcand)
            if fv < best_full:
                best_full = fv
                best_Y = Yopt
                best_Xcand = Xcand
        if best_Xcand is None:
            continue
        # filter: if far worse than best, skip LP (topology too disrupted)
        if best_full > best_val + 0.003:
            continue
        # global LP polish capped ~2.5-3s
        lp_sec = 2.8
        # shrink if little time left
        lp_sec = min(lp_sec, HARD - (time.time() - t0) - 0.5)
        if lp_sec < 0.8:
            continue
        # alternate r0 slightly for diversity
        r0 = 2e-3 if (attempt % 2 == 0) else 1.2e-3
        try:
            Xpol = _lp_polish(best_Xcand, seconds=float(lp_sec), r0=r0, cap=1500)
        except Exception:
            continue
        vpol = _max_cos(Xpol)
        if vpol < best_val - 1e-12:
            best = _norm(Xpol)
            best_val = vpol
            deg, _ = _degrees(best, thr=1e-7)
            try:
                np.save("best.npy", best)
            except Exception:
                pass
        # timeout guard
        if time.time() - t0 > HARD:
            break
    # final short polish with remaining budget (up to FINAL_DEAD)
    if FINAL_DEAD - (time.time() - t0) > 4:
        r = float(min(8.0, FINAL_DEAD - (time.time() - t0) - 1.0))
        if r > 1.5:
            try:
                best = _lp_polish(best, seconds=r, r0=8e-4, cap=1500)
                best_val = _max_cos(best)
            except Exception:
                pass
    if _max_cos(best) > inc_val:
        best = inc
    return _norm(best)

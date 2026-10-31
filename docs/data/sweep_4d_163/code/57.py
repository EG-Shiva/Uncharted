import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=2000):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 600:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
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

def _refine_voids_frozen(frozen, pool):
    out = np.empty_like(pool)
    for i, v0 in enumerate(pool):
        v = v0.copy()
        for _ in range(25):
            d = frozen @ v
            dm = float(d.max())
            w = np.exp(50.0 * (d - dm))
            g = w @ frozen
            gt = g - np.dot(g, v) * v
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-12:
                break
            v = v - 0.06 * gt / nrm
            n = float(np.linalg.norm(v))
            if n < 1e-12:
                break
            v /= n
        out[i] = v
    return out

def _voids_in_cap(frozen, centre, boundary, rng, k):
    # sample candidates near cap, score vs frozen, refine, diverse pick
    S = 5000
    cand = rng.normal(size=(S, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dc = cand @ centre
    keep = dc >= (boundary - 0.12)
    if int(np.count_nonzero(keep)) < k * 8:
        # relax margin
        keep = dc >= (boundary - 0.30)
    if int(np.count_nonzero(keep)) < k * 8:
        keep = np.ones(S, dtype=bool)
    cand = cand[keep]
    dc = dc[keep]
    sc = (cand @ frozen.T).max(axis=1)
    order = np.argsort(sc)[:k * 8]
    pool = cand[order]
    pool = _refine_voids_frozen(frozen, pool)
    sc2 = (pool @ frozen.T).max(axis=1)
    o2 = np.argsort(sc2)
    pool = pool[o2]
    sc2 = sc2[o2]
    sel = []
    rem = list(range(len(pool)))
    if rem:
        sel.append(rem.pop(0))
        while len(sel) < k and rem:
            Smat = pool[sel] @ pool[rem].T
            mm = Smat.max(axis=0)
            crit = sc2[rem] + 0.25 * mm
            j = int(np.argmin(crit))
            sel.append(rem.pop(j))
    return pool[np.array(sel)]

def _smooth_cap(X, free_idx, frozen_idx, iters, T0, T1, lr0, lr1):
    Xf = X[free_idx].copy()
    Xr = X[frozen_idx].copy()
    k = Xf.shape[0]
    for t in range(iters):
        frac = t / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        D1 = Xf @ Xr.T
        D2 = Xf @ Xf.T
        m1 = float(D1.max()) if D1.size else -2.0
        # off-diagonal max of D2
        tmp = D2.copy()
        np.fill_diagonal(tmp, -2.0)
        m2 = float(tmp.max()) if k > 1 else -2.0
        m = m1 if m1 > m2 else m2
        W1 = np.exp((D1 - m) / T)
        W2 = np.exp((tmp - m) / T)
        np.fill_diagonal(W2, 0.0)
        grad = W1 @ Xr + W2 @ Xf
        rs = W1.sum(axis=1, keepdims=True) + W2.sum(axis=1, keepdims=True)
        dot = np.sum(grad * Xf, axis=1, keepdims=True)
        gt = grad - dot * Xf
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        Xf = Xf - lr * step
        Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
    X = X.copy()
    X[free_idx] = Xf
    return X

def solve():
    t0 = time.time()
    DEAD = 172.0
    INTERNAL = 150.0
    rng = np.random.default_rng(0)
    # load starts
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=float))
    except Exception:
        pass
    if len(cands) == 0:
        x0 = rng.normal(size=(163, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        vals = []
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            vals.append(_max_cos(c))
        x0 = cands[int(np.argmin(vals))].copy()
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    n = 163
    x_best = x0.copy()
    best = _max_cos(x_best)
    # precompute row max for centre selection
    rowmax = _row_max(x_best)
    order_rm = np.argsort(rowmax)
    cap_sizes = [10, 12, 14, 11, 13]
    centre_cycle = ["rattler", "hub", "low", "random", "hub", "low", "random", "rattler"]
    it = 0
    T0s = [0.025, 0.015, 0.035, 0.02]
    while time.time() - t0 < INTERNAL and time.time() - t0 < DEAD - 25:
        it += 1
        ctype = centre_cycle[(it - 1) % len(centre_cycle)]
        if ctype == "rattler":
            centre_idx = int(order_rm[0])
        elif ctype == "hub":
            centre_idx = int(order_rm[-1])
            # occasionally second hub for diversity
            if (it // len(centre_cycle)) % 3 == 1:
                centre_idx = int(order_rm[-2])
            elif (it // len(centre_cycle)) % 3 == 2:
                centre_idx = int(order_rm[-3])
        elif ctype == "low":
            # random among 10 lowest
            bot10 = order_rm[:10]
            centre_idx = int(rng.choice(bot10))
        else:
            centre_idx = int(rng.integers(0, n))
        k = int(cap_sizes[(it - 1) % len(cap_sizes)])
        centre = x_best[centre_idx].copy()
        dots = x_best @ centre
        free_idx = np.argsort(-dots)[:k]
        # ensure centre included (it will be top)
        frozen_mask = np.ones(n, dtype=bool)
        frozen_mask[free_idx] = False
        frozen_idx = np.where(frozen_mask)[0]
        frozen = x_best[frozen_idx]
        # frozen-frozen max (constant for this cap)
        # boundary for cap
        boundary = float(dots[free_idx].min())
        try:
            voids = _voids_in_cap(frozen, centre, boundary, rng, k)
        except Exception:
            continue
        # tiny jitter to break symmetry
        voids = voids + rng.normal(scale=5e-4, size=voids.shape)
        voids /= np.linalg.norm(voids, axis=1, keepdims=True)
        Xt = x_best.copy()
        Xt[free_idx] = voids
        T0 = float(T0s[(it - 1) % len(T0s)])
        T1 = 0.0012
        lr0, lr1 = 0.08, 0.01
        # a few hundred inexpensive iterations
        nit = 300
        # adapt iters to time: if late, fewer
        remain = DEAD - (time.time() - t0)
        if remain < 40:
            nit = 220
        try:
            Xt = _smooth_cap(Xt, free_idx, frozen_idx, nit, T0, T1, lr0, lr1)
        except Exception:
            continue
        # evaluate complete config: free-involved max vs frozen max
        # quick full max
        v_full = _max_cos(Xt)
        # free-involved max for gating
        Xf = Xt[free_idx]
        D1m = float((Xf @ frozen.T).max())
        tmp2 = Xf @ Xf.T
        np.fill_diagonal(tmp2, -2.0)
        D2m = float(tmp2.max())
        free_max = D1m if D1m > D2m else D2m
        # gate: need slack in free region to allow LP to help
        # frozen max is <= best; if free_max too high skip LP
        if free_max > best + 0.0015 and v_full > best + 0.0005:
            continue
        # global LP polish capped ~3s (use 2.6s to stay in budget)
        lp_sec = 2.8
        if DEAD - (time.time() - t0) < lp_sec + 12:
            lp_sec = 1.5
        try:
            Xp = _lp_polish(Xt, lp_sec, r0=2e-3, cap=2000)
        except Exception:
            continue
        v = _max_cos(Xp)
        if v < best - 1e-12:
            x_best = Xp / np.linalg.norm(Xp, axis=1, keepdims=True)
            best = v
            rowmax = _row_max(x_best)
            order_rm = np.argsort(rowmax)
            try:
                np.save("best.npy", x_best)
            except Exception:
                pass
            if best < 0.8545:
                break
        # hard internal deadline check
        if time.time() - t0 > INTERNAL:
            break
    # final global polish with remaining time
    remain = DEAD - (time.time() - t0) - 1.0
    if remain > 3:
        lp_time = float(min(18.0, remain))
        try:
            x_best = _lp_polish(x_best, lp_time, r0=1.5e-3, cap=2000)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

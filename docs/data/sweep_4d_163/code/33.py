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
    return g.max(axis=1), g

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # avoid overflow: (G-m)<=0
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

def _refine_voids(base, pool):
    out = np.empty_like(pool)
    for i, v0 in enumerate(pool):
        v = v0.copy()
        for _ in range(30):
            d = base @ v
            dm = float(d.max())
            w = np.exp(50.0 * (d - dm))
            g = w @ base
            gt = g - np.dot(g, v) * v
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-12:
                break
            v = v - 0.06 * gt / nrm
            v /= np.linalg.norm(v)
        out[i] = v
    return out

def _find_voids(base, rng, k, n_samp=6000):
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ base.T).max(axis=1)
    order = np.argsort(sc)[:k * 8]
    pool = cand[order]
    pool = _refine_voids(base, pool)
    sc2 = (pool @ base.T).max(axis=1)
    o2 = np.argsort(sc2)
    pool = pool[o2]
    sc2 = sc2[o2]
    # greedy diverse pick
    sel = []
    rem = list(range(len(pool)))
    if rem:
        sel.append(rem.pop(0))
        while len(sel) < k and rem:
            S = pool[sel] @ pool[rem].T
            mm = S.max(axis=0)
            crit = sc2[rem] + 0.25 * mm
            j = int(np.argmin(crit))
            sel.append(rem.pop(j))
    return pool[np.array(sel)]

def _lp_polish(x, seconds, r0=2e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
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
        I = iu0[0][sel]; J = iu0[1][sel]
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
    DEAD = 172.0
    rng = np.random.default_rng(7)
    # load start
    try:
        b = np.load("best.npy")
    except Exception:
        b = None
    try:
        r = np.load("record.npy")
    except Exception:
        r = None
    cands = []
    if b is not None:
        cands.append(np.asarray(b, dtype=float))
    if r is not None:
        cands.append(np.asarray(r, dtype=float))
    if len(cands) == 0:
        x0 = rng.normal(size=(163, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        vals = []
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            vals.append(_max_cos(c))
        x0 = cands[int(np.argmin(vals))].copy()
        x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    n = 163
    x_best = x0.copy()
    best = _max_cos(x_best)
    x_cur = x_best.copy()
    cur = best
    # short initial smooth to ensure local clean
    # hopping budget ~100s
    HOP_BUDGET = 105.0
    it = 0
    # temperature for sideways acceptance
    while time.time() - t0 < HOP_BUDGET and time.time() - t0 < DEAD - 55:
        it += 1
        frac = min(1.0, (time.time() - t0) / HOP_BUDGET)
        # cycle move types
        mode = it % 4
        x = x_cur.copy()
        rm, Gfull = _row_max(x)
        worst_order = np.argsort(rm)  # ascending
        if mode == 0 or mode == 1:
            nk = 2 if mode == 0 else 3
            # victims: pick among 8 worst, random subset
            pool_w = worst_order[-8:]
            # always include the very worst
            victims = [int(worst_order[-1])]
            if nk > 1:
                others = [i for i in pool_w if i not in victims]
                ch = rng.choice(others, size=min(nk - 1, len(others)), replace=False)
                victims += [int(c) for c in ch]
            victims = np.array(victims, dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            keep_idx = np.where(keep_mask)[0]
            base = x[keep_idx]
            voids = _find_voids(base, rng, len(victims), n_samp=5000)
            # tiny jitter so symmetric voids break
            voids = voids + rng.normal(scale=1e-3, size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            x[victims] = voids
            # small shake to keep set
            sig = 0.008 * (1.0 - frac) + 0.002
            x[keep_idx] = x[keep_idx] + rng.normal(scale=sig, size=(len(keep_idx), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 2:
            sig = 0.018 * (1.0 - frac) + 0.004
            x = x + rng.normal(scale=sig, size=x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # cap shake around worst point
            w0 = int(worst_order[-1])
            d0 = (x @ x[w0])
            nbr = np.argsort(-d0)[:14]
            sig = 0.05 * (1.0 - frac) + 0.01
            x[nbr] = x[nbr] + rng.normal(scale=sig, size=(len(nbr), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # relax: annealed softmax
        # vary schedule slightly per iteration for diversity
        T0 = 0.03 if (it % 2 == 0) else 0.015
        T1 = 0.002
        lr0, lr1 = 0.07, 0.015
        x = _smooth(x, 220, T0, T1, lr0, lr1)
        v = _max_cos(x)
        # acceptance: greedy + small sideways early
        tol = 0.0015 * (1.0 - frac)
        if v < cur + tol:
            x_cur, cur = x, v
        if v < best:
            x_best, best = x.copy(), v
        # occasional restart from best if drifting
        if it % 12 == 0 and cur > best + 0.002:
            x_cur, cur = x_best.copy(), best
        if best < 0.8545:
            break
    # final polish of best with remaining time
    remain = DEAD - (time.time() - t0)
    lp_time = float(min(55.0, max(8.0, remain - 2.0)))
    if lp_time > 3:
        x_best = _lp_polish(x_best, lp_time)
        best = _max_cos(x_best)
    # quick final micro-shake + smooth if time left (basin edge)
    if DEAD - (time.time() - t0) > 12:
        for _ in range(3):
            y = x_best + rng.normal(scale=0.003, size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, 120, 0.008, 0.0015, 0.03, 0.008)
            v = _max_cos(y)
            if v < best:
                x_best, best = y, v
            if DEAD - (time.time() - t0) < 10:
                break
        # re-polish briefly
        r2 = float(min(10.0, DEAD - (time.time() - t0) - 1.0))
        if r2 > 2:
            x_best = _lp_polish(x_best, r2, r0=8e-4)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

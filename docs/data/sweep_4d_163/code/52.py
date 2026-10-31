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
    out = pool.copy()
    for i in range(len(pool)):
        v = out[i].copy()
        for _ in range(25):
            d = base @ v
            dm = float(d.max())
            w = np.exp(70.0 * (d - dm))
            s = float(w.sum())
            if s <= 0:
                break
            g = (w @ base) / s
            gt = g - np.dot(g, v) * v
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-12:
                break
            v = v - 0.08 * gt / nrm
            v /= np.linalg.norm(v)
        out[i] = v
    return out

def _find_voids(base, rng, k, n_samp=6000):
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ base.T).max(axis=1)
    take = min(n_samp, max(k * 12, 30))
    order = np.argpartition(sc, take - 1)[:take]
    pool = cand[order]
    pool = _refine_voids(base, pool)
    sc2 = (pool @ base.T).max(axis=1)
    o2 = np.argsort(sc2)
    pool = pool[o2]; sc2 = sc2[o2]
    if k >= len(pool):
        return pool
    sel = [0]
    while len(sel) < k:
        rem = [i for i in range(len(pool)) if i not in sel]
        S = pool[sel] @ pool[rem].T
        mm = S.max(axis=0)
        crit = sc2[rem] + 0.35 * mm
        sel.append(rem[int(np.argmin(crit))])
    return pool[np.array(sel)]

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=1600):
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
        if int(np.count_nonzero(sel)) == 0:
            break
        if int(np.count_nonzero(sel)) > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            if int(np.count_nonzero(sel)) > cap + 400:
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
            radius = min(radius * 1.6, 0.04)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = 170.0
    rng = np.random.default_rng(777)
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
    n = 163
    if len(cands) == 0:
        x0 = rng.normal(size=(n, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        vals = []
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            vals.append(_max_cos(c))
        x0 = cands[int(np.argmin(vals))].copy()
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    x_best = x0.copy()
    best = _max_cos(x_best)
    x_cur = x_best.copy()
    cur = best
    HOP = 125.0
    it = 0
    while time.time() - t0 < HOP and time.time() - t0 < DEAD - 45:
        it += 1
        frac = min(1.0, (time.time() - t0) / max(HOP, 1e-9))
        x = x_cur.copy()
        rm, Gf = _row_max(x)
        worst_order = np.argsort(rm)
        mnow = float(rm.max())
        deg = (Gf > mnow - 5e-4).sum(axis=1)
        # hubs: high degree; loose: low degree
        hub_order = np.argsort(deg)
        mode = it % 6
        if mode in (0, 1):
            nk = 3 if mode == 0 else 4
            victims = [int(worst_order[-1])]
            # add top hubs
            for h in hub_order[::-1]:
                if len(victims) >= nk:
                    break
                if int(h) not in victims:
                    victims.append(int(h))
            # add one random from 12 worst if still room
            pw = list(worst_order[-12:])
            rng.shuffle(pw)
            for c in pw:
                if len(victims) >= nk:
                    break
                if int(c) not in victims:
                    victims.append(int(c))
            victims = np.array(victims[:nk], dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            keep_idx = np.where(keep_mask)[0]
            base = x[keep_idx]
            voids = _find_voids(base, rng, len(victims), n_samp=6000)
            voids = voids + rng.normal(scale=7e-4, size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            x[victims] = voids
            sig = 0.008 * (1.0 - frac) + 0.0015
            x[keep_idx] += rng.normal(scale=sig, size=(len(keep_idx), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            nit = 200
            T0 = 0.024 if (it % 2 == 0) else 0.012
        elif mode == 2:
            # patch kill around worst: nearest 5
            w0 = int(worst_order[-1])
            nbr = np.argsort(-(x @ x[w0]))[:5]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[nbr] = False
            keep_idx = np.where(keep_mask)[0]
            base = x[keep_idx]
            voids = _find_voids(base, rng, len(nbr), n_samp=6000)
            voids += rng.normal(scale=9e-4, size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            x[nbr] = voids
            sig = 0.006 * (1.0 - frac) + 0.0015
            x[keep_idx] += rng.normal(scale=sig, size=(len(keep_idx), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            nit = 200
            T0 = 0.020
        elif mode == 3:
            sig = 0.018 * (1.0 - frac) + 0.003
            x = x + rng.normal(scale=sig, size=x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            nit = 170
            T0 = 0.018
        elif mode == 4:
            # disperse a random hub neighbourhood
            h = int(hub_order[-1 - (it % 5)])
            nbr = np.argsort(-(x @ x[h]))[:9]
            # only jitter, keep topology partly
            sig = 0.045 * (1.0 - frac) + 0.008
            x[nbr] += rng.normal(scale=sig, size=(len(nbr), 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            nit = 170
            T0 = 0.016
        else:
            # global small shake
            sig = 0.010 * (1.0 - frac) + 0.002
            x = x + rng.normal(scale=sig, size=x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            nit = 150
            T0 = 0.010
        T1 = 0.0015
        x = _smooth(x, nit, T0, T1, 0.060, 0.010)
        v = _max_cos(x)
        tol = 0.0015 * (1.0 - frac)
        if v < cur + tol:
            x_cur, cur = x, v
        if v < best:
            x_best, best = x.copy(), v
        if it % 15 == 0 and cur > best + 0.002:
            x_cur, cur = x_best.copy(), best
        if best < 0.8540:
            break
        if DEAD - (time.time() - t0) < 48:
            break
    remain = DEAD - (time.time() - t0)
    lp_time = float(min(32.0, max(8.0, remain - 3.0)))
    if lp_time > 3 and remain > 5:
        x_best = _lp_polish(x_best, lp_time, r0=2e-3)
        best = _max_cos(x_best)
    # quick shake screen without LP if time left
    if DEAD - (time.time() - t0) > 12:
        for _ in range(3):
            if DEAD - (time.time() - t0) < 10:
                break
            y = x_best + rng.normal(scale=0.0022, size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, 100, 0.006, 0.0012, 0.025, 0.006)
            v = _max_cos(y)
            if v < best:
                x_best, best = y, v
        r2 = float(min(8.0, DEAD - (time.time() - t0) - 1.0))
        if r2 > 2:
            x_best = _lp_polish(x_best, r2, r0=6e-4)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

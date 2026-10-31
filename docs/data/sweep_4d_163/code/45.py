import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

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
    for _ in range(20):
        d = out @ base.T
        dm = d.max(axis=1, keepdims=True)
        w = np.exp(60.0 * (d - dm))
        g = w @ base
        dot = np.sum(g * out, axis=1, keepdims=True)
        gt = g - dot * out
        nrm = np.linalg.norm(gt, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        out = out - 0.06 * gt / nrm
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out

def _find_voids(base, rng, k, n_samp=4000):
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ base.T).max(axis=1)
    order = np.argsort(sc)[:max(k * 8, k + 10)]
    pool = cand[order]
    pool = _refine_voids(base, pool)
    sc2 = (pool @ base.T).max(axis=1)
    o2 = np.argsort(sc2)
    pool = pool[o2]
    sc2 = sc2[o2]
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
    HARD = 170.0
    rng = np.random.default_rng(7)
    n, d = 163, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (163, 4):
                a = np.nan_to_num(a)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        z = rng.normal(size=(n, d)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None; best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    # quick anneal of starting point to squeeze without LP
    best = _smooth(best.copy(), 120, 0.008, 0.0008, 0.06, 0.02)
    best_val = _max_cos(best)
    # hopping budget: leave ~28s for final LP
    trial = 0
    while time.time() - t0 < HARD - 30.0:
        trial += 1
        mode = trial % 4
        y = best.copy()
        if mode == 0 or mode == 3:
            # kill worst k and reinsert into voids
            k = int(rng.integers(3, 7))
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            worst = np.argpartition(-rm, k)[:k]
            mask = np.ones(n, dtype=bool); mask[worst] = False
            base = y[mask]
            try:
                voids = _find_voids(base, rng, k, n_samp=4000)
            except Exception:
                voids = rng.normal(size=(k, d))
                voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            y[worst] = voids
            y = _smooth(y, 260, 0.012, 0.0006, 0.09, 0.015)
        elif mode == 1:
            # global shake
            sig = float(rng.choice([0.02, 0.04, 0.07, 0.11]))
            y = y + sig * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, 220, 0.010, 0.0007, 0.08, 0.015)
        else:
            # patch shake around random center + neighbours
            c0 = int(rng.integers(0, n))
            dots = y @ y[c0]
            nn = np.argpartition(-dots, 13)[:13]
            sig = float(rng.choice([0.05, 0.10, 0.16]))
            y[nn] = y[nn] + sig * rng.normal(size=(len(nn), d))
            y[nn] /= np.linalg.norm(y[nn], axis=1, keepdims=True)
            # small global jitter
            y = y + 0.008 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, 240, 0.011, 0.0006, 0.085, 0.015)
        v = _max_cos(y)
        if v < best_val:
            best = y; best_val = v
        # early exit if clearly record beating by margin
        if time.time() - t0 > HARD - 32.0:
            break
    # final polish with remaining time
    rem = HARD - (time.time() - t0) - 1.0
    rem = float(np.clip(rem, 5.0, 30.0))
    try:
        yp = _lp_polish(best, seconds=rem, r0=2e-3, cap=2000)
        vp = _max_cos(yp)
        if vp < best_val:
            best = yp; best_val = vp
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

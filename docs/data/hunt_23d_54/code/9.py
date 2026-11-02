import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t0=None, budget=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 600:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:600]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x_init, steps, k0, k1, lr0, lr1, d0, d1):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        delta = d0 + (d1 - d0) * frac
        G = x @ x.T
        m = float(G[iu].max())
        thr = m - delta
        mask = G > thr
        np.fill_diagonal(mask, False)
        E = np.zeros_like(G)
        # stable exp
        E[mask] = np.exp(k * (G[mask] - m))
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        x = x - lr * Gt / deg
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _greedy_fill(Xf, k, rng, ncand=3000):
    d = Xf.shape[1]
    Y = np.zeros((k, d))
    for a in range(k):
        cand = rng.randn(ncand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        md = cand @ Xf.T
        m1 = md.max(axis=1) if md.shape[1] > 0 else np.full(ncand, -10.0)
        if a > 0:
            m2 = (cand @ Y[:a].T).max(axis=1)
            m = np.maximum(m1, m2)
        else:
            m = m1
        Y[a] = cand[np.argmin(m)]
    return Y

def _refine_subset(Xf, Y, steps, k0, k1, lr0, lr1, rng=None):
    # optimize Y with Xf fixed
    k = Y.shape[0]
    nf = Xf.shape[0]
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        Df = Y @ Xf.T  # k x nf
        Ds = Y @ Y.T   # k x k
        # current max over all involved pairs
        m = float(Df.max()) if Df.size else -2.0
        if k > 1:
            tmp = Ds.copy()
            np.fill_diagonal(tmp, -2.0)
            m = max(m, float(tmp.max()))
        Wf = np.exp(kk * (Df - m))  # k x nf
        Ws = np.exp(kk * (Ds - m))
        np.fill_diagonal(Ws, 0.0)
        Gf = Wf @ Xf  # k x d
        Gs = Ws @ Y
        G = Gf + Gs
        dot = np.sum(G * Y, axis=1, keepdims=True)
        Gt = G - dot * Y
        deg = Wf.sum(axis=1, keepdims=True) + Ws.sum(axis=1, keepdims=True) + 1.0
        Y = Y - lr * Gt / deg
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(11)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(54, 23)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 54, 23
    if xb.shape != (54, 23):
        tmp = rng.randn(54, 23)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    best = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
        if vr < best:
            best, x_best = vr, xr.copy()
    except Exception:
        pass
    cur = x_best.copy()
    cur_v = best
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < BUDGET - 11.0:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(BUDGET - 11.0, 1.0)
        Temp = 0.004 * (1.0 - frac) + 0.0002
        remaining = BUDGET - 11.0 - elapsed
        if remaining < 2.0:
            break
        r = rng.rand()
        # contact analysis on cur
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        mc = float(Gc.max())
        cnt = (Gc > mc - 0.01).sum(axis=1)
        order_hi = np.argsort(-cnt)
        order_lo = np.argsort(cnt)
        if r < 0.28:
            # global shake of current (not best) to walk basins
            sigma = float(rng.choice([0.03, 0.06, 0.10, 0.14]))
            x = cur + rng.randn(n, d) * sigma
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 350 if remaining > 30 else 220
            xs = _smooth(x, steps, 25.0, 450.0, 0.30, 0.025, 0.10, 0.008)
        else:
            # hole resample: choose k and victim set
            u = rng.rand()
            if u < 0.30:
                k = int(rng.choice([3, 4, 4, 5]))
            elif u < 0.65:
                k = int(rng.choice([2, 3, 3, 4]))
            else:
                k = int(rng.choice([2, 5, 6]))
            k = min(k, 7)
            mode = rng.rand()
            if mode < 0.40:
                # jammed cluster: seed among most loaded + its top neighbours
                pool = order_hi[:8]
                seed = int(pool[rng.randint(0, len(pool))])
                neigh = np.argsort(-Gc[seed])[:k-1]
                idx = np.concatenate([[seed], neigh]).astype(int)
                # dedup
                idx = np.unique(idx)
                while len(idx) < k:
                    extra = int(rng.randint(0, n))
                    if extra not in idx:
                        idx = np.append(idx, extra)
                idx = idx[:k]
            elif mode < 0.60:
                # loose points
                pool = order_lo[:10]
                sel = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
                idx = pool[sel]
                if len(idx) < k:
                    extra = rng.choice(n, size=k-len(idx), replace=False)
                    idx = np.concatenate([idx, extra])
            elif mode < 0.80:
                # random spatial cluster
                seed = int(rng.randint(0, n))
                neigh = np.argsort(-Gc[seed])[:k-1]
                idx = np.concatenate([[seed], neigh]).astype(int)
                idx = np.unique(idx)
                while len(idx) < k:
                    extra = int(rng.randint(0, n))
                    if extra not in idx:
                        idx = np.append(idx, extra)
                idx = idx[:k]
            else:
                idx = rng.choice(n, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xf = cur[mask]
            ncand = 3000 if remaining > 25 else 1500
            Y0 = _greedy_fill(Xf, k, rng, ncand=ncand)
            # subset repair
            Y1 = _refine_subset(Xf, Y0, steps=350, k0=20.0, k1=350.0, lr0=0.35, lr1=0.05)
            x = cur.copy()
            x[idx] = Y1
            # tiny global jitter to break symmetry
            x += rng.randn(n, d) * 0.002
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps2 = 300 if remaining > 30 else 180
            xs = _smooth(x, steps2, 25.0, 400.0, 0.30, 0.025, 0.09, 0.007)
        vs = _max_cos(xs)
        # occasional LP polish if promising
        do_lp = False
        if vs < best + 0.006 and (it % 4 == 0 or vs < best or vs < cur_v):
            # only if enough time
            if remaining > 8:
                do_lp = True
        if do_lp:
            lp_sec = 1.2 if remaining > 25 else 0.7
            if vs < best - 1e-9:
                lp_sec = min(2.0, lp_sec + 0.6)
            try:
                xp = _lp_polish(xs, lp_sec, t0, BUDGET - 10.0)
                vp = _max_cos(xp)
                if vp < vs:
                    xs, vs = xp, vp
            except Exception:
                pass
        # accept / track
        if vs < best - 1e-12:
            best = vs
            x_best = xs.copy()
            cur, cur_v = xs.copy(), vs
        else:
            # annealed walk on cur
            if vs < cur_v or rng.rand() < np.exp((cur_v - vs) / max(Temp, 1e-6)):
                cur, cur_v = xs.copy(), vs
            # occasional restart cur to best to intensify
            if it % 25 == 0 and rng.rand() < 0.5:
                cur, cur_v = x_best.copy(), best
    # final long polish of champion
    try:
        left = BUDGET - (time.time() - t0)
        if left > 2:
            x_best = _lp_polish(x_best, min(left - 0.5, 10.0), t0, BUDGET)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

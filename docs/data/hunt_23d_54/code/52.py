import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.8, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    iu = np.triu_indices(n, 1)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        # current max (off-diagonal)
        m = float(G[iu].max())
        D = G - m
        W = np.zeros_like(G)
        sel = D > -w
        np.fill_diagonal(sel, False)
        if np.any(sel):
            Ds = D[sel]
            # k*Ds in [-k*w,0]
            E = np.exp(np.clip(k * Ds, -20.0, 0.0))
            W[sel] = E
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _void_fill(x, idx, rng, n_try=2000):
    n, d = x.shape
    y = x.copy()
    for j in idx:
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        kept = y[mask]
        C = rng.randn(n_try, d)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ kept.T
        m = dots.max(axis=1)
        o = int(np.argmin(m))
        y[j] = C[o]
    return y

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
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
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
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(20261102)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()
    iu = np.triu_indices(n, 1)

    # optional: one short anneal of loaded best to ensure tightness without LP
    # do not waste time; keep best of the two
    try:
        xc = _anneal(x_best.copy(), steps=250, k0=200, k1=3000, lr0=0.004, lr1=0.0005, w0=0.02, w1=0.005, mu=0.8)
        vc = _max_cos(xc)
        if vc < best:
            best, x_best = vc, xc
    except Exception:
        pass

    it = 0
    # LP-free hopping; reserve ~20s for final LP
    while time.time() - t0 < BUDGET - 22.0:
        it += 1
        rem = BUDGET - 22.0 - (time.time() - t0)
        if rem < 1.0:
            break
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        cnt = np.sum(G > best - 0.015, axis=1)
        ppm = np.max(G, axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-ppm)
        # PCA outliers: project onto top-2 PCs
        try:
            Xc = x_best - x_best.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Xc @ Vt[:2].T
            rad = np.linalg.norm(P, axis=1)
            order_out = np.argsort(-rad)
        except Exception:
            order_out = order_worst
        mode = it % 8
        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.010
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=380, k0=90, k1=2600, lr0=0.012, lr1=0.0008, w0=0.05, w1=0.005, mu=0.8)
        elif mode == 1:
            x += rng.randn(n, d) * 0.028
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=420, k0=60, k1=2000, lr0=0.015, lr1=0.001, w0=0.065, w1=0.006, mu=0.85)
        elif mode == 2:
            ksel = 3 + (it % 3)
            idx = order_lo[:ksel]
            x = _void_fill(x, idx, rng, n_try=1800)
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=450, k0=70, k1=2600, lr0=0.014, lr1=0.0008, w0=0.06, w1=0.005, mu=0.82)
        elif mode == 3:
            ksel = 2 + (it % 2)
            idx = order_worst[:ksel]
            # break worst pairs: replace one side of top pairs
            flat = (x_best @ x_best.T)[iu]
            top = np.argsort(flat)[::-1][:6]
            victims = set(idx.tolist())
            for tt in top:
                i, j = int(iu[0][tt]), int(iu[1][tt])
                victims.add(i if rng.rand() < 0.5 else j)
                if len(victims) >= 4:
                    break
            idx2 = np.array(sorted(victims)[:5], dtype=int)
            x = _void_fill(x, idx2, rng, n_try=1800)
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=450, k0=60, k1=2200, lr0=0.014, lr1=0.0009, w0=0.06, w1=0.005, mu=0.85)
        elif mode == 4:
            ksel = 3
            idx = order_out[:ksel]
            x = _void_fill(x, idx, rng, n_try=2000)
            x += rng.randn(n, d) * 0.005
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=400, k0=80, k1=2400, lr0=0.013, lr1=0.0008, w0=0.055, w1=0.005, mu=0.8)
        elif mode == 5:
            # medium global kick + tight anneal
            x += rng.randn(n, d) * 0.055
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=380, k0=120, k1=3000, lr0=0.010, lr1=0.0007, w0=0.045, w1=0.004, mu=0.8)
        elif mode == 6:
            # random-subset void refill
            ksel = 4
            idx = rng.choice(n, size=ksel, replace=False)
            x = _void_fill(x, idx, rng, n_try=1500)
            x += rng.randn(n, d) * 0.006
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=420, k0=70, k1=2300, lr0=0.014, lr1=0.0009, w0=0.06, w1=0.006, mu=0.85)
        else:
            # fresh random start, short anneal to probe new basin
            y = rng.randn(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # seed with a few voids from best to keep some structure half the time
            if rng.rand() < 0.5:
                keep = rng.choice(n, size=n // 2, replace=False)
                y[keep] = x_best[keep] + rng.randn(len(keep), d) * 0.02
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            x = _anneal(y, steps=500, k0=50, k1=1800, lr0=0.016, lr1=0.001, w0=0.07, w1=0.007, mu=0.85)
        v = _max_cos(x)
        if v < best - 1e-12:
            x_best, best = x, v
        # opportunistic small random polish around best even without improvement:
        # keep hopping; no LP inside loop to save time

    # final LP polish of best (time-limited)
    rem = BUDGET - (time.time() - t0)
    sec = min(max(rem - 1.0, 1.0), 20.0)
    if sec > 0.5:
        try:
            y = _lp_polish(x_best, sec, t0, BUDGET - 1.0)
            vy = _max_cos(y)
            if vy < best:
                x_best, best = y, vy
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

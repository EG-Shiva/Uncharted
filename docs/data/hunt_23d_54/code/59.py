import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _anneal_mask(x_init, steps, k0, k1, lr0, lr1, mask=None):
    x = x_init
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    if mask is None:
        msk = None
    else:
        msk = mask.astype(bool)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 else k1
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 else lr1
        G = x @ x.T
        m = float(np.max(G[iu]))
        A = k * (G - m)
        # threshold
        # in-place clip for speed
        np.clip(A, -12.0, 0.0, out=A)
        E = np.exp(A)
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.einsum('ij,ij->i', Gr, x)[:, None]
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        step = Gt / deg
        if msk is not None:
            step[~msk] = 0.0
        x = x - lr * step
        # renormalize (cheap, keeps frozen unit too)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _void_fill(x, idx_list, rng, n_cand=2000, noise=0.0):
    n, d = x.shape
    for j in idx_list:
        kept_mask = np.ones(n, dtype=bool)
        kept_mask[j] = False
        kept = x[kept_mask]
        cand = rng.randn(n_cand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ kept.T
        worst = dots.max(axis=1)
        b = int(np.argmin(worst))
        v = cand[b]
        if noise > 0:
            v = v + rng.randn(d) * noise
            v /= np.linalg.norm(v)
        x[j] = v
    return x

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
        cnt = int(np.count_nonzero(sel))
        if cnt > 600:
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
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(2)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            vr = _max_cos(xr)
            if vr < best:
                best, x_best = vr, xr.copy()
    except Exception:
        pass
    x_cur = x_best.copy()
    cur = best
    Temp = 0.0025
    it = 0
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        remain = BUDGET - 12.0 - (time.time() - t0)
        if remain < 1.5:
            break
        # quick contact stats on cur
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        ppm = G.max(axis=1)
        cnt = np.sum(G > cur - 0.02, axis=1)
        order_worst = np.argsort(-ppm)
        order_high = np.argsort(-cnt)
        order_low = np.argsort(cnt)
        mode = it % 6
        x = x_cur.copy()
        mask = None
        s1 = 220
        s2 = 220
        if remain < 12:
            s1, s2 = 120, 120
        if mode == 0:
            # 1 worst hole + tiny noise, global anneal
            j = [int(order_worst[0])]
            x = _void_fill(x, j, rng, n_cand=2000, noise=0.0)
            x += rng.randn(n, d) * 0.006
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal_mask(x, steps=s1 + s2, k0=150.0, k1=2000.0, lr0=0.02, lr1=0.002, mask=None)
        elif mode == 1:
            # 2 worst holes
            j = [int(order_worst[0]), int(order_worst[1])]
            x = _void_fill(x, j, rng, n_cand=1600, noise=0.01)
            x += rng.randn(n, d) * 0.005
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal_mask(x, steps=s1 + s2, k0=120.0, k1=2200.0, lr0=0.02, lr1=0.002, mask=None)
        elif mode == 2:
            # 6 high-contact cluster rewiring with frozen background
            k = 6
            j = [int(v) for v in order_high[:k]]
            x = _void_fill(x, j, rng, n_cand=1500, noise=0.02)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            msk = np.zeros(n, dtype=bool); msk[j] = True
            x = _anneal_mask(x, steps=s1, k0=80.0, k1=1200.0, lr0=0.03, lr1=0.004, mask=msk)
            x = _anneal_mask(x, steps=s2, k0=150.0, k1=2200.0, lr0=0.015, lr1=0.0015, mask=None)
        elif mode == 3:
            # 9 random points rewiring with frozen background (heterogeneity)
            k = 9
            j = list(rng.choice(n, k, replace=False))
            # sequential void fill
            x = _void_fill(x, j, rng, n_cand=1200, noise=0.02)
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            msk = np.zeros(n, dtype=bool); msk[j] = True
            x = _anneal_mask(x, steps=s1, k0=70.0, k1=1000.0, lr0=0.03, lr1=0.004, mask=msk)
            x = _anneal_mask(x, steps=s2, k0=130.0, k1=2000.0, lr0=0.015, lr1=0.0015, mask=None)
        elif mode == 4:
            # medium global kick
            x += rng.randn(n, d) * 0.022
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal_mask(x, steps=s1 + s2, k0=90.0, k1=2000.0, lr0=0.02, lr1=0.0015, mask=None)
        else:
            # low-contact + random subset collective push
            k = 5
            j = [int(v) for v in order_low[:k]]
            # add subset noise
            x[j] += rng.randn(k, d) * 0.12
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            msk = np.zeros(n, dtype=bool); msk[j] = True
            x = _anneal_mask(x, steps=s1, k0=80.0, k1=1100.0, lr0=0.028, lr1=0.004, mask=msk)
            x = _anneal_mask(x, steps=s2, k0=140.0, k1=2100.0, lr0=0.015, lr1=0.0015, mask=None)
        v = _max_cos(x)
        if v < best:
            best = v
            x_best = x.copy()
            cur = v
            x_cur = x.copy()
        else:
            if v < cur or rng.rand() < np.exp(-(v - cur) / max(Temp, 1e-12)):
                cur = v
                x_cur = x.copy()
            if it % 14 == 0:
                x_cur = x_best.copy()
                cur = best
                Temp *= 0.90
                if Temp < 0.0004:
                    Temp = 0.0004
    try:
        x_best = _lp_polish(x_best, seconds=11.0, t0=t0, budget=BUDGET - 0.5)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.80):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        sel = D > -w
        W = np.zeros_like(G)
        # stable exp: D in [-w,0]
        W[sel] = np.exp(k * D[sel])
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        x = x - step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, r0=0.008, rmax=0.03, cap=800):
    n, d = x.shape
    x = _norm(x.copy())
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size > cap:
            part = np.argpartition(c_all, -cap)[-cap:]
            # keep only those above thresh if still too many? just take top cap
            sel = part
        c_sel = c_all[sel]
        I = iu0[sel]; J = iu1[sel]
        p = sel.size
        if p == 0:
            break
        nv = n * d + 1
        W = 2 * d + 1
        cols = np.empty((p, W), dtype=np.int64)
        vals = np.empty((p, W), dtype=np.float64)
        cols[:, :d] = I[:, None] * d + ar[None, :]
        cols[:, d:2*d] = J[:, None] * d + ar[None, :]
        cols[:, 2*d] = n * d
        vals[:, :d] = x[J]
        vals[:, d:2*d] = x[I]
        vals[:, 2*d] = -1.0
        rows = np.repeat(np.arange(p), W)
        A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
        erows = np.repeat(np.arange(n), d)
        ecols = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x, best

def _void_vec(x, keep_mask, rng, n_try=1500):
    kept = x[keep_mask]
    C = rng.standard_normal(size=(n_try, x.shape[1]))
    C = _norm(C)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[np.argmin(m)]

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_HARD = t0 + 172.0
    T_EXP = t0 + 158.0
    rng = _np.random.default_rng(7)
    n, d = 54, 23
    try:
        x0 = _np.load("best.npy").astype(float)
    except Exception:
        x0 = None
    try:
        xr = _np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if x0 is not None and x0.shape == (n, d):
        cands.append(_norm(x0))
    if xr is not None and xr.shape == (n, d):
        cands.append(_norm(xr))
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal(size=(n, d))))
    best = 1e9
    x_best = cands[0].copy()
    for c in cands:
        v = _max_cos(c)
        if v < best:
            best = v; x_best = c.copy()
    # initial squeeze 4s
    if _time.time() < T_EXP:
        xb, vb = _lp_polish(x_best, min(_time.time() + 4.0, T_EXP - 1.0), r0=0.004, rmax=0.02, cap=800)
        if vb < best:
            x_best, best = xb, vb
    it = 0
    while _time.time() < T_EXP:
        it += 1
        now = _time.time()
        if now >= T_EXP or now > T_HARD - 14:
            break
        G = x_best @ x_best.T
        _np.fill_diagonal(G, -2.0)
        ppm = _np.max(G, axis=1)
        cnt = (G > (best - 0.012)).sum(axis=1)
        order_lo = _np.argsort(cnt)
        order_worst = _np.argsort(-ppm)
        loose10 = order_lo[:10]
        mode = it % 7
        x = x_best.copy()
        if mode == 0:
            sig = 0.008 + 0.006 * rng.random()
            x = _norm(x + sig * rng.standard_normal(size=x.shape))
            steps, k0, k1, lr0, lr1, w0, w1 = 350, 100, 2500, 0.010, 0.0008, 0.05, 0.005
        elif mode == 1:
            sig = 0.018 + 0.010 * rng.random()
            x = _norm(x + sig * rng.standard_normal(size=x.shape))
            steps, k0, k1, lr0, lr1, w0, w1 = 400, 70, 2000, 0.013, 0.0009, 0.06, 0.006
        elif mode == 2:
            ksel = 2 + (it % 2)
            idx = order_lo[:ksel]
            for j in idx:
                mask = _np.ones(n, dtype=bool); mask[int(j)] = False
                nv = _void_vec(x, mask, rng, n_try=1200)
                x[int(j)] = _norm((nv + 0.02 * rng.standard_normal(size=(d,)))[None, :])[0]
            x = _norm(x + 0.003 * rng.standard_normal(size=x.shape))
            steps, k0, k1, lr0, lr1, w0, w1 = 450, 70, 2500, 0.013, 0.0008, 0.055, 0.005
        elif mode == 3:
            ksel = 1 + (it % 2)
            idx = order_worst[:ksel]
            for j in idx:
                mask = _np.ones(n, dtype=bool); mask[int(j)] = False
                nv = _void_vec(x, mask, rng, n_try=1200)
                x[int(j)] = nv
            x = _norm(x + 0.004 * rng.standard_normal(size=x.shape))
            steps, k0, k1, lr0, lr1, w0, w1 = 450, 80, 2500, 0.013, 0.0008, 0.055, 0.005
        elif mode == 4:
            idx = int(rng.choice(loose10))
            sig = 0.20 + 0.15 * rng.random()
            x[idx] = x[idx] + sig * rng.standard_normal(size=(d,))
            x[idx] /= _np.linalg.norm(x[idx])
            steps, k0, k1, lr0, lr1, w0, w1 = 400, 70, 2200, 0.013, 0.0009, 0.06, 0.005
        elif mode == 5:
            idx = int(rng.integers(0, n))
            v = rng.standard_normal(size=(d,)); v /= _np.linalg.norm(v)
            x[idx] = v
            steps, k0, k1, lr0, lr1, w0, w1 = 450, 60, 2000, 0.014, 0.001, 0.06, 0.006
        else:
            idxs = rng.choice(loose10, size=2, replace=False)
            for jj in idxs:
                v = rng.standard_normal(size=(d,)); v /= _np.linalg.norm(v)
                x[int(jj)] = v
            steps, k0, k1, lr0, lr1, w0, w1 = 500, 60, 2000, 0.014, 0.001, 0.06, 0.006
        # cheap anneal
        try:
            xa = _anneal(x, steps=steps, k0=k0, k1=k1, lr0=lr0, lr1=lr1, w0=w0, w1=w1)
        except Exception:
            continue
        va = _max_cos(xa)
        # only LP-polish if promising (within 6e-3 of best) to save time
        if va > best + 0.006:
            continue
        if _time.time() > T_HARD - 14:
            # no time for LP, accept if better
            if va < best:
                x_best, best = xa, va
            continue
        budget = 1.2 if mode in (0, 1) else 1.8
        deadline = min(_time.time() + budget, T_EXP, T_HARD - 12)
        if deadline <= _time.time() + 0.2:
            if va < best:
                x_best, best = xa, va
            continue
        r0 = 0.012 if mode in (0, 1) else 0.018
        try:
            yp, vp = _lp_polish(xa, deadline, r0=r0, rmax=0.04, cap=700)
        except Exception:
            continue
        if vp < best - 1e-12:
            x_best, best = yp, vp
    rem = T_HARD - _time.time()
    if rem > 3:
        try:
            xb, vb = _lp_polish(x_best, _time.time() + rem - 0.6, r0=0.004, rmax=0.025, cap=800)
            if vb < best:
                x_best, best = xb, vb
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

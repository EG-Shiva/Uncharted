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

def _smooth(x, steps, a0, a1, lr0, lr1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    # precompute log ratio for exp schedule
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # weights focused on near-max pairs
        # clip exponent to avoid overflow (<=0 so fine)
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True)
        Gn = Gt / (deg + 1.0)
        vel = mu * vel + Gn
        x = x - lr * vel
        # renormalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _lp_polish(x, deadline, r0=0.005, rmax=0.03, cap=800):
    n, d = x.shape
    x = _norm(x.copy())
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:cap]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = xb.copy()
        if time.time() >= deadline:
            break
    return xb, best

def _best_hole(x, idx, rng, n_cand=4000):
    n, d = x.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand = _norm(cand)
    G = cand @ x.T
    G[:, idx] = -10.0
    scores = G.max(axis=1)
    bi = int(np.argmin(scores))
    x[idx] = cand[bi]

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_HARD = t0 + 176.0
    T_EXPLORE = t0 + 150.0
    rng = _np.random.default_rng(7)
    try:
        x0 = _np.load("best.npy").astype(float)
    except Exception:
        x0 = None
    try:
        xr = _np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if x0 is not None and x0.shape == (54, 23):
        cands.append(_norm(x0))
    if xr is not None and xr.shape == (54, 23):
        cands.append(_norm(xr))
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal(size=(54, 23))))
    best = 1e9
    x_best = cands[0]
    for c in cands:
        v = _max_cos(c)
        if v < best:
            best = v
            x_best = c.copy()
    n, d = 54, 23
    it = 0
    # short LP snap to ensure tight (2s max)
    if _time.time() < T_EXPLORE:
        try:
            xb2, vb2 = _lp_polish(x_best, min(_time.time() + 2.0, T_EXPLORE), r0=0.003, rmax=0.02)
            if vb2 < best:
                x_best, best = xb2, vb2
        except Exception:
            pass
    while _time.time() < T_EXPLORE:
        it += 1
        now = _time.time()
        if now >= T_EXPLORE:
            break
        # contact degrees for guided kicks
        G = x_best @ x_best.T
        _np.fill_diagonal(G, -2.0)
        deg = (G > best - 0.008).sum(axis=1)
        order_lo = _np.argsort(deg)
        mode = it % 8
        y = x_best.copy()
        if mode == 0:
            sig = 0.015 + 0.015 * rng.random()
            y = _norm(y + sig * rng.standard_normal(size=y.shape))
            steps, a0, a1, lr0, lr1 = 700, 40.0, 600.0, 0.06, 0.004
        elif mode == 1:
            idx = int(rng.choice(order_lo[:12]))
            _best_hole(y, idx, rng, n_cand=4000)
            # tiny jitter others to break symmetry
            y = _norm(y + 0.003 * rng.standard_normal(size=y.shape))
            steps, a0, a1, lr0, lr1 = 900, 30.0, 700.0, 0.05, 0.003
        elif mode == 2:
            idxs = rng.choice(order_lo[:14], size=2, replace=False)
            for idx in idxs:
                _best_hole(y, int(idx), rng, n_cand=3000)
            steps, a0, a1, lr0, lr1 = 1000, 30.0, 700.0, 0.05, 0.003
        elif mode == 3:
            idx = int(rng.integers(0, n))
            v = rng.standard_normal(size=(d,))
            v /= _np.linalg.norm(v)
            y[idx] = v
            steps, a0, a1, lr0, lr1 = 900, 30.0, 700.0, 0.05, 0.003
        elif mode == 4:
            # shake endpoint of a max pair hard
            iu0, iu1 = _np.triu_indices(n, 1)
            c = (x_best @ x_best.T)[iu0, iu1]
            top = _np.argpartition(c, -40)[-40:]
            pick = int(rng.choice(top))
            i, j = int(iu0[pick]), int(iu1[pick])
            idx = i if rng.random() < 0.5 else j
            y[idx] = _norm((y[idx] + 0.3 * rng.standard_normal(size=(d,)))[None, :])[0]
            steps, a0, a1, lr0, lr1 = 800, 40.0, 600.0, 0.06, 0.004
        elif mode == 5:
            y = _norm(y + 0.04 * rng.standard_normal(size=y.shape))
            idx = int(rng.choice(order_lo[:10]))
            _best_hole(y, idx, rng, n_cand=3000)
            steps, a0, a1, lr0, lr1 = 900, 30.0, 650.0, 0.05, 0.003
        elif mode == 6:
            idxs = rng.choice(order_lo[:16], size=3, replace=False)
            for idx in idxs:
                _best_hole(y, int(idx), rng, n_cand=2500)
            steps, a0, a1, lr0, lr1 = 1200, 25.0, 800.0, 0.05, 0.003
        else:
            sig = 0.008 + 0.008 * rng.random()
            y = _norm(y + sig * rng.standard_normal(size=y.shape))
            steps, a0, a1, lr0, lr1 = 600, 50.0, 500.0, 0.06, 0.005
        # fast smooth descent (numpy only, no LP)
        try:
            ys = _smooth(y, steps, a0, a1, lr0, lr1, mu=0.85)
        except Exception:
            continue
        v = _max_cos(ys)
        if v < best - 1e-12:
            # confirm with short LP polish
            deadline = min(_time.time() + 2.5, T_HARD - 12.0)
            if deadline > _time.time() + 0.4:
                try:
                    yp, vp = _lp_polish(ys, deadline, r0=0.012, rmax=0.04)
                except Exception:
                    yp, vp = ys, v
                if vp < best:
                    x_best, best = yp, vp
                elif v < best:
                    # keep smoothed even if LP failed
                    x_best, best = ys, v
            else:
                x_best, best = ys, v
        # opportunistic: if close but not better, occasionally polish buffer
        # to avoid timeout we do not LP otherwise
        if _time.time() > T_HARD - 13:
            break
    rem = T_HARD - _time.time()
    if rem > 3:
        try:
            xb, vb = _lp_polish(x_best, _time.time() + rem - 0.6, r0=0.005, rmax=0.03)
            if vb < best:
                x_best, best = xb, vb
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

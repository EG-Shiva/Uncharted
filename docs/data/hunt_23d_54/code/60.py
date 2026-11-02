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

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu, deadline, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if (s & 63) == 0 and time.time() > deadline:
            break
        f = s / max(steps - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        # weights only for near-max band
        sel = D > -w
        W = np.zeros_like(G)
        # k*D in [-k*w,0]; keep in reasonable range
        Ds = D[sel]
        W[sel] = np.exp(k * Ds)
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        if an < 1e-15:
            continue
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        x = x - (lr / vn) * v
        x = _norm(x)
    return x

def _void_vec(kept, rng, n_try=1500):
    d = kept.shape[1]
    C = rng.standard_normal(size=(n_try, d))
    C = _norm(C)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(x, deadline, r0=0.006, rmax=0.03, cap=700):
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
        thr = m - max(20.0 * radius, 2e-4)
        # also hard window
        thr = max(thr, m - 0.01)
        sel = np.where(c_all > thr)[0]
        if sel.size > cap:
            part = np.argpartition(c_all, -cap)[-cap:]
            # keep only those above thr among top? just take top cap
            sel = part
        if sel.size == 0:
            break
        c_sel = c_all[sel]
        I = iu0[sel]; J = iu1[sel]
        p = sel.size
        nv = n * d + 1
        # vectorized A_ub
        r_rep = np.repeat(np.arange(p), d)
        rows = np.concatenate([r_rep, r_rep + 0, np.arange(p)])  # placeholder fix below
        # Actually need 2*d+1 per row: build correctly
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([(I[:, None] * d + ar).ravel(), (J[:, None] * d + ar).ravel(), np.full(p, nv - 1, dtype=np.int64)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
    return best_x, best

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    T_EXP = t0 + 138.0
    rng = np.random.default_rng(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (n, d):
        cands.append(_norm(xb))
    if xr is not None and xr.shape == (n, d):
        cands.append(_norm(xr))
    if not cands:
        cands.append(_norm(rng.standard_normal(size=(n, d))))
    best = 1e9
    x_best = cands[0]
    for c in cands:
        v = _max_cos(c)
        if v < best:
            best = v; x_best = c.copy()
    x_cur = x_best.copy(); cur = best
    Temp = 0.0006
    it = 0
    # quick tiny anneal to tighten loaded point (cheap)
    try:
        xt = _anneal(x_best.copy(), steps=250, k0=300, k1=3000, lr0=0.002, lr1=0.0003, w0=0.015, w1=0.004, mu=0.85, deadline=min(time.time()+4.0, T_EXP))
        vt = _max_cos(xt)
        if vt < best:
            best = vt; x_best = xt.copy(); x_cur = xt.copy(); cur = vt
    except Exception:
        pass
    while time.time() < T_EXP:
        it += 1
        now = time.time()
        if now > T_EXP - 0.5:
            break
        # contact info of current centre
        try:
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            ppm = np.max(G, axis=1)
            cnt = np.sum(G > (cur - 0.012), axis=1)
        except Exception:
            ppm = np.zeros(n); cnt = np.zeros(n)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-ppm)
        x = x_cur.copy()
        mode = it % 9
        if mode == 0:
            # small global shake
            x = x + 0.008 * rng.standard_normal(size=x.shape)
            x = _norm(x)
            steps, k0, k1, lr0, lr1, w0, w1, mu = 450, 80, 2500, 0.008, 0.0006, 0.05, 0.005, 0.85
        elif mode == 1:
            # medium global
            x = x + 0.02 * rng.standard_normal(size=x.shape)
            x = _norm(x)
            steps, k0, k1, lr0, lr1, w0, w1, mu = 500, 60, 2000, 0.010, 0.0007, 0.06, 0.006, 0.85
        elif mode == 2:
            # single worst -> void
            j = int(order_worst[0])
            mask = np.ones(n, dtype=bool); mask[j] = False
            nv = _void_vec(x[mask], rng, n_try=1500)
            x[j] = nv + 0.01 * rng.standard_normal(size=(d,))
            x[j] /= np.linalg.norm(x[j])
            steps, k0, k1, lr0, lr1, w0, w1, mu = 500, 70, 2500, 0.010, 0.0006, 0.055, 0.005, 0.86
        elif mode == 3:
            # 2-3 loose -> void
            ksel = 2 + (it % 2)
            idx = order_lo[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[int(j)] = False
                nv = _void_vec(x[mask], rng, n_try=1200)
                x[int(j)] = nv + 0.015 * rng.standard_normal(size=(d,))
            x = _norm(x)
            steps, k0, k1, lr0, lr1, w0, w1, mu = 550, 70, 2500, 0.010, 0.0006, 0.055, 0.005, 0.86
        elif mode == 4:
            # 2 worst -> void
            ksel = 2
            idx = order_worst[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[int(j)] = False
                nv = _void_vec(x[mask], rng, n_try=1200)
                x[int(j)] = nv + 0.015 * rng.standard_normal(size=(d,))
            x = _norm(x)
            steps, k0, k1, lr0, lr1, w0, w1, mu = 550, 70, 2500, 0.010, 0.0006, 0.055, 0.005, 0.86
        elif mode == 5:
            # collective subset shift (creates heterogeneity)
            ksel = 5 + int(rng.integers(0, 4))  # 5-8
            # pick spatially coherent subset: neighbours of a random seed in dot
            seed = int(rng.integers(0, n))
            sims = x_cur[seed] @ x_cur.T
            idx = np.argsort(-sims)[:ksel]
            u = rng.standard_normal(size=(d,)); u /= np.linalg.norm(u)
            # tangential component
            amp = 0.05 + 0.05 * rng.random()
            for j in idx:
                jj = int(j)
                uj = u - float(u @ x[jj]) * x[jj]
                uj /= np.linalg.norm(uj) + 1e-12
                x[jj] = x[jj] + amp * uj
            x = _norm(x)
            steps, k0, k1, lr0, lr1, w0, w1, mu = 550, 60, 2200, 0.010, 0.0006, 0.06, 0.005, 0.85
        elif mode == 6:
            # random single full reinsert
            j = int(rng.integers(0, n))
            vv = rng.standard_normal(size=(d,)); vv /= np.linalg.norm(vv)
            x[j] = vv
            steps, k0, k1, lr0, lr1, w0, w1, mu = 600, 60, 2000, 0.012, 0.0007, 0.06, 0.006, 0.85
        elif mode == 7:
            # opposite-void for loose point
            j = int(rng.choice(order_lo[:8]))
            row = (x_cur @ x_cur.T)[j].copy()
            nbrs = np.where(row > cur - 0.01)[0]
            if len(nbrs) > 0:
                cent = x_cur[nbrs].mean(axis=0)
                cent /= np.linalg.norm(cent) + 1e-12
                vv = -cent + 0.25 * rng.standard_normal(size=(d,))
                vv /= np.linalg.norm(vv)
                x[j] = vv
            else:
                vv = rng.standard_normal(size=(d,)); vv /= np.linalg.norm(vv)
                x[j] = vv
            steps, k0, k1, lr0, lr1, w0, w1, mu = 550, 70, 2500, 0.010, 0.0006, 0.055, 0.005, 0.86
        else:
            # shake loose region + small global
            x = x + 0.006 * rng.standard_normal(size=x.shape)
            j = int(rng.choice(order_lo[:10]))
            x[j] = x[j] + 0.2 * rng.standard_normal(size=(d,))
            x = _norm(x)
            steps, k0, k1, lr0, lr1, w0, w1, mu = 500, 70, 2400, 0.010, 0.0006, 0.055, 0.005, 0.86
        budget = 2.2
        deadline = min(now + budget, T_EXP)
        try:
            xa = _anneal(x, steps=steps, k0=k0, k1=k1, lr0=lr0, lr1=lr1, w0=w0, w1=w1, mu=mu, deadline=deadline)
            va = _max_cos(xa)
        except Exception:
            continue
        if va < best - 1e-12:
            best = va; x_best = xa.copy(); x_cur = xa.copy(); cur = va
        elif va < cur:
            cur = va; x_cur = xa.copy()
        else:
            # Metropolis sideways to allow basin change
            if rng.random() < float(np.exp(-(va - cur) / Temp)):
                # only allow limited uphill to avoid losing competitiveness
                if va < cur + 0.004:
                    cur = va; x_cur = xa.copy()
            # periodic reset if cur drifted too far
            if cur > best + 0.004:
                cur = best; x_cur = x_best.copy()
        # anneal temperature decay
        Temp *= 0.9995
        Temp = max(Temp, 0.00015)
    # final LP polish of best
    rem = T_END - time.time()
    if rem > 4.0:
        try:
            xp, vp = _lp_polish(x_best, time.time() + rem - 0.7, r0=0.005, rmax=0.03, cap=700)
            if vp < best:
                x_best, best = xp, vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

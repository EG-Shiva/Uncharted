import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _max_pair(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # find one argmax
    idx = int(np.argmax(G))
    n = G.shape[0]
    a = idx // n
    b = idx % n
    return m, int(a), int(b)


def _smooth_polish(x, steps=500, T0=0.03, T1=0.002, lr0=0.025, lr1=0.002, mu=0.5):
    x = _norm(x.copy())
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ x) / rs
        dot = (D * x).sum(axis=1, keepdims=True)
        Dt = D - dot * x
        v = mu * v - lr * Dt
        # keep velocity tangential
        vdot = (v * x).sum(axis=1, keepdims=True)
        v = v - vdot * x
        x = x + v
        x = _norm(x)
    return x


def _refine_void(F, c0, iters=70, T0=0.012, T1=0.002, s0=0.25, s1=0.03):
    c = c0.copy()
    nc = c / (np.linalg.norm(c) + 1e-300)
    c = nc
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        step = s0 * (s1 / s0) ** frac
        d = F @ c
        mx = float(d.max())
        w = np.exp((d - mx) / T)
        w = w / (w.sum() + 1e-300)
        D = w @ F
        dot = float(D @ c)
        Dt = D - dot * c
        c = c - step * Dt
        c = c / (np.linalg.norm(c) + 1e-300)
    return c


def _lp_polish(x, seconds, radius_init=2e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu0]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cfull > thr
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3500:
            idx = np.argpartition(-csel, 3500)[:3500]
            I, J, csel = I[idx], J[idx], csel[idx]
            p = 3500
        nv = n * d + 1
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar[None, :]).ravel()
        cols_J = (J[:, None] * d + ar[None, :]).ravel()
        cols_T = np.full(p, n * d, dtype=np.int64)
        cols = np.concatenate([cols_I, cols_J, cols_T])
        rows_I = np.repeat(np.arange(p), d)
        rows_J = np.repeat(np.arange(p), d)
        rows_T = np.arange(p)
        rows = np.concatenate([rows_I, rows_J, rows_T])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb


def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.RandomState(12345)
    n, d = 184, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=np.float64)
        except Exception:
            continue
        if c.shape == (n, d):
            c = _norm(c)
            cands.append(((_max_cos(c)), c))
    if len(cands) == 0:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        best_val, x_best = cands[0][0], cands[0][1].copy()
    x_best = _norm(x_best)
    x_cur = x_best.copy()
    v_cur = best_val
    no_improve = 0
    trial = 0
    TS = 0.005
    while time.time() - t0 < LIMIT - 25.0:
        trial += 1
        if time.time() - t0 > LIMIT - 26.0:
            break
        # stress of current walker
        Gc = x_cur @ x_cur.T
        np.fill_diagonal(Gc, -2.0)
        mcur = float(Gc.max())
        # max pair endpoints
        fidx = int(np.argmax(Gc))
        a0 = fidx // n
        b0 = fidx % n
        # loads
        W = np.exp((Gc - mcur) / TS)
        # diag is ~0 already
        load = W.sum(axis=1)
        order = np.argsort(-load, kind="stable")
        # kick size
        r = rng.rand()
        if trial % 22 == 0:
            k = 10 + int(rng.randint(0, 5))
        elif trial % 7 == 0:
            k = 7 + int(rng.randint(0, 4))
        else:
            k = 2 + int(rng.randint(0, 4))
        victims = set()
        # always break one end of max pair
        victims.add(int(a0) if rng.rand() < 0.5 else int(b0))
        if rng.rand() < 0.30:
            victims.add(int(a0))
            victims.add(int(b0))
        # one high-stress
        if rng.rand() < 0.85:
            top_pool = order[:15]
            victims.add(int(top_pool[rng.randint(len(top_pool))]))
        # one more high with smaller prob
        if rng.rand() < 0.35:
            top_pool2 = order[:30]
            victims.add(int(top_pool2[rng.randint(len(top_pool2))]))
        while len(victims) < k:
            victims.add(int(rng.randint(n)))
        victims = list(victims)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        x_keep = x_cur[keep_mask]
        x_new = x_cur.copy()
        filled = x_keep.copy()
        rng.shuffle(victims)
        for vv in victims:
            ncand = 4000 if (k <= 7) else 2500
            cand = rng.randn(ncand, d)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300)
            dots = cand @ filled.T
            md = dots.max(axis=1)
            # take 3 best
            if ncand >= 3:
                part = np.argpartition(md, 3)[:3]
            else:
                part = np.array([int(np.argmin(md))])
            best_c = None
            best_md = 1e9
            for bi in part:
                c0 = cand[int(bi)]
                cref = _refine_void(filled, c0, iters=60)
                sc = float((filled @ cref).max())
                if sc < best_md:
                    best_md = sc
                    best_c = cref
            # occasional noise
            if rng.rand() < 0.15:
                nz = best_c + 0.01 * rng.randn(d)
                nz /= (np.linalg.norm(nz) + 1e-300)
                sc2 = float((filled @ nz).max())
                if sc2 < best_md + 0.005:
                    # keep refined anyway most of time; rarely take noisy
                    if rng.rand() < 0.3:
                        best_c = nz
            x_new[vv] = best_c
            filled = np.vstack([filled, best_c[None, :]])
        # reconverge
        if k >= 9:
            steps = 800
        elif k >= 6:
            steps = 600
        else:
            steps = 450
        T0 = 0.022 + 0.018 * rng.rand()
        T1 = 0.0018 + 0.0012 * rng.rand()
        lr0 = 0.022 + 0.010 * rng.rand()
        x_try = _smooth_polish(x_new, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=0.002, mu=0.5)
        v_try = _max_cos(x_try)
        if v_try < best_val - 1e-12:
            budget = min(3.0, max(0.6, (LIMIT - 25.0 - (time.time() - t0)) / 10.0))
            x_snapped = _lp_polish(x_try, seconds=budget, radius_init=2e-3)
            v_snapped = _max_cos(x_snapped)
            if v_snapped < best_val:
                x_best = x_snapped.copy()
                best_val = v_snapped
                x_cur = x_snapped.copy()
                v_cur = v_snapped
                no_improve = 0
            else:
                # keep smooth if still best
                if v_try < best_val:
                    x_best = _norm(x_try)
                    best_val = v_try
                    x_cur = _norm(x_try)
                    v_cur = v_try
                    no_improve = 0
        elif v_try < v_cur:
            x_cur = _norm(x_try)
            v_cur = v_try
            no_improve = 0
        else:
            if v_try < v_cur + 0.0025 and rng.rand() < 0.18:
                x_cur = _norm(x_try)
                v_cur = v_try
            no_improve += 1
            if no_improve >= 28:
                x_cur = x_best.copy()
                v_cur = best_val
                no_improve = 0
                # small global jitter to leave basin
                if rng.rand() < 0.7 and (time.time() - t0 < LIMIT - 35.0):
                    xj = x_cur + 0.008 * rng.randn(n, d)
                    xj = _norm(xj)
                    xj = _smooth_polish(xj, steps=350, T0=0.015, T1=0.002, lr0=0.015, lr1=0.0015, mu=0.5)
                    vj = _max_cos(xj)
                    # accept as walker regardless to diversify
                    x_cur = xj
                    v_cur = vj
        if trial % 45 == 0 and (time.time() - t0 < LIMIT - 35.0):
            # occasional diversification from best
            xd = x_best + 0.012 * rng.randn(n, d)
            xd = _norm(xd)
            xd = _smooth_polish(xd, steps=400, T0=0.020, T1=0.002, lr0=0.020, lr1=0.002, mu=0.5)
            vd = _max_cos(xd)
            if vd < v_cur:
                x_cur = xd
                v_cur = vd
            if vd < best_val:
                xb2 = _lp_polish(xd, seconds=1.5, radius_init=1.5e-3)
                vb2 = _max_cos(xb2)
                if vb2 < best_val:
                    x_best = xb2
                    best_val = vb2
    # final tightening
    remain = LIMIT - (time.time() - t0) - 0.8
    if remain > 2.5:
        x_best = _smooth_polish(x_best, steps=350, T0=0.007, T1=0.0012, lr0=0.007, lr1=0.001, mu=0.5)
        remain2 = LIMIT - (time.time() - t0) - 0.4
        if remain2 > 1.0:
            x_best = _lp_polish(x_best, seconds=min(remain2, 21.0), radius_init=1.5e-3)
    x_best = _norm(x_best)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-12:
            v = rng.randn(d)
            v /= (np.linalg.norm(v) + 1e-300)
            x_best[i] = v
    x_best = _norm(x_best)
    return x_best

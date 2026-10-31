import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, float(rm.max())


def repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm(x)
    return x


def lp_polish(x, deadline):
    # deadline: absolute time.time() to stop
    x = _norm(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
            cnt = 6000
        if cnt == 0:
            radius *= 0.5
            continue
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def face_walk(x, rng, allow=2e-4, radius=0.012, window=0.006, cap=4000):
    n, d = x.shape
    try:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu = np.triu_indices(n, 1)
        c_all = G[iu]
        sel = c_all > (m - window)
        if int(np.count_nonzero(sel)) > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            return _norm(x + rng.standard_normal(x.shape) * 0.005)
        R = rng.standard_normal((n, d))
        R -= np.sum(R * x, axis=1, keepdims=True) * x
        rn = np.linalg.norm(R)
        if rn > 1e-12:
            R /= rn
        nv = n * d
        rows = np.repeat(np.arange(p), 2 * d)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d)], 1).ravel()
        vals = np.concatenate([x[J], x[I]], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        b_ub = (m + allow) - c
        obj = (-R.ravel()).astype(float)
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * nv, method="highs")
        if res.status != 0:
            return _norm(x + rng.standard_normal(x.shape) * 0.005)
        y = x + res.x.reshape(n, d)
        return _norm(y)
    except Exception:
        return _norm(x + rng.standard_normal(x.shape) * 0.005)


def refine_point(pt, others, steps=160, s0=0.03, alpha=110.0):
    pt = pt / np.linalg.norm(pt)
    for s in range(steps):
        dots = others @ pt
        m = float(dots.max())
        w = np.exp(alpha * (dots - m))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ others
        gt = g - float(g @ pt) * pt
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-13:
            break
        step = s0 / (1.0 + s * 0.04)
        pt = pt - (step / nrm) * gt
        pt /= np.linalg.norm(pt)
    return pt


def coord_sweep(x, worst_k=None, steps=140):
    n, d = x.shape
    rm, _ = row_max(x)
    order = np.argsort(-rm)
    if worst_k is not None:
        order = order[:worst_k]
    for idx in order:
        others = np.delete(x, idx, axis=0)
        x[idx] = refine_point(x[idx], others, steps=steps)
    return _norm(x)


def reinsert_worst(x, rng, n_replace=2, n_sample=8000):
    n, d = x.shape
    rm, m = row_max(x)
    order = np.argsort(-rm)
    K = min(n, max(n_replace * 4, 12))
    pool_idx = order[:K]
    chosen = pool_idx[rng.choice(len(pool_idx), size=min(n_replace, len(pool_idx)), replace=False)]
    y = x.copy()
    for idx in chosen:
        others = np.delete(y, idx, axis=0)
        best_pt = None
        best_v = 2.0
        chunk = 4000
        rem = n_sample
        while rem > 0:
            cs = min(chunk, rem)
            S = rng.standard_normal((cs, d))
            S = _norm(S)
            D = S @ others.T
            vm = D.max(axis=1)
            k = int(np.argmin(vm))
            if float(vm[k]) < best_v:
                best_v = float(vm[k])
                best_pt = S[k].copy()
            rem -= cs
        best_pt = refine_point(best_pt, others, steps=120, s0=0.03)
        y[idx] = best_pt
    return _norm(y)


def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    rng = np.random.default_rng(20261031)
    n, d = 272, 4
    cands = []
    try:
        bx = np.load("best.npy")
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            cands.append(_norm(bx))
    except Exception:
        pass
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            cands.append(_norm(rx))
    except Exception:
        pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [(max_cos(c), c) for c in cands]
    vals.sort(key=lambda t: t[0])
    best_v, best_x = vals[0][0], vals[0][1].copy()
    # quick initial lock to true local optimum
    best_x = lp_polish(best_x, min(T_END - 150.0, time.time() + 6.0))
    best_v = max_cos(best_x)
    # one greedy sweep immediately
    if time.time() < T_END - 145.0:
        best_x = coord_sweep(best_x, worst_k=80, steps=120)
        best_v = max_cos(best_x)
        best_x = lp_polish(best_x, min(T_END - 143.0, time.time() + 3.0))
        best_v = max_cos(best_x)

    pool = [(best_v, best_x.copy())]

    def base_pick():
        if len(pool) == 1 or rng.random() < 0.65:
            return pool[0][1]
        return pool[rng.integers(len(pool))][1]

    def add_pool(v, x):
        pool.append((v, x.copy()))
        pool.sort(key=lambda t: t[0])
        # keep diverse top 5
        del pool[5:]

    it = 0
    iu_all = np.triu_indices(n, 1)
    LOOP_END = T_END - 20.0
    while time.time() < LOOP_END:
        it += 1
        remain = LOOP_END - time.time()
        if remain < 2.0:
            break
        base = base_pick()
        rm, mcur = row_max(base)
        u = rng.random()
        # build one candidate per iteration (fast), occasionally two
        if u < 0.52:
            sigma = float(10.0 ** rng.uniform(-2.7, -1.05))
            xp = base + rng.standard_normal(base.shape) * sigma
            # push apart a random top pair
            if rng.random() < 0.6:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                cg = G[iu_all]
                k = 20
                top = np.argpartition(-cg, k - 1)[:k]
                ch = top[rng.integers(k)]
                i = int(iu_all[0][ch]); j = int(iu_all[1][ch])
                push = 0.5 * sigma + 0.004
                xp[i] += push * (base[i] - base[j])
                xp[j] += push * (base[j] - base[i])
            xp = _norm(xp)
            a0 = 18.0 + 14.0 * rng.random()
            a1 = 120.0 + 120.0 * rng.random()
            s0 = 0.006 + 0.008 * rng.random()
            s1 = 0.0005 + 0.0012 * rng.random()
            steps = 700 + int(rng.integers(0, 500))
            if remain < 25:
                steps = min(steps, 500)
            mu = 0.80 + 0.12 * rng.random()
            xp = repulse_gd(xp, steps, a0, a1, s0, s1, mu=mu)
            # light greedy fix of worst few
            if max_cos(xp) < best_v + 0.001:
                xp = coord_sweep(xp, worst_k=20, steps=80)
        elif u < 0.72:
            # face walk then short GD
            allow = float(10.0 ** rng.uniform(-4.0, -2.7))
            rad = float(10.0 ** rng.uniform(-2.3, -1.3))
            xp = face_walk(base, rng, allow=allow, radius=rad)
            a0 = 25.0 + 10.0 * rng.random()
            a1 = 150.0 + 100.0 * rng.random()
            xp = repulse_gd(xp, 450 + int(rng.integers(0, 300)), a0, a1, 0.007, 0.0008, mu=0.85)
            if max_cos(xp) < best_v + 0.001:
                xp = coord_sweep(xp, worst_k=20, steps=80)
        elif u < 0.88:
            nr = int(rng.integers(1, 5))
            xp = reinsert_worst(base, rng, n_replace=nr, n_sample=7000)
            xp = xp + rng.standard_normal(xp.shape) * 0.0025
            xp = _norm(xp)
            a0 = 22.0 + 10.0 * rng.random()
            a1 = 150.0 + 100.0 * rng.random()
            xp = repulse_gd(xp, 450 + int(rng.integers(0, 300)), a0, a1, 0.006, 0.0008, mu=0.85)
        else:
            # partial fresh restart biased to spread
            xp = _norm(rng.standard_normal(base.shape))
            xp = repulse_gd(xp, 450, 7.0, 28.0, 0.02, 0.005, mu=0.8)
            xp = repulse_gd(xp, 550, 28.0, 200.0, 0.008, 0.0008, mu=0.85)
        v = max_cos(xp)
        if v < best_v - 1e-12:
            best_v, best_x = v, xp.copy()
            add_pool(v, xp)
            # lock immediately with short LP
            if time.time() < LOOP_END - 3.0:
                yp = lp_polish(xp, min(LOOP_END, time.time() + 2.0))
                vy = max_cos(yp)
                if vy < best_v:
                    best_v, best_x = vy, yp.copy()
                    add_pool(vy, yp)
                elif vy < best_v + 0.002:
                    add_pool(vy, yp)
        elif v < best_v + 0.0035:
            # maybe keep diverse
            if len(pool) < 5 or v < pool[-1][0]:
                add_pool(v, xp)
            # opportunistically polish near-best
            if v < best_v + 0.0006 and rng.random() < 0.30 and time.time() < LOOP_END - 3.0:
                yp = lp_polish(xp, min(LOOP_END, time.time() + 1.2))
                vy = max_cos(yp)
                if vy < best_v - 1e-12:
                    best_v, best_x = vy, yp.copy()
                add_pool(vy, yp)
        # keep pool sorted, ensure pool[0] is best
        pool.sort(key=lambda t: t[0])
        if pool[0][0] > best_v:
            pool.insert(0, (best_v, best_x.copy()))
            del pool[5:]
    # final intensive polish of best
    bx = best_x.copy()
    # full sweeps
    while time.time() < T_END - 12.0:
        # alternate sweep + short LP; break if no progress
        old = max_cos(bx)
        bx = coord_sweep(bx, worst_k=None, steps=160)
        nv = max_cos(bx)
        if time.time() >= T_END - 12.0:
            break
        bx = lp_polish(bx, min(T_END - 4.0, time.time() + 4.0))
        nv2 = max_cos(bx)
        if min(nv, nv2) >= old - 1e-13:
            break
        if time.time() >= T_END - 4.0:
            break
    if time.time() < T_END - 2.0:
        bx = lp_polish(bx, T_END - 1.0)
    bx = _norm(bx)
    # safety: return best seen
    if max_cos(bx) > best_v:
        bx = best_x.copy()
    return _norm(bx)

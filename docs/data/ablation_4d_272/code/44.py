import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n = x.shape[0]
    vel = np.zeros_like(x)
    best_x = x.copy()
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            best_x = x.copy()
        W = np.exp(np.clip(alpha * (G - m), -50, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm < 1e-14:
            continue
        x = x - (smax / nrm) * vel
        x = _norm_rows(x)
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    fv = float(Gf.max())
    if fv < bv:
        return x, fv
    return best_x, bv

def _lp_polish(x, deadline, cap=5000):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_lse(Y, z0, beta=60.0, steps=180, s0=0.03, mu=0.5):
    z = z0 / (np.linalg.norm(z0) + 1e-300)
    vel = np.zeros(4)
    s = s0
    for _ in range(steps):
        dots = Y @ z
        m = float(dots.max())
        w = np.exp(np.clip(beta * (dots - m), -50, 0.0))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        g = (w @ Y) / sw
        gt = g - float(g @ z) * z
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = mu * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        z = z - (s / vn) * vel
        z = z / (np.linalg.norm(z) + 1e-300)
        s *= 0.997
    return z / (np.linalg.norm(z) + 1e-300)

def _largest_hole(Y, rng, n_samples=50000):
    d = Y.shape[1]
    best_s = 2.0
    best_c = None
    bs = 5000
    # batch generation to bound memory
    for a in range(0, n_samples, bs):
        nb = min(bs, n_samples - a)
        C = rng.standard_normal((nb, d))
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Y.T
        sm = S.max(axis=1)
        k = int(np.argmin(sm))
        if float(sm[k]) < best_s:
            best_s = float(sm[k])
            best_c = C[k].copy()
    best_c = _refine_lse(Y, best_c, beta=60.0, steps=200, s0=0.03, mu=0.5)
    return best_c

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    n, d = 272, 4
    rng0 = np.random.default_rng(20261031)
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        cand.append(_norm_rows(rng0.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    base = cand[bi].copy()
    base_val = float(vals[bi])
    gbest = base.copy()
    gval = base_val

    # contact analysis on base
    G = base @ base.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    # degrees with tight threshold
    deg = np.count_nonzero(G > m0 - 0.002, axis=1)
    # dual-stress proxy
    stress = np.exp(np.clip(60.0 * (G - m0), -50, 0.0)).sum(axis=1)
    order_deg = np.argsort(-deg)
    order_stress = np.argsort(-stress)

    # define deletion sets (parallel time slices)
    plans = []
    # 0: highest-degree 32
    plans.append(("deg32", order_deg[:32].tolist()))
    # 1: highest-stress 28
    k1 = [i for i in order_stress[:28].tolist()]
    plans.append(("stress28", k1))
    # 2: spatially clustered 36 (seed = worst point + nearest)
    worst = int(np.argmax(stress))
    dots_w = base @ base[worst]
    order_near = np.argsort(-dots_w)
    plans.append(("cluster36", order_near[:36].tolist()))
    # 3: random 32
    rng_r = np.random.default_rng(777)
    plans.append(("rand32", rng_r.choice(n, size=32, replace=False).tolist()))
    # 4: highest-degree 24 (smaller hole, if time)
    plans.append(("deg24", order_deg[:24].tolist()))

    for pi, (pname, del_idx) in enumerate(plans):
        if time.time() > HARD_END - 25:
            break
        rng = np.random.default_rng(1000 + pi * 977)
        k = len(del_idx)
        mask = np.ones(n, dtype=bool)
        mask[np.array(del_idx, dtype=int)] = False
        Y = base[mask].copy()  # (n-k,4)
        # anneal remaining code
        Y, _ = _smooth_gd(Y, steps=500, a0=25.0, a1=320.0, s0=0.05, s1=0.004, mu=0.85)
        if time.time() > HARD_END - 20:
            # evaluate anyway below
            pass
        else:
            Y = _lp_polish(Y, min(HARD_END - 15, time.time() + 2.5), cap=5000)
        cur = Y.copy()
        # reinsert one at a time
        for ins in range(k):
            if time.time() > HARD_END - 18:
                break
            z = _largest_hole(cur, rng, n_samples=50000)
            # freeze older points briefly: optimise only new point (already refined)
            # brief jitter + re-refine against updated cur to be safe
            z = _refine_lse(cur, z, beta=60.0, steps=120, s0=0.02, mu=0.5)
            cur = np.vstack([cur, z.reshape(1, 4)])
            cur = _norm_rows(cur)
            # fully re-anneal after every 6 insertions (and at end)
            if ((ins + 1) % 6 == 0) or (ins + 1 == k):
                if time.time() > HARD_END - 15:
                    continue
                # short full anneal; fewer steps mid-way
                steps = 350 if (ins + 1 != k) else 600
                cur, _ = _smooth_gd(cur, steps=steps, a0=30.0, a1=350.0, s0=0.02 if (ins+1)!=k else 0.03, s1=0.003, mu=0.85)
        # final polish for this trial
        if cur.shape[0] != n:
            continue
        if time.time() < HARD_END - 10:
            cur, _ = _smooth_gd(cur, steps=400, a0=40.0, a1=400.0, s0=0.015, s1=0.002, mu=0.85)
        if time.time() < HARD_END - 6:
            cur = _lp_polish(cur, min(HARD_END - 2, time.time() + 4.0), cap=6000)
        try:
            v = _max_cos(cur)
        except Exception:
            continue
        if v < gval and np.all(np.isfinite(cur)):
            gval = v
            gbest = cur.copy()
            # update base for subsequent plans to exploit new basin
            base = cur.copy()
            base_val = v
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            deg = np.count_nonzero(G > m0 - 0.002, axis=1)
            stress = np.exp(np.clip(60.0 * (G - m0), -50, 0.0)).sum(axis=1)

    # fallback + final safety polish if time remains
    if time.time() < HARD_END - 8:
        try:
            gbest = _lp_polish(gbest, min(HARD_END - 1, time.time() + 3.0), cap=6000)
            gval = _max_cos(gbest)
        except Exception:
            pass
    gbest = _norm_rows(gbest)
    # strict feasibility: no zero rows / NaN (already ensured)
    return gbest

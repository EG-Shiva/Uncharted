import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam(x, steps, T0, T1, lr0, lr1, deadline=1e99):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ x) / max(s, 1e-30)
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (it + 51)
        c2 = 1.0 - b2 ** (it + 51)
        st = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        st = st - (st * x).sum(axis=1, keepdims=True) * x
        nm = np.linalg.norm(st, axis=1, keepdims=True)
        sc = np.minimum(1.0, 3.0 / (nm + 1e-12))
        x = x - lr * st * sc
        x = _norm(x)
    return x

def _replace(x, w, K):
    n, d = x.shape
    R = np.random.randn(K, d)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-30
    S = R @ x.T
    S[:, w] = -2.0
    sm = S.max(axis=1)
    k = int(np.argmin(sm))
    return R[k], float(sm[k])

def _lp_jam(x, best, deadline, r0=8e-4):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    radius = r0
    stall = 0
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        cfull = Gfull[iu]
        m = float(cfull.max())
        if m < best:
            best = m
        thresh = m - max(25.0 * radius, 5e-7)
        sel = cfull > thresh
        cnt = int(sel.sum())
        if cnt < 200:
            kk = min(800, cfull.size)
            idx = np.argpartition(-cfull, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            cnt = kk
        if cnt > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-4000:]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
            cnt = 4000
        I = iu[0][sel]; J = iu[1][sel]
        p = cnt
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            stall += 1
            if stall >= 6:
                rnd = np.random.randn(n, d) * 1e-6
                rnd -= (rnd * x).sum(axis=1, keepdims=True) * x
                x = _norm(x + rnd)
                stall = 0
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-13:
                x = y; best = v
                radius = min(radius * 1.6, 0.015)
                improved = True; stall = 0
                break
        if not improved:
            radius *= 0.5
            stall += 1
            if stall >= 8:
                rnd = np.random.randn(n, d) * (radius * 0.5 + 1e-7)
                rnd -= (rnd * x).sum(axis=1, keepdims=True) * x
                yt = _norm(x + rnd)
                vt = _max_cos(yt)
                if vt < best + 2e-7:
                    x = yt
                    if vt < best:
                        best = vt
                stall = 0
    return x, best

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 166.0
    t_end = t0 + BUDGET
    tA = t0 + 108.0
    best_x = None; best_v = 1e99
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (182, 9):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(np.random.randn(182, 9))
        best_v = _max_cos(best_x)
    n, d = 182, 9
    x_cur = best_x.copy(); cur_v = best_v
    it = 0
    while time.time() < tA:
        it += 1
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        if it % 10 == 0:
            krep = 3
        elif it % 3 == 0:
            krep = 2
        else:
            krep = 1
        order = np.argsort(-row_max)
        pool = order[:5]
        # deterministic random pick among top-5
        perm = np.random.permutation(5)[:krep]
        cand_w = sorted([int(pool[j]) for j in perm])
        x_try = x_cur.copy()
        for w in cand_w:
            K = 15000 if krep == 1 else 7000
            bv, _ = _replace(x_try, int(w), K)
            x_try[int(w)] = bv
        if it % 4 == 0:
            sig = 0.012
        elif it % 11 == 0:
            sig = 0.03
        else:
            sig = 0.0
        if sig > 0:
            rnd = np.random.randn(n, d) * sig
            rnd -= (rnd * x_try).sum(axis=1, keepdims=True) * x_try
            x_try = _norm(x_try + rnd)
        if it % 2 == 0:
            T0, lr0 = 0.008, 0.004
        else:
            T0, lr0 = 0.015, 0.005
        steps = 320 if krep == 1 else 220
        x_try = _adam(x_try, steps, T0, 0.0008, lr0, 0.0004, deadline=tA + 2)
        v = _max_cos(x_try)
        if v < best_v - 1e-12:
            # jam new best briefly to squeeze bundle
            x_try, v = _lp_jam(x_try, v, min(time.time() + 4.0, t_end - 25.0), r0=6e-4)
            best_v = v; best_x = x_try.copy()
            x_cur = x_try.copy(); cur_v = v
        elif v < cur_v or (v < best_v + 6e-4 and np.random.rand() < 0.12):
            x_cur = x_try.copy(); cur_v = v
        else:
            sig2 = 0.025 if (it % 6 == 0) else 0.005
            rnd = np.random.randn(n, d) * sig2
            rnd -= (rnd * best_x).sum(axis=1, keepdims=True) * best_x
            x_cur = _norm(best_x + rnd)
            cur_v = best_v + sig2 * 0.05
        if time.time() >= tA:
            break
    x = best_x.copy()
    if time.time() < t_end - 30:
        x = _adam(x, 350, 0.003, 0.0004, 0.001, 0.0002, deadline=t_end - 25)
        v = _max_cos(x)
        if v < best_v:
            best_v = v; best_x = x.copy()
        else:
            x = best_x.copy()
    if time.time() < t_end - 2.0:
        x, best_v = _lp_jam(x, best_v, t_end - 0.8, r0=8e-4)
        best_x = x.copy()
    best_x = _norm(best_x)
    return best_x

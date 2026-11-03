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

def _adam_relax(x, steps, T0, T1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
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
        # bias correction with global step approx
        c1 = 1.0 - b1 ** (it + 1 + 50)
        c2 = 1.0 - b2 ** (it + 1 + 50)
        st = (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        # tangent
        dn = (st * x).sum(axis=1, keepdims=True)
        st = st - dn * x
        # clip
        nm = np.linalg.norm(st, axis=1, keepdims=True)
        sc = np.minimum(1.0, 3.0 / (nm + 1e-12))
        x = x - lr * st * sc
        x = _norm(x)
    return x

def _best_replacement(x, w, K, extra=None):
    n, d = x.shape
    R = np.random.randn(K, d)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    if extra is not None:
        R = np.vstack([R, extra])
    S = R @ x.T
    # ignore self column
    S[:, w] = -2.0
    sm = S.max(axis=1)
    k = int(np.argmin(sm))
    return R[k], float(sm[k])

def _lp_polish(x, best, deadline):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    radius = 8e-4
    stall = 0
    while time.time() < deadline and radius > 1e-13:
        Gfull = x @ x.T
        cfull = Gfull[iu]
        m = float(cfull.max())
        if m < best:
            best = m
        thresh = max(15.0 * radius, 3e-7)
        sel = cfull > m - thresh
        p = int(sel.sum())
        if p < 10:
            kk = min(800, cfull.size)
            idx = np.argpartition(-cfull, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-3500:]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
            p = 3500
        I = iu[0][sel]; J = iu[1][sel]
        csel = cfull[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            stall += 1
            if stall >= 6:
                # jitter to escape flat face
                rnd = np.random.randn(n, d) * 1e-6
                rnd = rnd - (rnd * x).sum(axis=1, keepdims=True) * x
                x = _norm(x + rnd)
                stall = 0
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.3, 1.2e-3)
            stall = 0
        else:
            radius *= 0.5
            stall += 1
            if stall >= 8:
                rnd = np.random.randn(n, d) * (radius * 0.5 + 1e-7)
                rnd = rnd - (rnd * x).sum(axis=1, keepdims=True) * x
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
    BUDGET = 168.0
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
    # Phase A: basin hopping until ~95s
    tA = t0 + 95.0
    t_end = t0 + BUDGET
    x_cur = best_x.copy()
    cur_v = best_v
    it = 0
    while time.time() < tA:
        it += 1
        # compute worst points of x_cur
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        # choose number to replace
        if it % 7 == 0:
            krep = 2
        elif it % 23 == 0:
            krep = 3
        else:
            krep = 1
        worst = np.argsort(-row_max)[:krep+2]
        # randomize order slightly
        cand_w = worst[:krep]
        x_try = x_cur.copy()
        # sequential greedy reinsertion
        for w in cand_w:
            K = 12000 if krep == 1 else 6000
            bv, _ = _best_replacement(x_try, int(w), K)
            x_try[int(w)] = bv
        # short relax with random schedule
        T0 = 0.008 if (it % 2 == 0) else 0.015
        T1 = 0.001
        lr0, lr1 = 0.004, 0.0004
        steps = 350 if krep == 1 else 250
        # add small kick before relax half the time
        if it % 3 == 0:
            sig = 0.01
            rnd = np.random.randn(n, d) * sig
            rnd = rnd - (rnd * x_try).sum(axis=1, keepdims=True) * x_try
            # only kick non-worst a bit less? kick all
            x_try = _norm(x_try + rnd)
        x_try = _adam_relax(x_try, steps, T0, T1, lr0, lr1)
        v = _max_cos(x_try)
        if v < best_v:
            best_v = v; best_x = x_try.copy()
            x_cur = x_try.copy(); cur_v = v
        elif v < cur_v or (v < best_v + 8e-4 and np.random.rand() < 0.15):
            x_cur = x_try.copy(); cur_v = v
        else:
            # restart from best with jitter, magnitude decays if stuck
            sig = 0.004 if (it % 5 != 0) else 0.02
            rnd = np.random.randn(n, d) * sig
            rnd = rnd - (rnd * best_x).sum(axis=1, keepdims=True) * best_x
            x_cur = _norm(best_x + rnd)
            # quick eval
            # cur_v = _max_cos(x_cur)
            cur_v = best_v + sig * 0.1
        if time.time() >= tA:
            break
    # Phase B: LP jamming from best
    x = best_x.copy()
    # a quick medium relax to smooth before LP
    if time.time() < t_end - 30:
        x = _adam_relax(x, 400, 0.004, 0.0005, 0.001, 0.0002)
        v = _max_cos(x)
        if v < best_v:
            best_v = v; best_x = x.copy()
        else:
            x = best_x.copy()
    if time.time() < t_end - 2.0:
        x, best_v = _lp_polish(x, best_v, t_end - 1.0)
        best_x = x.copy()
    best_x = _norm(best_x)
    return best_x

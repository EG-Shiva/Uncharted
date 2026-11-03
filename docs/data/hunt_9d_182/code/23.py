import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam(X0, steps, T0, T1, lr0, lr1, deadline=1e18):
    X = _norm(X0.copy())
    n, d = X.shape
    try:
        bv = _max_cos(X)
    except Exception:
        bv = 1e9
    bx = X.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    b1p = 1.0
    b2p = 1.0
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(np.sum(E) * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / max(s, 1e-30)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        m2 *= b2
        m2 += (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        st = mh / (np.sqrt(vh) + eps)
        dn = np.sum(st * X, axis=1, keepdims=True)
        st = st - dn * X
        nm = np.linalg.norm(st, axis=1, keepdims=True)
        sc = np.minimum(1.0, 2.5 / (nm + 1e-12))
        X = X - lr * st * sc
        X = _norm(X)
        if (it % 120) == 119:
            v = _max_cos(X)
            if v < bv:
                bv = v
                bx = X.copy()
    v = _max_cos(X)
    if v < bv:
        bv = v
        bx = X.copy()
    return bx, bv

def _hole_replace(X, w, K):
    n, d = X.shape
    R = np.random.randn(K, d)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ X.T
    S[:, int(w)] = -2.0
    sm = np.max(S, axis=1)
    k = int(np.argmin(sm))
    return R[k], float(sm[k])

def _lp_jam(x0, best, deadline):
    x = _norm(x0.copy())
    n, d = x.shape
    bx = x.copy()
    try:
        v0 = _max_cos(x)
        if v0 < best:
            best = v0
            bx = x.copy()
    except Exception:
        pass
    iu = np.triu_indices(n, 1)
    radius = 6e-4
    stall = 0
    while time.time() < deadline and radius > 1e-13:
        Gf = x @ x.T
        cf = Gf[iu]
        m = float(np.max(cf))
        if m < best:
            best = m
            bx = x.copy()
        thresh = max(15.0 * radius, 3e-7)
        sel = cf > (m - thresh)
        p = int(np.sum(sel))
        if p < 500:
            kk = min(800, cf.size)
            idx = np.argpartition(-cf, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 2200:
            idx = np.where(sel)[0]
            order = np.argsort(cf[idx])[-2200:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
            p = 2200
        I = iu[0][sel]
        J = iu[1][sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            stall += 1
            if stall >= 6:
                rnd = np.random.randn(n, d) * 8e-7
                rnd -= np.sum(rnd * x, axis=1, keepdims=True) * x
                x = _norm(x + rnd)
                stall = 0
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25):
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-15:
                x = y
                best = v
                bx = y.copy()
                radius = min(radius * 1.3, 1.0e-3)
                stall = 0
                improved = True
                break
        if not improved:
            radius *= 0.5
            stall += 1
            if stall >= 8:
                amp = radius * 0.5 + 8e-8
                rnd = np.random.randn(n, d) * amp
                rnd -= np.sum(rnd * x, axis=1, keepdims=True) * x
                yt = _norm(x + rnd)
                vt = _max_cos(yt)
                if vt < best + 3e-7:
                    x = yt
                    if vt < best:
                        best = vt
                        bx = yt.copy()
                stall = 0
    return bx, best

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    t_hop = t0 + 118.0
    best_x = None
    best_v = 1e99
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (182, 9):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(np.random.randn(182, 9))
        best_v = _max_cos(best_x)
    n, d = 182, 9
    x_cur = best_x.copy()
    cur_v = best_v
    it = 0
    while time.time() < t_hop:
        it += 1
        if time.time() > t_hop:
            break
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        row_max = np.max(G, axis=1)
        if it % 30 == 0:
            krep = 3
        elif it % 8 == 0:
            krep = 2
        else:
            krep = 1
        worst = np.argsort(-row_max)[:krep + 2]
        # slight random choice among top to diversify
        if krep == 1 and np.random.rand() < 0.3:
            cand = [int(worst[1])]
        else:
            cand = [int(w) for w in worst[:krep]]
        x_try = x_cur.copy()
        for w in cand:
            K = 20000 if krep == 1 else 8000
            # cap K near end to save time
            if time.time() > t_hop - 10:
                K = 6000
            bv_, _ = _hole_replace(x_try, int(w), K)
            x_try[int(w)] = bv_
        if it % 3 == 0:
            sig = 0.008 if (it % 2 == 0) else 0.015
            rnd = np.random.randn(n, d) * sig
            rnd -= np.sum(rnd * x_try, axis=1, keepdims=True) * x_try
            x_try = _norm(x_try + rnd)
        # schedule choice
        if it % 4 == 0:
            T0, T1 = 0.012, 0.0008
            lr0, lr1 = 0.005, 0.0004
            steps = 320
        elif it % 4 == 1:
            T0, T1 = 0.006, 0.0005
            lr0, lr1 = 0.003, 0.0003
            steps = 380
        elif it % 4 == 2:
            T0, T1 = 0.003, 0.0003
            lr0, lr1 = 0.002, 0.0002
            steps = 380
        else:
            T0, T1 = 0.008, 0.0006
            lr0, lr1 = 0.004, 0.0003
            steps = 300
        x_rel, v_rel = _adam(x_try, steps, T0, T1, lr0, lr1, deadline=t_hop + 2.0)
        v = float(v_rel)
        if v < best_v:
            best_v = v
            best_x = x_rel.copy()
            x_cur = x_rel.copy()
            cur_v = v
        elif v < cur_v or (v < best_v + 6e-4 and np.random.rand() < 0.18):
            x_cur = x_rel.copy()
            cur_v = v
        else:
            sig = 0.004 if (it % 6 != 0) else 0.022
            rnd = np.random.randn(n, d) * sig
            rnd -= np.sum(rnd * best_x, axis=1, keepdims=True) * best_x
            x_cur = _norm(best_x + rnd)
            cur_v = best_v + sig * 0.05
    # smooth before LP if time allows
    x = best_x.copy()
    if time.time() < t_end - 35.0:
        xs, vs = _adam(x, 500, 0.004, 0.0004, 0.0012, 0.0002, deadline=t_end - 30.0)
        if vs < best_v:
            best_v = vs
            best_x = xs.copy()
            x = xs.copy()
        else:
            x = best_x.copy()
    if time.time() < t_end - 3.0:
        x, best_v = _lp_jam(x, best_v, t_end - 1.0)
        best_x = x.copy()
    best_x = _norm(best_x)
    return best_x

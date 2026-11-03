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
    return float(np.max(G))

def _smooth(x, steps, T0, T1, lr0, lr1):
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ x) / rs
        dot = (D * x).sum(axis=1, keepdims=True)
        Dt = D - dot * x
        x = x - lr * Dt
        x = _norm(x)
    return x

def _greedy(x, steps, m0, m1, lr):
    n = x.shape[0]
    for s in range(steps):
        f = s / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        d = gt / np.maximum(deg, 1.0)
        x = x - lr * d
        x = _norm(x)
    return x

def _refine_hole(Xf, y, steps=80):
    T0, T1 = 0.03, 0.005
    lr0, lr1 = 0.08, 0.01
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        sc = Xf @ y
        m = float(np.max(sc))
        w = np.exp((sc - m) / T)
        sw = float(np.sum(w)) + 1e-300
        g = w @ Xf / sw
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _best_hole(Xf, rng, ns=3000):
    d = Xf.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y = _norm(Y)
    S = Y @ Xf.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i].copy()

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 8e-4
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() > deadline - 2.0:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1800:
            idx = np.argpartition(-cs, 1800)[:1800]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1800
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = max(1.0, deadline - time.time() - 0.6)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = _norm(x + res.x[:-1].reshape(n, d))
        except Exception:
            radius *= 0.5
            continue
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    try:
        x0 = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        x0 = np.asarray(np.load("record.npy"), dtype=np.float64)
    if x0.shape != (n, d):
        x0 = x0.reshape(n, d)
    x0 = _norm(x0)
    gval = _max_cos(x0)
    gbest = x0.copy()
    cur = x0.copy()
    cur_val = gval
    iu = np.triu_indices(n, 1)

    trial = 0
    while time.time() < HOP_END and time.time() < HARD - 45:
        trial += 1
        frac = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        Tsa = 6e-06 * (1.0 - frac) + 3e-07
        base = cur.copy()
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        thr = mcur - 0.004
        cf = G[iu]
        freq = np.zeros(n, dtype=np.int64)
        mk = cf >= thr
        if np.any(mk):
            np.add.at(freq, iu[0][mk], 1)
            np.add.at(freq, iu[1][mk], 1)
        permax = np.max(G, axis=1)
        mode = trial % 3
        if mode == 0:
            K = 16 + int(rng.integers(0, 5))  # 16-20 cluster kill
            c = int(rng.integers(0, n))
            order = np.argsort(-G[c], kind="stable")
            rem = order[:K].tolist()
        elif mode == 1:
            Kcap = 14 + int(rng.integers(0, 5))
            u = rng.standard_normal(size=(d,))
            u /= np.linalg.norm(u) + 1e-300
            proj = base @ u
            rem = np.argsort(-proj, kind="stable")[:Kcap].tolist()
        else:
            # hubs + colds + random
            hubs = np.argsort(-freq, kind="stable")[:6].tolist()
            colds = np.argsort(permax, kind="stable")[:6].tolist()
            cand = [i for i in range(n) if i not in set(hubs) | set(colds)]
            extra = rng.choice(cand, size=6, replace=False).tolist()
            rem = hubs + colds + extra
        remset = set(rem)
        keep = [i for i in range(n) if i not in remset]
        Xfix = base[keep].copy()
        # sequential farthest-hole regrowth
        newpts = []
        Xgrow = Xfix.copy()
        for _ in range(len(rem)):
            y0 = _best_hole(Xgrow, rng, ns=3000)
            y0 = _refine_hole(Xgrow, y0, steps=70)
            newpts.append(y0)
            Xgrow = np.vstack([Xgrow, y0[None, :]])
        xt = np.empty_like(base)
        xt[keep] = Xfix
        xt[rem] = np.array(newpts)
        xt = _norm(xt)
        # reconverge cheap
        xt = _smooth(xt, steps=170, T0=0.022, T1=0.0032, lr0=0.030, lr1=0.003)
        xt = _greedy(xt, steps=70, m0=0.006, m1=0.0008, lr=0.007)
        # extra light smooth to settle
        xt = _smooth(xt, steps=40, T0=0.004, T1=0.002, lr0=0.004, lr1=0.001)
        v = _max_cos(xt)
        if v < gval:
            gval = float(v); gbest = xt.copy()
        if v < cur_val + Tsa or v < gval:
            cur = xt.copy(); cur_val = float(v)
        # early exit if time low
        if time.time() > HOP_END:
            break

    # final polish with remaining time
    deadline = HARD - 1.0
    remain = deadline - time.time()
    if remain > 10:
        try:
            yp, vp = _lp_polish(gbest, deadline)
            if vp < gval and np.all(np.isfinite(yp)):
                gbest = yp; gval = vp
        except Exception:
            pass
    gbest = _norm(gbest)
    return gbest

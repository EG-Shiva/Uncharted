import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth_refine(x, steps=300, a0=40.0, a1=320.0, lr0=0.03, lr1=0.003):
    n, d = x.shape
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = np.max(G)
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        direction = gt / S
        x = x - lr * direction
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy_refine(x, steps=120, m0=0.008, m1=0.0008, lr=0.008):
    n, d = x.shape
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        thr = m - margin
        M = (G > thr)
        grad = M.astype(np.float64) @ x
        deg = M.sum(axis=1, keepdims=True).astype(np.float64)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        direction = gt / np.maximum(deg, 1.0)
        x = x - lr * direction
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, t_start, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        now = time.time()
        if now - t_start >= seconds:
            break
        if now > deadline - 1.5:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        cf = G[iu]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu[0][sel]; J = iu[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            idx = np.argpartition(-cs, 1500)[:1500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1500
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
        try:
            tl = max(1.0, deadline - time.time() - 0.5)
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def _build_E8():
    s2 = np.sqrt(2.0)
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8, dtype=np.float64)
                    v[i] = si / s2
                    v[j] = sj / s2
                    roots.append(v)
    # (±1)^8 with even minus parity, scaled to unit
    for mask in range(256):
        bits = [(1.0 if ((mask >> k) & 1 else -1.0) for k in range(8)]
        nminus = sum(1 for b in bits if b < 0)
        if nminus % 2 == 0:
            v = np.array(bits, dtype=np.float64) / (2.0 * s2)
            roots.append(v)
    E = np.stack(roots, axis=0)
    return E  # (240,8)

def solve():
    t0 = time.time()
    HARD = t0 + 174.0
    DEADLINE165 = t0 + 165.0
    E = _build_E8()
    # safety fallback
    try:
        xb = np.asarray(np.load("best.npy"), dtype=np.float64)
        if xb.shape != (184, 8):
            xb = xb.reshape(184, 8)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        fb = _max_cos(xb)
    except Exception:
        try:
            xb = np.asarray(np.load("record.npy"), dtype=np.float64).reshape(184, 8)
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
            fb = _max_cos(xb)
        except Exception:
            xb = None
            fb = np.inf
    Ggram = E @ E.T
    NSEED = 24
    subsets = []
    THR = 0.499
    for s in range(NSEED):
        if time.time() > DEADLINE165 - 55:
            break
        if time.time() > HARD - 60:
            break
        rng = np.random.default_rng(1000 + s)
        active = np.ones(240, dtype=bool)
        act_idx = np.arange(240)
        # greedy delete 56
        for _ in range(56):
            act = act_idx[active]
            # sub gram
            Gs = Ggram[np.ix_(act, act)]
            deg = np.sum(Gs >= THR, axis=1) - 1  # exclude self
            mx = np.max(deg)
            cand = np.where(deg == mx)[0]
            if len(cand) > 1:
                pick = int(rng.integers(0, len(cand)))
                drop_local = int(cand[pick])
            else:
                drop_local = int(cand[0])
            drop_global = int(act[drop_local])
            active[drop_global] = False
        act = act_idx[active]
        Gs = Ggram[np.ix_(act, act)]
        iu = np.triu_indices(len(act), 1)
        cnt = int(np.sum(Gs[iu] >= THR))
        subsets.append((cnt, act.copy()))
    if len(subsets) == 0:
        return xb
    subsets.sort(key=lambda z: z[0])
    K = min(4, len(subsets))
    top = subsets[:K]
    best_x = None
    best_v = np.inf
    # reconverge each top subset
    for k in range(K):
        if time.time() > DEADLINE165 - 35:
            break
        if time.time() > HARD - 45:
            break
        _, act = top[k]
        x = E[act].copy()
        rng = np.random.default_rng(2000 + k)
        jit_mag = 0.01 + 0.01 * (k % 3)  # 0.01,0.02,0.03 diversity, group-agnostic
        # tangential jitter
        R = rng.standard_normal(size=x.shape)
        dot = np.sum(R * x, axis=1, keepdims=True)
        Rt = R - dot * x
        nrm = np.linalg.norm(Rt, axis=1, keepdims=True) + 1e-30
        Rt = Rt / nrm
        x = x + jit_mag * Rt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = _smooth_refine(x, steps=300, a0=40.0, a1=320.0, lr0=0.03, lr1=0.003)
        if time.time() > HARD - 40:
            v = _max_cos(x)
            if v < best_v:
                best_v = v; best_x = x.copy()
            break
        x = _greedy_refine(x, steps=120, m0=0.008, m1=0.0008, lr=0.008)
        v = _max_cos(x)
        if v < best_v:
            best_v = v; best_x = x.copy()
    if best_x is None:
        return xb
    # single 25s LP polish for best candidate
    remain_hard = HARD - time.time() - 1.0
    if remain_hard > 3 and time.time() < DEADLINE165:
        lp_budget = min(25.0, remain_hard, max(0.0, DEADLINE165 - time.time() - 1.0))
        if lp_budget > 3:
            best_x = _lp_polish(best_x, lp_budget, time.time(), HARD)
            best_v = _max_cos(best_x)
    # return better of E8-derived vs fallback
    if xb is not None and fb <= best_v:
        return xb
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    if not np.all(np.isfinite(best_x)):
        return xb
    return best_x

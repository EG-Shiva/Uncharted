import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_IU = None

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _row_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 400:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _voids(base, rng, k, n_samp=2500):
    if k <= 0:
        return np.zeros((0, 4))
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ base.T
    sc = s.max(axis=1)
    order = np.argsort(sc)[:k * 8]
    pool = cand[order]
    out = []
    for v0 in pool:
        v = v0.copy()
        for _ in range(20):
            d1 = base @ v
            mx = d1.max()
            w = np.exp(30.0 * (d1 - mx))
            g = w @ base
            gt = g - np.dot(g, v) * v
            nrm = np.linalg.norm(gt)
            if nrm < 1e-12:
                break
            v = v - 0.10 * gt / nrm
            v /= np.linalg.norm(v)
        out.append(v)
    out = np.stack(out)
    scores = (out @ base.T).max(axis=1)
    idx = np.argsort(scores)[:k * 2]
    sel = []
    rem = list(idx)
    if rem:
        sel.append(rem.pop(0))
        while len(sel) < k and rem:
            S = out[sel] @ out[rem].T
            mm = S.max(axis=0)
            crit = scores[rem] + 0.3 * mm
            j = int(np.argmin(crit))
            sel.append(rem.pop(j))
    return out[np.array(sel)]

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    HARD = 168.0
    xb = None
    xr = None
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                c = np.asarray(c, dtype=float)
                if c.shape == (163, 4):
                    c = c / np.linalg.norm(c, axis=1, keepdims=True)
                    c = np.nan_to_num(c)
                    c = c / np.linalg.norm(c, axis=1, keepdims=True)
                    cands.append(c)
            except Exception:
                pass
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    n, d = 163, 4
    cur = best.copy()
    cur_val = best_val
    trial = 0
    # hopping phase, strictly time guarded
    while time.time() - t0 < 112 and trial < 24:
        if time.time() - t0 > HARD - 30:
            break
        Tmet = 0.0008 * (0.3 ** (trial / 24.0)) + 1e-5
        mv = trial % 4
        y = cur.copy()
        if mv == 0:
            sig = [0.05, 0.12, 0.03][(trial // 4) % 3]
            y = y + rng.normal(scale=sig, size=y.shape)
        elif mv == 1:
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            k = int(rng.integers(10, 24))
            idx = np.argsort(dots)[-k:]
            noise = rng.normal(scale=0.010, size=y.shape)
            noise[idx] = rng.normal(scale=0.20, size=(k, d))
            y = y + noise
        elif mv == 2:
            rm, G = _row_maxs(y)
            m = float(rm.max())
            thr = m - 1e-7
            C = (G >= thr).sum(axis=1)
            k = int(rng.integers(2, 5))
            worst = np.argsort(C + 0.5 * rm)[-k:]
            keep = np.array([i for i in range(n) if i not in set(worst.tolist())])
            voids = _voids(y[keep], rng, k)
            for t, w in enumerate(worst):
                y[w] = voids[t]
            y = y + rng.normal(scale=0.008, size=y.shape)
        else:
            rm, G = _row_maxs(y)
            m = float(rm.max())
            ratt = np.where(rm < m - 5e-4)[0]
            target = []
            if len(ratt) > 0:
                kk = min(len(ratt), int(rng.integers(1, 3)))
                o = np.argsort(rm[ratt])[:kk]
                target = ratt[o].tolist()
            w = int(np.argmax(rm))
            if w not in target:
                target.append(w)
            if rng.random() < 0.3:
                r = int(rng.integers(0, n))
                if r not in target:
                    target.append(r)
            target = np.array(target, dtype=int)
            keep = np.array([i for i in range(n) if i not in set(target.tolist())])
            voids = _voids(y[keep], rng, len(target))
            for t, w2 in enumerate(target):
                y[w2] = voids[t]
            y = y + rng.normal(scale=0.009, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if mv == 1:
            y = _smooth(y, iters=280, T0=0.030, T1=0.0025, lr0=0.011, lr1=0.0012)
        elif mv == 2:
            y = _smooth(y, iters=250, T0=0.022, T1=0.002, lr0=0.009, lr1=0.001)
        else:
            y = _smooth(y, iters=220, T0=0.020, T1=0.002, lr0=0.008, lr1=0.001)
        remain = HARD - (time.time() - t0)
        if remain > 8.0:
            sec = 1.0 if mv in (2, 3) else 0.7
            sec = min(sec, max(0.4, (HARD - 12.0 - (time.time() - t0)) / 8.0))
            if sec > 0.35:
                y = _lp_polish(y, seconds=sec)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best = y.copy(); best_val = v
            cur = y.copy(); cur_val = v
        else:
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Tmet, 1e-9)):
                cur = y.copy(); cur_val = v
        trial += 1
    # final squeeze of incumbent
    try:
        if time.time() - t0 < HARD - 6:
            best = _smooth(best, iters=180, T0=0.010, T1=0.0015, lr0=0.005, lr1=0.0008)
        left = HARD - (time.time() - t0) - 1.0
        if left > 2.0:
            left = min(left, 30.0)
            best = _lp_polish(best, seconds=left, r0=1.5e-3)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

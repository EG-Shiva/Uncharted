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

def _smooth(x, steps, T0, T1, lr0, lr1, rng=None):
    x = _norm(x.copy())
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ x) / rs
        dot = (D * x).sum(axis=1, keepdims=True)
        Dt = D - dot * x
        # clip extreme steps
        nr = np.linalg.norm(Dt, axis=1, keepdims=True) + 1e-300
        # scale lr adaptively to avoid blowup
        x = x - lr * Dt
        x = _norm(x)
    return x

def _lp_polish(x, seconds, r0=1.0e-3, pcap=1800):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() - start < seconds and radius > 1e-13:
        it += 1
        if time.time() - start > seconds - 0.3:
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
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 3e-7:
                # tight enough; try one smaller restart once
                if it > 30:
                    break
    return xb

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.RandomState(7)
    n, d = 184, 8
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    else:
        x_best = _norm(rng.randn(n, d)); best = _max_cos(x_best)
    x_cur = x_best.copy()
    cur = best

    trial = 0
    # main hopping, reserve ~22s for final
    while time.time() - t0 < LIM - 24.0:
        trial += 1
        # choose base: mostly best, sometimes wandering cur
        if rng.rand() < 0.8:
            base = x_best; base_val = best
        else:
            base = x_cur; base_val = cur
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        # frequencies of near-max pairs for victim choice
        iu = np.triu_indices(n, 1)
        cf = G[iu]
        # worst pair
        w = int(np.argmax(cf))
        wi, wj = int(iu[0][w]), int(iu[1][w])
        # near-max threshold for jam estimate
        thr = mcur - 0.004
        mask = cf >= thr
        freq = np.zeros(n, dtype=np.int64)
        if np.any(mask):
            np.add.at(freq, iu[0][mask], 1)
            np.add.at(freq, iu[1][mask], 1)
        order_hi = np.argsort(-freq, kind="stable")
        order_lo = np.argsort(freq, kind="stable")
        r = rng.rand()
        if trial % 28 == 0:
            k = 9 + int(rng.randint(0, 4))
        elif trial % 7 == 0:
            k = 5 + int(rng.randint(0, 4))
        else:
            k = 2 + int(rng.randint(0, 3))
        victims = set()
        victims.add(wi if rng.rand() < 0.5 else wj)
        if rng.rand() < 0.5:
            victims.add(wi); victims.add(wj)
        if rng.rand() < 0.6:
            victims.add(int(order_hi[rng.randint(min(15, n))]))
        if rng.rand() < 0.5:
            victims.add(int(order_lo[rng.randint(min(25, n))]))
        while len(victims) < k:
            victims.add(int(rng.randint(n)))
        victims = list(victims)[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        kept = base[keep_mask]
        x_new = base.copy()
        fill = kept  # (n-k, d)
        # sequential greedy void fill, cheap sampling
        for v in victims:
            ncand = 2500 if (k < 8) else 1500
            cand = rng.randn(ncand, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300
            md = (cand @ fill.T).max(axis=1)
            bi = int(np.argmin(md))
            cb = cand[bi]
            if rng.rand() < 0.20:
                nb = cb + 0.015 * rng.randn(d)
                nb /= np.linalg.norm(nb) + 1e-300
                if float(fill @ nb.reshape(-1, 1).squeeze() if False else (fill @ nb).max()) < md[bi] + 0.015:
                    cb = nb
            x_new[v] = cb
            fill = np.vstack([fill, cb[None, :]])
        # anneal params jitter
        T0 = 0.020 + 0.020 * rng.rand()
        T1 = 0.0015 + 0.0015 * rng.rand()
        lr0 = 0.028
        lr1 = 0.002
        steps = 380 if k <= 6 else 480
        # keep steps modest to stay fast
        if time.time() - t0 > LIM - 40:
            steps = 250
        x_try = _smooth(x_new, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        v_try = _max_cos(x_try)
        # wandering acceptance
        if v_try < cur:
            x_cur = x_try; cur = v_try
        elif rng.rand() < 0.08:
            x_cur = x_try; cur = v_try
        if v_try < best - 1e-12:
            x_best = x_try.copy(); best = v_try
            x_cur = x_try.copy(); cur = v_try
        # occasional global jitter of wandering state
        if trial % 35 == 0 and time.time() - t0 < LIM - 30:
            xp = x_best + 0.012 * rng.randn(n, d)
            xp = _norm(xp)
            xp = _smooth(xp, steps=220, T0=0.015, T1=0.002, lr0=0.02, lr1=0.002)
            vp = _max_cos(xp)
            if vp < cur:
                x_cur = xp; cur = vp
            if vp < best:
                x_best = xp.copy(); best = vp
    # final: light smooth then LP tighten
    remain = LIM - (time.time() - t0) - 0.8
    if remain > 4.0:
        # short cool-down
        x_best = _smooth(x_best, steps=250, T0=0.006, T1=0.0012, lr0=0.006, lr1=0.001)
        remain2 = LIM - (time.time() - t0) - 0.4
        if remain2 > 1.5:
            x_best = _lp_polish(x_best, seconds=min(remain2, 20.0), r0=1.2e-3)
    x_best = _norm(x_best)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-12:
            v = rng.randn(d); v /= np.linalg.norm(v)
            x_best[i] = v
    return _norm(x_best)

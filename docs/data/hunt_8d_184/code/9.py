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

def _smooth(x, steps, T0, T1, lr0, lr1):
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
        x = x - lr * Dt
        x = _norm(x)
    return x

def _lp(x, seconds, r0=1.5e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            idx = np.argpartition(-cs, 2500)[:2500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2500
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
            if radius < 5e-7:
                break
    return xb

def solve():
    t0 = time.time()
    LIM = 170.0
    rng = np.random.RandomState(12345)
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
        best, xb = cands[0][0], cands[0][1].copy()
    else:
        xb = _norm(rng.randn(n, d)); best = _max_cos(xb)
    # short warm LP snap on start to get tight if time allows (2s)
    # skip heavy work if little time; go straight to hopping
    trial = 0
    # pre-allocate index pools
    while time.time() - t0 < LIM - 28.0:
        trial += 1
        # kick size
        r = rng.rand()
        if trial % 30 == 0:
            k = 10 + int(rng.randint(0, 5))
        elif trial % 8 == 0:
            k = 6 + int(rng.randint(0, 4))
        else:
            k = 1 + int(rng.randint(0, 3))
        # find bottleneck participants from current best
        G = xb @ xb.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        # top pairs
        iu = np.triu_indices(n, 1)
        cf = G[iu]
        # threshold for near-max
        topk = 60
        if len(cf) > topk:
            th = np.partition(cf, -topk)[-topk]
        else:
            th = mcur - 1e-9
        Ii = iu[0][cf >= th]; Ji = iu[1][cf >= th]
        freq = np.zeros(n, dtype=np.int64)
        np.add.at(freq, Ii, 1); np.add.at(freq, Ji, 1)
        order = np.argsort(freq)[::-1]
        victims = set()
        # force include worst pair members with prob
        worst = np.argmax(cf)
        wi, wj = int(iu[0][worst]), int(iu[1][worst])
        victims.add(wi if rng.rand() < 0.5 else wj)
        if rng.rand() < 0.6:
            victims.add(wi); victims.add(wj)
        # add high-freq jammed
        if rng.rand() < 0.7:
            victims.add(int(order[rng.randint(min(12, n))]))
        # add rattler/low-freq (contacts approx by tight thr)
        if rng.rand() < 0.5:
            low = np.argsort(freq)
            victims.add(int(low[rng.randint(min(20, n))]))
        while len(victims) < k:
            victims.add(int(rng.randint(n)))
        victims = list(victims)[:max(k, 2)]
        keep = np.ones(n, dtype=bool); keep[victims] = False
        kept = xb[keep]
        xn = xb.copy()
        fill = kept.copy()
        for v in victims:
            nc = 4000 if (k < 8) else 2500
            cand = rng.randn(nc, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300
            md = (cand @ fill.T).max(axis=1)
            bi = int(np.argmin(md))
            b = cand[bi]
            if rng.rand() < 0.2:
                nb = b + 0.01 * rng.randn(d)
                nb /= np.linalg.norm(nb) + 1e-300
                if float(fill @ nb .max() if False else (fill @ nb).max()) < md[bi] + 0.01:
                    b = nb
            xn[v] = b
            fill = np.vstack([fill, b[None, :]])
        T0 = 0.02 + 0.02 * rng.rand()
        steps = 700 if k >= 8 else 550
        xt = _smooth(xn, steps=steps, T0=T0, T1=0.002, lr0=0.03, lr1=0.002)
        vt = _max_cos(xt)
        if vt < best - 1e-12:
            # snap with short LP
            bud = min(3.0, max(0.6, (LIM - 28.0 - (time.time() - t0)) / 10.0))
            xt2 = _lp(xt, seconds=bud, r0=2e-3)
            v2 = _max_cos(xt2)
            if v2 < best:
                xb, best = xt2, v2
            elif vt < best:
                xb, best = _norm(xt), vt
        elif vt < best + 0.001 and rng.rand() < 0.02:
            # rare sideways diversification: smooth-perturb best slightly
            xp = xb + 0.008 * rng.randn(n, d)
            xp = _smooth(_norm(xp), steps=300, T0=0.012, T1=0.002, lr0=0.015, lr1=0.002)
            vp = _max_cos(xp)
            if vp < best:
                xb, best = _lp(xp, seconds=1.5), vp
                best = _max_cos(xb)
        if time.time() - t0 > LIM - 29.0:
            break
    # final polish
    rem = LIM - (time.time() - t0) - 0.8
    if rem > 2.0:
        xb = _smooth(xb, steps=350, T0=0.007, T1=0.0012, lr0=0.007, lr1=0.001)
        rem2 = LIM - (time.time() - t0) - 0.4
        if rem2 > 1.0:
            xb = _lp(xb, seconds=min(rem2, 24.0), r0=1.2e-3)
    xb = _norm(np.where(np.isfinite(xb), xb, 0.0))
    nr = np.linalg.norm(xb, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-12:
            v = rng.randn(d); v /= np.linalg.norm(v)
            xb[i] = v
    return _norm(xb)

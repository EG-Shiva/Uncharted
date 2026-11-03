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
    # annealed softmax (log-sum-exp) tangential repulsion
    x = _norm(x.copy())
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f if steps > 1 else T0
        lr = lr0 * (lr1 / lr0) ** f if steps > 1 else lr0
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

def _pick_holes(Xkeep, rng, k, n_sample=4000):
    d = Xkeep.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y = _norm(Y)
    S = Y @ Xkeep.T
    m = S.max(axis=1)
    order = np.argsort(m, kind="stable")
    picked = []
    for idx in order:
        y = Y[idx]
        ok = True
        for q in picked:
            if float(y @ q) > 0.92:
                ok = False
                break
        if ok:
            picked.append(y.copy())
            if len(picked) >= k:
                break
    # fallback if too correlated
    if len(picked) < k:
        for idx in order:
            picked.append(Y[idx].copy())
            if len(picked) >= k:
                break
    return np.array(picked[:k])

def _lp_polish(x, seconds, pcap=1800):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-13:
        if time.time() - start > seconds - 0.5:
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
    return xb

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(0)
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
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _norm(rng.standard_normal(size=(n, d)))
        gval = _max_cos(gbest)
    # wandering state
    x_cur = gbest.copy()
    cur = gval
    iu = np.triu_indices(n, 1)

    trial = 0
    # hopping phase: reserve ~28s for final refine+LP
    while time.time() - t0 < LIM - 28.0:
        trial += 1
        if trial > 600:
            break
        # base: mostly global best, sometimes wanderer
        if rng.random() < 0.82:
            base = gbest
        else:
            base = x_cur
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        cf = G[iu]
        w = int(np.argmax(cf))
        wi, wj = int(iu[0][w]), int(iu[1][w])
        # contact frequency for hub / rattler detection
        thr = mcur - 0.004
        mask = cf >= thr
        freq = np.zeros(n, dtype=np.int64)
        if np.any(mask):
            np.add.at(freq, iu[0][mask], 1)
            np.add.at(freq, iu[1][mask], 1)
        hub = int(np.argmax(freq))
        low = int(np.argmin(freq))
        # victim count schedule
        r = rng.random()
        if trial % 25 == 0:
            k = 8 + int(rng.integers(0, 3))
        elif trial % 8 == 0:
            k = 5 + int(rng.integers(0, 3))
        else:
            k = 2 + int(rng.integers(0, 3))
        victims = set()
        victims.add(wi if rng.random() < 0.5 else wj)
        victims.add(hub)
        victims.add(low)
        # add random low-freq points
        order_lo = np.argsort(freq, kind="stable")
        ptr = 0
        while len(victims) < k:
            if ptr < n and rng.random() < 0.7:
                victims.add(int(order_lo[ptr]))
                ptr += 1
            else:
                victims.add(int(rng.integers(0, n)))
            ptr += 1
            if ptr > 5 * n:
                break
        victims = list(victims)[:k]
        k = len(victims)
        keep_idx = np.array([i for i in range(n) if i not in set(victims)], dtype=int)
        Xkeep = base[keep_idx]
        holes = _pick_holes(Xkeep, rng, k, n_sample=4000)
        x_try = base.copy()
        for vi, h in zip(victims, holes):
            # small tangential jitter to break symmetry
            jitter = rng.standard_normal(size=d) * 0.01
            jitter -= float(jitter @ h) * h
            y = h + jitter
            y /= (np.linalg.norm(y) + 1e-300)
            x_try[vi] = y
        # anneal params varied per trial
        T0 = float(rng.uniform(0.012, 0.028))
        T1 = float(rng.uniform(0.0015, 0.004))
        lr0 = float(rng.uniform(0.02, 0.045))
        lr1 = float(rng.uniform(0.002, 0.006))
        steps = int(rng.integers(160, 300))
        x_try = _smooth(x_try, steps, T0, T1, lr0, lr1)
        v = _max_cos(x_try)
        if v < gval:
            gval = v
            gbest = x_try.copy()
            x_cur = x_try.copy()
            cur = v
        elif v < cur or rng.random() < 0.08:
            x_cur = x_try.copy()
            cur = v

    # fine anneal of global best (cheap, no LP)
    try:
        gbest = _smooth(gbest, 350, 0.012, 0.0018, 0.02, 0.002)
        gval = _max_cos(gbest)
    except Exception:
        pass
    # single final LP tightening with remaining time
    remain = LIM - (time.time() - t0)
    if remain > 6.0:
        try:
            gbest = _lp_polish(gbest, min(remain - 1.0, 24.0))
        except Exception:
            pass
    gbest = _norm(gbest)
    return gbest

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _anneal(x, steps, T0, T1, lr0, lr1, rng=None, kick=0.0):
    x = _norm(x.copy())
    n, d = x.shape
    if kick > 0 and rng is not None:
        x = x + kick * rng.standard_normal(size=x.shape)
        x = _norm(x)
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

def _hole_for_fixed(Xf, rng, n_sample=3000):
    d = Xf.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    S = Y @ Xf.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx], float(m[idx])

def _lp(x, seconds, t_start=None, deadline=None, pcap=1500, r0=1.2e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    if t_start is None:
        t_start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t_start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(np.max(cf))
        thr = m - max(20.0 * radius, 2e-9)
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
            if radius < 4e-7:
                break
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
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
    # quick contact info from gbest for victim choice caching
    trial = 0
    iu = np.triu_indices(n, 1)
    # hopping phase: leave ~25s for final polish
    while time.time() < DEADLINE - 25.0:
        trial += 1
        if time.time() > DEADLINE - 26:
            break
        G = gbest @ gbest.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        cf = G[iu]
        # contact frequency using tight threshold
        thr_f = mcur - 1e-4
        sel_f = cf >= thr_f
        freq = np.zeros(n, dtype=np.int64)
        np.add.at(freq, iu[0][sel_f], 1)
        np.add.at(freq, iu[1][sel_f], 1)
        order_hi = np.argsort(-freq)
        order_lo = np.argsort(freq)
        worst = int(np.argmax(cf))
        wi, wj = int(iu[0][worst]), int(iu[1][worst])
        # kick size schedule
        r = rng.random()
        if trial % 25 == 0:
            k = 8 + int(rng.integers(0, 5))
        elif trial % 6 == 0:
            k = 4 + int(rng.integers(0, 3))
        else:
            k = 1 + int(rng.integers(0, 3))
        victims = set()
        victims.add(wi if rng.random() < 0.5 else wj)
        if rng.random() < 0.5:
            victims.add(wi); victims.add(wj)
        # jammed hub
        if rng.random() < 0.75:
            victims.add(int(order_hi[int(rng.integers(0, min(15, n)))]))
        # rattler / low-contact
        if rng.random() < 0.65:
            victims.add(int(order_lo[int(rng.integers(0, min(25, n)))]))
        while len(victims) < k:
            if rng.random() < 0.5:
                victims.add(int(order_hi[int(rng.integers(0, min(30, n)))]))
            else:
                victims.add(int(rng.integers(0, n)))
        victims = list(victims)[:k]
        x = gbest.copy()
        # global small tangential noise to loosen jam
        if rng.random() < 0.7:
            sig = 0.008 + 0.02 * rng.random()
            x = x + sig * rng.standard_normal(size=x.shape)
            x = _norm(x)
        # relocate / kick victims
        # for hole relocation use non-victim points as fixed set
        mask = np.ones(n, dtype=bool); mask[victims] = False
        Xfixed_base = x[mask]
        for v in victims:
            u = rng.random()
            if u < 0.55:
                # hole relocation: fixed = all except v (use current x)
                Xf = np.delete(x, v, axis=0)
                try:
                    y, _ = _hole_for_fixed(Xf, rng, n_sample=2500)
                except Exception:
                    y = rng.standard_normal(size=(d,)); y /= (np.linalg.norm(y) + 1e-300)
                # small jitter around hole
                y = y + 0.02 * rng.standard_normal(size=(d,))
                y /= (np.linalg.norm(y) + 1e-300)
                x[v] = y
            else:
                sig = 0.25 + 0.45 * rng.random()
                y = x[v] + sig * rng.standard_normal(size=(d,))
                y /= (np.linalg.norm(y) + 1e-300)
                x[v] = y
        # anneal: randomize schedule slightly per trial
        T0 = 0.012 * (0.7 + 0.8 * rng.random())
        T1 = 0.0015 * (0.7 + 0.8 * rng.random())
        lr0 = 0.05 * (0.7 + 0.6 * rng.random())
        lr1 = 0.008 * (0.7 + 0.6 * rng.random())
        steps = 180 + int(rng.integers(0, 120))
        try:
            x = _anneal(x, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        v = _max_cos(x)
        # screen: only LP if somewhat promising to save time
        # always LP if beats gval, else LP with prob if close
        do_lp = False
        if v < gval:
            do_lp = True
            lp_budget = 2.5
        elif v < gval + 0.002 and v < 0.505 and (rng.random() < 0.35):
            do_lp = True
            lp_budget = 1.2
        elif v < gval + 0.005 and (trial % 12 == 0):
            do_lp = True
            lp_budget = 1.0
        if not do_lp:
            continue
        if time.time() > DEADLINE - 26:
            break
        try:
            xp, vp = _lp(x, seconds=lp_budget, t_start=time.time(), deadline=DEADLINE - 25.0)
        except Exception:
            continue
        if vp < gval - 1e-12:
            gbest, gval = xp, vp
        # opportunistic extended polish if big win and time remains
        if vp < gval and False:
            pass
    # final polish of best
    remain = DEADLINE - time.time()
    if remain > 3:
        try:
            xp, vp = _lp(gbest, seconds=min(remain - 1.0, 22.0), t_start=time.time(), deadline=DEADLINE)
            if vp < gval:
                gbest, gval = xp, vp
        except Exception:
            pass
    gbest = _norm(gbest)
    return gbest

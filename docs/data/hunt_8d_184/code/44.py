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
    n, d = x.shape
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

def _hole_for_fixed(Xf, rng, ns=2500):
    d = Xf.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y = _norm(Y)
    S = Y @ Xf.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i].copy()

def _lp_polish(x, budget, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        now = time.time()
        if now > deadline - 2.0:
            break
        if now > deadline - budget - 50:  # safety, never loop past deadline
            pass
        if radius < 1e-13:
            break
        # stop if budget exhausted (checked via deadline)
        if time.time() > deadline - 1.5:
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
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(1.0, deadline - time.time() - 0.5)})
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
        if time.time() > deadline - 2.0:
            break
    return xb, best

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    HOP_END = t0 + 125.0
    rng = np.random.default_rng(12345)
    n, d = 184, 8
    try:
        xb0 = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        xb0 = np.asarray(np.load("record.npy"), dtype=np.float64)
    if xb0.shape != (n, d):
        xb0 = xb0.reshape(n, d)
    xb0 = _norm(xb0)
    gval = _max_cos(xb0)
    gbest = xb0.copy()
    cur = xb0.copy()
    cur_val = gval
    iu = np.triu_indices(n, 1)

    trial = 0
    # temperature for sideways moves: allow ~1e-5 degradations early
    while time.time() < HOP_END and time.time() < HARD - 40:
        trial += 1
        frac = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        Tsa = 8e-06 * (1.0 - frac) + 5e-07
        base = cur.copy()
        # contact frequency with loose threshold to find hubs
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        cf = G[iu]
        w = int(np.argmax(cf))
        wi, wj = int(iu[0][w]), int(iu[1][w])
        thr = mcur - 0.004
        freq = np.zeros(n, dtype=np.int64)
        mk = cf >= thr
        if np.any(mk):
            np.add.at(freq, iu[0][mk], 1)
            np.add.at(freq, iu[1][mk], 1)
        permax = np.max(G, axis=1)
        order_low = np.argsort(permax, kind="stable")  # rattlers first
        order_hi = np.argsort(-freq, kind="stable")
        r = rng.random()
        if trial % 25 == 0:
            k = 8 + int(rng.integers(0, 4))
        elif trial % 6 == 0:
            k = 4 + int(rng.integers(0, 3))
        else:
            k = 2 + int(rng.integers(0, 2))
        victims = set()
        victims.add(wi if rng.random() < 0.5 else wj)
        # add hubs
        nh = min(k - 1, 6)
        for j in range(nh):
            if rng.random() < 0.8 and j < n:
                victims.add(int(order_hi[j]))
        # add rattlers / low-contact
        while len(victims) < k:
            # mostly lowest permax
            if rng.random() < 0.7:
                idx = int(order_low[int(rng.integers(0, min(n, 12)))])
            else:
                idx = int(rng.integers(0, n))
            victims.add(idx)
        victims = list(victims)
        y = base.copy()
        # relocate 1-2 victims into voids, noise on rest
        nvoid = 1 if k <= 3 else 2
        # choose void victims as lowest permax among victims
        vm = sorted(victims, key=lambda a: permax[a])
        void_set = set(vm[:nvoid])
        # fixed set for hole search = all except void points
        mask_fixed = np.ones(n, dtype=bool)
        for v in void_set:
            mask_fixed[v] = False
        Xf = y[mask_fixed]
        for v in victims:
            if v in void_set:
                h = _hole_for_fixed(Xf, rng, ns=2500)
                # small jitter around hole
                h = h + 0.01 * rng.standard_normal(d)
                h /= (np.linalg.norm(h) + 1e-300)
                y[v] = h
            else:
                sig = 0.06 + 0.10 * rng.random()
                t = rng.standard_normal(d)
                t -= np.dot(t, y[v]) * y[v]
                t /= (np.linalg.norm(t) + 1e-300)
                y[v] = y[v] + sig * t
                y[v] /= (np.linalg.norm(y[v]) + 1e-300)
        # reconverge: cheap softmax anneal (no LP)
        if k >= 6:
            steps = 220
        elif k >= 4:
            steps = 170
        else:
            steps = 130
        y = _smooth(y, steps=steps, T0=0.006, T1=0.0006, lr0=0.12, lr1=0.02)
        # extra fine phase
        y = _smooth(y, steps=60, T0=0.0006, T1=0.00025, lr0=0.02, lr1=0.008)
        v = _max_cos(y)
        if v < gval:
            gval = v; gbest = y.copy()
            cur = y.copy(); cur_val = v
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Tsa, 1e-12)):
            cur = y.copy(); cur_val = v
        # occasional restart wandering to best
        if trial % 40 == 0 and rng.random() < 0.5:
            cur = gbest.copy(); cur_val = gval

    # final polish: short smooth then strictly bounded LP
    if time.time() < HARD - 5:
        gbest = _smooth(gbest, steps=80, T0=0.0005, T1=0.0002, lr0=0.015, lr1=0.006)
        gval = _max_cos(gbest)
    if time.time() < HARD - 6:
        budget = max(5.0, HARD - time.time() - 3.0)
        budget = min(budget, 35.0)
        gbest, gval = _lp_polish(gbest, budget=budget, deadline=HARD)
    gbest = _norm(gbest)
    return gbest

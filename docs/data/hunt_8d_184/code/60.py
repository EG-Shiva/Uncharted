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

def _anneal(x, steps, T0, T1, lr0, lr1):
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
        # normalize step to avoid blowup
        x = x - lr * Dt
        # fast renorm
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

def _greedy_voids(Xfixed, k, rng, ns=2048, jitter=0.01):
    d = Xfixed.shape[1]
    out = np.empty((k, d))
    F = Xfixed.copy()
    for i in range(k):
        Y = rng.standard_normal(size=(ns, d))
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
        S = Y @ F.T
        m = S.max(axis=1)
        idx = int(np.argmin(m))
        y = Y[idx]
        if jitter > 0:
            y = y + jitter * rng.standard_normal(size=(d,))
            y /= (np.linalg.norm(y) + 1e-300)
        out[i] = y
        F = np.vstack([F, y[None, :]])
    return out

def _lp_polish(x, seconds, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.0e-3
    t_start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t_start < seconds and time.time() < deadline:
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 4e-7:
                # try a couple more shrinks then stop
                pass
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(7)
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
    cur = gbest.copy()
    cur_val = gval
    iu = np.triu_indices(n, 1)
    trial = 0
    while time.time() < HOP_END:
        trial += 1
        if time.time() > HOP_END - 0.5:
            break
        frac = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        # sideways tolerance decays
        Tsa = 3e-4 * (1.0 - frac) + 2e-6
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        cf = G[iu]
        # frequencies for hub / loose detection
        thr = mcur - 0.004
        freq = np.zeros(n, dtype=np.int64)
        mk = cf >= thr
        if np.any(mk):
            np.add.at(freq, iu[0][mk], 1)
            np.add.at(freq, iu[1][mk], 1)
        permax = np.max(G, axis=1)
        order_hi = np.argsort(-freq, kind="stable")
        order_lo = np.argsort(permax, kind="stable")
        worst = int(np.argmax(cf))
        wi, wj = int(iu[0][worst]), int(iu[1][worst])
        # block size: large moves to change topology
        r = rng.random()
        if trial % 20 == 0:
            k = 14 + int(rng.integers(0, 5))  # 14-18
        elif trial % 5 == 0:
            k = 10 + int(rng.integers(0, 4))  # 10-13
        else:
            k = 6 + int(rng.integers(0, 5))   # 6-10
        victims = set()
        victims.add(wi if rng.random() < 0.5 else wj)
        if rng.random() < 0.6:
            victims.add(wi); victims.add(wj)
        # hubs
        nh = 2 + int(rng.integers(0, 3))
        for t in range(nh):
            victims.add(int(order_hi[int(rng.integers(0, min(20, n)))]))
        # loose / rattlers
        nl = 2 + int(rng.integers(0, 3))
        for t in range(nl):
            victims.add(int(order_lo[int(rng.integers(0, min(30, n)))]))
        while len(victims) < k:
            u = rng.random()
            if u < 0.4:
                victims.add(int(order_hi[int(rng.integers(0, min(40, n)))]))
            elif u < 0.7:
                victims.add(int(order_lo[int(rng.integers(0, min(60, n)))]))
            else:
                victims.add(int(rng.integers(0, n)))
        victims = list(victims)[:k]
        x = cur.copy()
        # small global tangential loosen (helps leave jam without LP)
        if rng.random() < 0.6:
            sig = 0.004 + 0.012 * rng.random()
            x = x + sig * rng.standard_normal(size=x.shape)
            x = _norm(x)
        # greedy void reinsertion for victims
        mask = np.ones(n, dtype=bool); mask[victims] = False
        Xfixed = x[mask]
        try:
            repl = _greedy_voids(Xfixed, len(victims), rng, ns=2048, jitter=0.015)
        except Exception:
            continue
        for vv, y in zip(victims, repl):
            x[vv] = y
        # anneal schedules randomized
        T0 = 0.008 * (0.7 + 0.9 * rng.random())
        T1 = 0.0012 * (0.7 + 0.9 * rng.random())
        lr0 = 0.06 * (0.7 + 0.7 * rng.random())
        lr1 = 0.010 * (0.7 + 0.7 * rng.random())
        steps = 160 + int(rng.integers(0, 140))
        try:
            x = _anneal(x, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        v = _max_cos(x)
        if v < gval - 1e-12:
            gval = v; gbest = x.copy()
            cur = x.copy(); cur_val = v
        elif v < cur_val + Tsa:
            cur = x.copy(); cur_val = v
            # keep global best unchanged but walk sideways
            # occasional restart to global best to avoid drift
            if rng.random() < 0.08:
                cur = gbest.copy(); cur_val = gval
        else:
            # occasional restart
            if trial % 40 == 0 and rng.random() < 0.5:
                cur = gbest.copy(); cur_val = gval
        # early exit if time low
        if time.time() > HOP_END:
            break
    # final polish of global best with single time-boxed LP
    remain = DEADLINE - time.time()
    if remain > 4:
        budget = min(remain - 1.5, 34.0)
        try:
            xp, vp = _lp_polish(gbest, seconds=budget, deadline=DEADLINE - 0.5)
            if vp < gval:
                gbest, gval = xp, vp
        except Exception:
            pass
    gbest = _norm(gbest)
    return gbest

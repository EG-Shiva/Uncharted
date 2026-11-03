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
    # annealed softmax repulsion, tangential
    x = _norm(x.copy())
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

def _hole_against(fixed, rng, n_sample=3000):
    d = fixed.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
    if len(fixed) == 0:
        return Y[0]
    S = Y @ fixed.T
    m = S.max(axis=1)
    return Y[int(np.argmin(m))]

def _lp_polish(x, seconds, t_start=None, deadline=None):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    if t_start is None:
        t_start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    pcap = 1800
    while time.time() - t_start < seconds:
        if deadline is not None and time.time() > deadline:
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
            break
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
    DEAD = t0 + 172.0
    rng = np.random.default_rng(12345)
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
    x_cur = gbest.copy()
    cur = gval
    iu = np.triu_indices(n, 1)

    trial = 0
    # reserve ~28s for final LP
    while time.time() < DEAD - 30.0:
        trial += 1
        if time.time() > DEAD - 32:
            break
        # base: mostly global best, sometimes wanderer
        if rng.random() < 0.82 or trial < 3:
            base = gbest; base_val = gval
        else:
            base = x_cur; base_val = cur
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        cf = G[iu]
        w = int(np.argmax(cf))
        wi, wj = int(iu[0][w]), int(iu[1][w])
        margin = 0.004 + 0.004 * rng.random()
        mask = cf >= (mcur - margin)
        freq = np.zeros(n, dtype=np.int64)
        if np.any(mask):
            np.add.at(freq, iu[0][mask], 1)
            np.add.at(freq, iu[1][mask], 1)
        per = G.max(axis=1)
        order_hi = np.argsort(-freq, kind="stable")
        order_lo = np.argsort(per, kind="stable")  # small per-max = isolated/rattler
        # choose kick size
        r = rng.random()
        if trial % 31 == 0:
            k = 9 + int(rng.integers(0, 4))
        elif trial % 8 == 0:
            k = 5 + int(rng.integers(0, 4))
        else:
            k = 2 + int(rng.integers(0, 3))
        victims = set()
        victims.add(wi if rng.random() < 0.5 else wj)
        # add overloaded hubs
        nh = 0
        for idx in order_hi:
            if len(victims) >= k:
                break
            if int(idx) in victims:
                continue
            if nh < 2 or rng.random() < 0.5:
                # skip with some prob to diversify
                victims.add(int(idx))
                nh += 1
                if nh >= max(1, k // 2):
                    # allow random fill for rest
                    pass
        # ensure at least one rattler relocated
        n_hole = 1 + (1 if rng.random() < 0.4 else 0)
        hole_cands = [int(v) for v in order_lo[:12] if int(v) not in victims]
        rng.shuffle(hole_cands)
        for h in hole_cands[:n_hole]:
            victims.add(h)
        while len(victims) < k:
            v = int(rng.integers(0, n))
            victims.add(v)
        victims = list(victims)
        hole_set = set()
        # pick hole victims: most isolated among victims + maybe random
        # rank victims by per (small first)
        vper = [(float(per[v]), v) for v in victims]
        vper.sort()
        for _, v in vper[:n_hole]:
            hole_set.add(v)

        x = base.copy()
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        fixed = x[keep_mask]
        # relocate hole victims to voids (sequential to avoid collision)
        cur_fixed = fixed.copy()
        for v in victims:
            if v in hole_set:
                h = _hole_against(cur_fixed, rng, n_sample=3000)
                x[v] = h
                # append for next hole so holes repel each other
                cur_fixed = np.vstack([cur_fixed, h[None, :]])
            else:
                # tangential shake
                if trial % 31 == 0:
                    sig = 0.25 + 0.25 * rng.random()
                elif trial % 8 == 0:
                    sig = 0.12 + 0.15 * rng.random()
                else:
                    sig = 0.03 + 0.10 * rng.random()
                g = rng.standard_normal(size=(d,))
                g = g - np.dot(g, x[v]) * x[v]
                gn = np.linalg.norm(g) + 1e-300
                g /= gn
                # random magnitude
                step = sig * (0.5 + rng.random())
                x[v] = x[v] + step * g
                x[v] /= np.linalg.norm(x[v]) + 1e-300
        # anneal params randomized
        T0 = 0.015 + 0.015 * rng.random()
        T1 = 0.0015 + 0.0025 * rng.random()
        lr0 = 0.03 + 0.05 * rng.random()
        lr1 = 0.003 + 0.005 * rng.random()
        steps = 220 + int(rng.integers(0, 140))
        # fewer steps for large kicks early, more for small
        x = _smooth(x, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        v = _max_cos(x)
        # wandering acceptance: accept if better than cur or with small prob if close
        if v < cur:
            x_cur = x.copy(); cur = v
        elif rng.random() < 0.08 and v < cur + 0.004:
            x_cur = x.copy(); cur = v
        if v < gval:
            gval = v; gbest = x.copy()
            # quick opportunistic tiny LP? skip to save time, rely on final
            if time.time() > DEAD - 32:
                break
        # early stop if time low
        if trial > 200:
            break

    # final polish on global best with remaining time
    remain = DEAD - time.time()
    if remain > 8:
        gbest = _lp_polish(gbest, seconds=min(remain - 1.0, 28.0), t_start=time.time(), deadline=DEAD - 0.5)
    gbest = _norm(gbest)
    return gbest

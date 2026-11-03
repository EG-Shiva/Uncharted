import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth(x, steps, T0, T1, lr0, lr1):
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
        x = _norm(x - lr * Dt)
    return x

def _greedy(x, steps, margin0, margin1, lr):
    x = _norm(x.copy())
    n = x.shape[0]
    for s in range(steps):
        f = s / max(steps - 1, 1)
        mg = margin0 + (margin1 - margin0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > (m - mg)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        div = np.maximum(deg, 1.0)
        x = _norm(x - lr * gt / div)
    return x

def _hole_refine(y, Xf, steps=60, a0=80.0, a1=400.0, lr0=0.08, lr1=0.01):
    y = y / (np.linalg.norm(y) + 1e-300)
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        al = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = Xf @ y
        mx = float(np.max(s))
        w = np.exp(al * (s - mx))
        sw = float(np.sum(w)) + 1e-300
        g = w @ Xf
        gt = g - float(g @ y) * y
        y = y - (lr / sw) * gt
        nrm = np.linalg.norm(y)
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _sample_hole(Xf, rng, ns=4000):
    d = Xf.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    S = Y @ Xf.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx]

def _lp_polish_capped(x, t_start, deadline, pcap=1200):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 8e-4
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline - 0.5 and radius > 1e-13:
        G = x @ x.T
        cf = G[iu0]
        m = float(np.max(cf))
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    HOP_END = t0 + 138.0
    rng = np.random.RandomState(0)
    n, d = 184, 8
    xb = None
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                if xb is None or _max_cos(c) < _max_cos(xb):
                    xb = c
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.standard_normal(size=(n, d)))
    gbest = xb.copy()
    gval = _max_cos(gbest)
    cur = gbest.copy()
    cval = gval
    iu = np.triu_indices(n, 1)

    trial = 0
    TEMP = 0.0025
    while time.time() < HOP_END:
        trial += 1
        if time.time() > DEADLINE - 30:
            break
        # choose base: mostly best, sometimes wandering cur
        if rng.rand() < 0.72:
            base = gbest; base_val = gval
        else:
            base = cur; base_val = cval
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        cf = G[iu]
        # frequencies of near-max pairs
        thr_f = mcur - 0.005
        freq = np.zeros(n, dtype=np.int64)
        mask = cf >= thr_f
        if np.any(mask):
            np.add.at(freq, iu[0][mask], 1)
            np.add.at(freq, iu[1][mask], 1)
        per = np.max(G, axis=1)
        # rattlers / low contact: far from max
        ratt = np.where(per < mcur - 0.002)[0]
        order_hi = np.argsort(-freq, kind="stable")
        # victim count schedule
        if trial % 25 == 0:
            k = 7 + int(rng.randint(0, 3))
        elif trial % 6 == 0:
            k = 4 + int(rng.randint(0, 3))
        else:
            k = 2 + int(rng.randint(0, 2))
        k = min(k, n - 8)
        victims = []
        # rattlers first
        rl = list(ratt)
        rng.shuffle(rl)
        for v in rl:
            if len(victims) >= k:
                break
            victims.append(int(v))
        # worst pair endpoint
        w = int(np.argmax(cf))
        wi, wj = int(iu[0][w]), int(iu[1][w])
        cand_w = wi if rng.rand() < 0.5 else wj
        if len(victims) < k and cand_w not in victims:
            victims.append(cand_w)
        # overloaded hubs
        for v in order_hi:
            if len(victims) >= k:
                break
            v = int(v)
            if v not in victims:
                # randomize slightly: skip with small prob to diversify
                if rng.rand() < 0.85:
                    victims.append(v)
        # fill randomly if needed
        while len(victims) < k:
            v = int(rng.randint(0, n))
            if v not in victims:
                victims.append(v)
        vset = set(victims)
        keep = np.array([i for i in range(n) if i not in vset], dtype=int)
        xn = base.copy()
        # sequential hole insertion
        placed = []
        # shake non-victims slightly to break symmetry
        shake = 0.004 if (trial % 3 == 0) else 0.0015
        noise = rng.standard_normal(size=(n, d))
        # tangential projection w.r.t base
        dotn = (noise * base).sum(axis=1, keepdims=True)
        noise = noise - dotn * base
        xn = xn + shake * noise
        xn = _norm(xn)
        for v in victims:
            if time.time() > HOP_END:
                break
            if len(placed) == 0:
                Xf = xn[keep]
            else:
                Xf = np.vstack([xn[keep], np.vstack(placed)])
            y0 = _sample_hole(Xf, rng, ns=4000)
            y0 = _hole_refine(y0, Xf, steps=50)
            placed.append(y0)
            xn[v] = y0
        xn = _norm(xn)
        # reconverge: fast anneal
        # vary schedule slightly per trial
        if k >= 6:
            steps = 300
        else:
            steps = 240
        T0 = 0.022 if (trial % 2 == 0) else 0.030
        T1 = 0.0018
        xn = _smooth(xn, steps=steps, T0=T0, T1=T1, lr0=0.05, lr1=0.006)
        # short greedy tightening on top pairs
        xn = _greedy(xn, steps=60, margin0=0.006, margin1=0.0006, lr=0.008)
        vnew = _max_cos(xn)
        if vnew < gval - 1e-12:
            gbest = xn.copy(); gval = vnew
            cur = xn.copy(); cval = vnew
        else:
            # wandering acceptance (SA)
            if vnew < cval or rng.rand() < np.exp(-(vnew - cval) / TEMP):
                cur = xn.copy(); cval = vnew
            # mild cooling of TEMP
            TEMP *= 0.9995
            if TEMP < 0.0008:
                TEMP = 0.0008
        # occasional reheat of wandering to best if too bad
        if cval > gval + 0.015:
            cur = gbest.copy(); cval = gval
            TEMP = 0.0025

    # final tightening of global best: smooth + greedy, then capped LP
    try:
        gbest = _smooth(gbest, steps=180, T0=0.008, T1=0.0008, lr0=0.02, lr1=0.004)
        gbest = _greedy(gbest, steps=80, margin0=0.004, margin1=0.0004, lr=0.006)
        nv = _max_cos(gbest)
        if nv < gval:
            gval = nv
        if time.time() < DEADLINE - 4:
            gbest2 = _lp_polish_capped(gbest, time.time(), DEADLINE - 1.0, pcap=1200)
            nv2 = _max_cos(gbest2)
            if nv2 < gval:
                gbest = gbest2; gval = nv2
    except Exception:
        pass
    gbest = _norm(gbest)
    return gbest

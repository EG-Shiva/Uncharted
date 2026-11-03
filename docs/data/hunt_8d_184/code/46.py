import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nrm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _anneal(x, steps, a0, a1, lr0, lr1):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        al = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        W = np.exp(al * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        x = x - lr * (Gt / S)
        x = _nrm(x)
    return x

def _greedy(x, steps, m0, m1, lr):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        mg = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > m - mg).astype(np.float64)
        dg = M.sum(axis=1, keepdims=True)
        Gr = M @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        x = x - lr * (Gt / np.maximum(dg, 1.0))
        x = _nrm(x)
    return x

def _void(xfix, rng, ns=2000):
    d = xfix.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y = _nrm(Y)
    S = Y @ xfix.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i].copy(), float(m[i])

def _lp(x, deadline, budget):
    t0 = time.time()
    x = _nrm(x.copy())
    n, d = x.shape
    best = _mc(x)
    xb = x.copy()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    rad = 8e-4
    while True:
        now = time.time()
        if now > deadline - 3.0:
            break
        if now - t0 > budget:
            break
        if rad < 1e-13:
            break
        G = x @ x.T
        cf = G[iu]
        m = float(cf.max())
        thr = m - max(20.0 * rad, 1e-9)
        sel = cf > thr
        I = iu[0][sel]; J = iu[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            rad *= 0.5
            continue
        if p > 1400:
            k = np.argpartition(-cs, 1400)[:1400]
            I, J, cs = I[k], J[k], cs[k]
            p = 1400
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = max(1.0, deadline - time.time() - 1.0)
        try:
            res = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                          bounds=[(-rad, rad)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            rad *= 0.5
            continue
        if res.status != 0:
            rad *= 0.5
            continue
        try:
            y = _nrm(x + res.x[:-1].reshape(n, d))
        except Exception:
            rad *= 0.5
            continue
        v = _mc(y)
        if v < best and np.isfinite(v):
            best = v; xb = y.copy(); x = y
            rad = min(rad * 1.5, 0.02)
        else:
            rad *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    HOP_END = t0 + 100.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    try:
        x0 = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        x0 = np.asarray(np.load("record.npy"), dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64).reshape(n, d)
    x0 = _nrm(x0)
    gbest = x0.copy()
    gval = _mc(gbest)
    cur = x0.copy()
    cval = gval
    iu = np.triu_indices(n, 1)
    trial = 0
    while time.time() < HOP_END and time.time() < HARD - 45:
        trial += 1
        frac = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        Tsa = 6e-06 * (1.0 - frac) + 3e-07
        xt = cur.copy()
        G = xt @ xt.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        cf = G[iu]
        # contact freq for hubs
        thrC = mcur - 0.004
        freq = np.zeros(n, dtype=np.int64)
        mk = cf >= thrC
        if np.any(mk):
            np.add.at(freq, iu[0][mk], 1)
            np.add.at(freq, iu[1][mk], 1)
        permax = np.max(G, axis=1)
        ord_low = np.argsort(permax, kind="stable")
        ord_hi = np.argsort(-freq, kind="stable")
        mode = trial % 6
        if mode == 1:
            # global small shake
            xt = xt + rng.standard_normal(size=xt.shape) * 0.012
            xt = _nrm(xt)
        elif mode == 2:
            # shake hubs
            hubs = ord_hi[:8]
            xt[hubs] += rng.standard_normal(size=(8, d)) * 0.085
            xt[hubs] = _nrm(xt[hubs])
        elif mode == 3:
            # relocate 2 lowest into voids + tiny noise
            lows = ord_low[:2].tolist()
            keep = [i for i in range(n) if i not in set(lows)]
            Xf = xt[keep].copy()
            for li in lows:
                y0, _ = _void(Xf, rng, ns=2000)
                xt[li] = y0
                Xf = np.vstack([Xf, y0[None, :]])
            xt[keep] += rng.standard_normal(size=(len(keep), d)) * 0.003
            xt = _nrm(xt)
        elif mode == 4:
            # neighbourhood of worst pair
            w = int(np.argmax(cf))
            wi, wj = int(iu[0][w]), int(iu[1][w])
            # nearest to wi
            nn = np.argsort(-G[wi])[:10]
            xt[nn] += rng.standard_normal(size=(10, d)) * 0.05
            xt[nn] = _nrm(xt[nn])
            xt[wj] += rng.standard_normal(size=(1, d)) * 0.05
            xt = _nrm(xt)
        elif mode == 5:
            idx = rng.choice(n, size=8, replace=False)
            xt[idx] += rng.standard_normal(size=(8, d)) * 0.10
            xt[idx] = _nrm(xt[idx])
        else:
            # worst 5 shake
            worst = np.argsort(-permax)[:5]
            xt[worst] += rng.standard_normal(size=(5, d)) * 0.06
            xt[worst] = _nrm(xt[worst])
        # reconverge cheap
        if mode == 3:
            xt = _anneal(xt, steps=170, a0=35.0, a1=320.0, lr0=0.030, lr1=0.0025)
            xt = _greedy(xt, steps=60, m0=0.007, m1=0.0007, lr=0.007)
        else:
            xt = _anneal(xt, steps=150, a0=40.0, a1=300.0, lr0=0.028, lr1=0.0025)
            xt = _greedy(xt, steps=50, m0=0.006, m1=0.0007, lr=0.007)
        v = _mc(xt)
        if not np.isfinite(v):
            continue
        if v < gval:
            gval = float(v); gbest = xt.copy()
        # sideways SA move for current
        if v < cval or rng.random() < np.exp(-(v - cval) / max(Tsa, 1e-12)):
            cur = xt.copy(); cval = float(v)
        # early exit if approaching hard limit
        if time.time() > HARD - 45:
            break
    # final polish time-boxed
    remain = HARD - time.time() - 2.0
    if remain > 6 and np.all(np.isfinite(gbest)):
        budget = min(remain, 35.0)
        try:
            gbest2, v2 = _lp(gbest, HARD, budget)
            if np.all(np.isfinite(gbest2)) and v2 < gval:
                gbest = gbest2; gval = v2
        except Exception:
            pass
    gbest = _nrm(np.asarray(gbest, dtype=np.float64).reshape(n, d))
    if not np.all(np.isfinite(gbest)):
        gbest = _nrm(x0.copy())
    return gbest

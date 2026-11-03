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

def _smooth(x, steps, a0, a1, lr0, lr1):
    n, d = x.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = np.max(G)
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / S
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy(x, steps, m0, m1, lr):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        M = (G > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / np.maximum(deg, 1.0)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole(xfix, rng, ns=3000):
    d = xfix.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    S = Y @ xfix.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i]

def _refine_one(xfix, y, steps=60, alpha=200.0, lr=0.02):
    for _ in range(steps):
        s = xfix @ y
        mx = np.max(s)
        w = np.exp(alpha * (s - mx))
        sw = np.sum(w) + 1e-30
        g = w @ xfix
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        y /= np.linalg.norm(y) + 1e-300
    return y

def _degrees(x, m, gap):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    M = G > (m - gap)
    return M.sum(axis=1), G

def _lp_polish(x, seconds, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.2e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t0 < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
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
        if p > 2200:
            idx = np.argpartition(-cs, 2200)[:2200]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2200
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-7:
                break
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
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
    gbest = np.ascontiguousarray(gbest)

    trial = 0
    # hopping phase, leave ~35s for final LP
    while time.time() < DEADLINE - 38.0 and time.time() - t0 < 135.0:
        trial += 1
        xt = gbest.copy()
        G0 = xt @ xt.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        deg, _ = _degrees(xt, m0, 5e-4)
        order_hi = np.argsort(-deg)
        order_lo = np.argsort(deg)
        mode = trial % 6
        if mode == 0:
            # relocate 1-2 lowest-degree (rattlers) to voids
            k = 1 if (trial % 2 == 0) else 2
            victims = order_lo[:k].tolist()
            keep = [i for i in range(n) if i not in set(victims)]
            xfix = xt[keep]
            for vi in victims:
                y = _hole(xfix, rng, ns=3000)
                y = _refine_one(xfix, y, steps=60)
                xt[vi] = y
                xfix = np.vstack([xfix, y[None, :]])
            # tiny jitter rest
            xt += rng.standard_normal(size=xt.shape) * 0.002
            xt = _norm(xt)
            xt = _smooth(xt, steps=300, a0=40.0, a1=350.0, lr0=0.030, lr1=0.003)
            xt = _greedy(xt, steps=140, m0=0.008, m1=0.0007, lr=0.007)
        elif mode == 1:
            # shake overloaded hubs
            k = 5 + int(rng.integers(0, 4))
            victims = order_hi[:k]
            xt[victims] += rng.standard_normal(size=(k, d)) * 0.06
            xt[victims] = _norm(xt[victims])
            xt = _smooth(xt, steps=260, a0=50.0, a1=300.0, lr0=0.026, lr1=0.0025)
            xt = _greedy(xt, steps=110, m0=0.007, m1=0.0008, lr=0.007)
        elif mode == 2:
            # shake worst per-max points
            pm = np.max(G0, axis=1)
            worst = np.argsort(-pm)[:4]
            xt[worst] += rng.standard_normal(size=(4, d)) * 0.075
            xt[worst] = _norm(xt[worst])
            xt = _smooth(xt, steps=260, a0=45.0, a1=320.0, lr0=0.028, lr1=0.0025)
            xt = _greedy(xt, steps=110, m0=0.007, m1=0.0008, lr=0.007)
        elif mode == 3:
            # block shake around random center (topology change in dense region)
            c = int(rng.integers(0, n))
            row = G0[c]
            near = np.argsort(-row)[:12]
            xt[near] += rng.standard_normal(size=(12, d)) * 0.05
            xt[near] = _norm(xt[near])
            xt = _smooth(xt, steps=260, a0=50.0, a1=300.0, lr0=0.026, lr1=0.0025)
            xt = _greedy(xt, steps=110, m0=0.007, m1=0.0008, lr=0.007)
        elif mode == 4:
            # global small jitter + anneal
            xt = xt + rng.standard_normal(size=xt.shape) * 0.009
            xt = _norm(xt)
            xt = _smooth(xt, steps=280, a0=45.0, a1=340.0, lr0=0.028, lr1=0.0028)
            xt = _greedy(xt, steps=120, m0=0.007, m1=0.0007, lr=0.007)
        else:
            # relocate one hub into void + jitter
            hub = int(order_hi[int(rng.integers(0, 8))])
            keep = [i for i in range(n) if i != hub]
            xfix = xt[keep]
            y = _hole(xfix, rng, ns=4000)
            y = _refine_one(xfix, y, steps=80)
            xt[hub] = y
            idx = rng.choice(n, size=8, replace=False)
            xt[idx] += rng.standard_normal(size=(8, d)) * 0.02
            xt = _norm(xt)
            xt = _smooth(xt, steps=300, a0=40.0, a1=350.0, lr0=0.030, lr1=0.003)
            xt = _greedy(xt, steps=140, m0=0.008, m1=0.0007, lr=0.007)
        v = _max_cos(xt)
        if v < gval:
            gval = float(v)
            gbest = xt.copy()
        if time.time() > DEADLINE - 40:
            break

    remain = DEADLINE - time.time() - 1.0
    if remain > 10:
        gbest = _lp_polish(gbest, min(remain, 32.0), DEADLINE - 0.5)
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(np.asarray(np.load("best.npy"), dtype=np.float64))
    return gbest

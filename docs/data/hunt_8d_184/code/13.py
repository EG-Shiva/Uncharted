import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

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
        m = np.max(G)
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / S
        x = _norm(x)
    return x

def _greedy(x, steps, m0, m1, lr):
    n, d = x.shape
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > m - margin).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        div = np.maximum(deg, 1.0)
        x = x - lr * gt / div
        x = _norm(x)
    return x

def _hole(x_fixed, rng, ns=12000):
    d = x_fixed.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
    S = Y @ x_fixed.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i], float(m[i])

def _lp(x, seconds, t0, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _mc(x)
    xb = x.copy()
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and time.time() < deadline and rad > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * rad, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            rad *= 0.5
            continue
        if p > 3500:
            idx = np.argpartition(-cs, 3500)[:3500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 3500
        nv = n * d + 1
        ar = np.arange(d)
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        rows = np.concatenate([np.repeat(np.arange(p), d),
                               np.repeat(np.arange(p), d),
                               np.arange(p)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            rad *= 0.5
            continue
        if res.status != 0:
            rad *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            rad = min(rad * 1.5, 0.03)
        else:
            rad *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEAD = t0 + 174.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64).reshape(184, 8)
    xb = _norm(xb)
    try:
        xr = np.load("record.npy")
        xr = _norm(np.asarray(xr, dtype=np.float64).reshape(184, 8))
        if _mc(xr) < _mc(xb):
            xb = xr.copy()
    except Exception:
        pass
    gbest = xb.copy(); gval = _mc(gbest)
    cur = gbest.copy(); cval = gval
    # quick initial smooth on loaded point to squeeze
    # hopping phase: reserve 38s for final polish
    LP_RESERVE = 38.0
    trial = 0
    while time.time() < DEAD - LP_RESERVE:
        trial += 1
        # per-point diagnostics of cur
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        per = G.max(axis=1)
        order_worst = np.argsort(-per)
        deg = (G > m - 0.004).sum(axis=1)
        order_hub = np.argsort(-deg)
        y = cur.copy()
        # choose perturbation strength (anneal slightly)
        # alternate modes
        mode = trial % 4
        if mode == 0:
            # relocate 1-2 rattlers (lowest per) to holes
            order_best = np.argsort(per)  # lowest first
            krel = 1 if rng.random() < 0.7 else 2
            rel_idx = order_best[:krel].copy()
            fixed = np.delete(y, rel_idx, axis=0)
            for ri in rel_idx:
                h, _ = _hole(fixed, rng, ns=12000)
                # small tangential noise
                h = h + 0.01 * rng.standard_normal(8)
                h /= np.linalg.norm(h) + 1e-300
                y[ri] = h
                fixed = np.vstack([fixed, h[None, :]])
            # also shake few hubs lightly
            k2 = int(rng.integers(2, 6))
            for k in range(k2):
                idx = int(order_hub[int(rng.integers(0, 25))])
                v = rng.standard_normal(8)
                v -= np.dot(v, y[idx]) * y[idx]
                y[idx] = y[idx] + rng.uniform(0.05, 0.15) * v / (np.linalg.norm(v) + 1e-300)
                y[idx] /= np.linalg.norm(y[idx]) + 1e-300
        elif mode == 1:
            # shake worst endpoints strongly
            k = int(rng.integers(4, 10))
            pool = np.concatenate([order_worst[:25], order_hub[:25]])
            sels = rng.choice(pool, size=k, replace=False)
            sig = rng.uniform(0.08, 0.28)
            for idx in sels:
                v = rng.standard_normal(8)
                v -= np.dot(v, y[int(idx)]) * y[int(idx)]
                nrm = np.linalg.norm(v) + 1e-300
                y[int(idx)] = y[int(idx)] + sig * v / nrm
                y[int(idx)] /= np.linalg.norm(y[int(idx)]) + 1e-300
        elif mode == 2:
            # global light noise + medium shake of hubs
            sig = rng.uniform(0.02, 0.08)
            v = rng.standard_normal(size=y.shape)
            v -= np.sum(v * y, axis=1, keepdims=True) * y
            y = y + sig * v
            y = _norm(y)
            k = int(rng.integers(2, 5))
            for kk in range(k):
                idx = int(order_hub[int(rng.integers(0, 15))])
                vv = rng.standard_normal(8)
                vv -= np.dot(vv, y[idx]) * y[idx]
                y[idx] = y[idx] + rng.uniform(0.1, 0.25) * vv / (np.linalg.norm(vv) + 1e-300)
                y[idx] /= np.linalg.norm(y[idx]) + 1e-300
        else:
            # shake random subset medium
            k = int(rng.integers(6, 14))
            sels = rng.choice(184, size=k, replace=False)
            sig = rng.uniform(0.05, 0.2)
            for idx in sels:
                v = rng.standard_normal(8)
                v -= np.dot(v, y[int(idx)]) * y[int(idx)]
                y[int(idx)] = y[int(idx)] + sig * v / (np.linalg.norm(v) + 1e-300)
                y[int(idx)] /= np.linalg.norm(y[int(idx)]) + 1e-300
        # anneal refine: fewer steps for speed
        y = _smooth(y, steps=320, a0=35.0, a1=320.0, lr0=0.03, lr1=0.003)
        y = _greedy(y, steps=90, m0=0.006, m1=0.0008, lr=0.008)
        v = _mc(y)
        if v < gval:
            gval = v; gbest = y.copy()
        # basin-hopping acceptance for cur (temperature)
        T = 0.0008 * (1.0 - min(1.0, (time.time() - t0) / 130.0)) + 0.00015
        if v < cval or rng.random() < np.exp(-(v - cval) / T):
            cur = y.copy(); cval = v
        else:
            # occasionally restart from gbest
            if rng.random() < 0.12:
                cur = gbest.copy(); cval = gval
        if time.time() > DEAD - LP_RESERVE - 2:
            break
    # polish global best
    gbest = _smooth(gbest, steps=250, a0=50.0, a1=400.0, lr0=0.02, lr1=0.002)
    gbest = _greedy(gbest, steps=120, m0=0.005, m1=0.0005, lr=0.007)
    gval = _mc(gbest)
    # final LP with remaining time
    rem = DEAD - time.time() - 1.0
    if rem > 8:
        gbest = _lp(gbest, seconds=rem, t0=time.time(), deadline=DEAD - 0.5)
    gbest = _norm(gbest)
    return gbest

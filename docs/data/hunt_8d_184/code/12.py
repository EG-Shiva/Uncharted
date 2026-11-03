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
        M = (G > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        div = np.maximum(deg, 1.0)
        x = x - lr * gt / div
        x = _norm(x)
    return x

def _hole_refine(xf, y, steps=120, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = xf @ y
        m = np.max(s)
        w = np.exp(alpha * (s - m))
        sw = np.sum(w) + 1e-30
        g = w @ xf
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        nrm = np.linalg.norm(y)
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _best_hole(xf, rng, ns=4000):
    d = xf.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
    S = Y @ xf.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i]

def _lp_polish(x, seconds, t0, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _mc(x)
    xb = x.copy()
    radius = 2e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and time.time() < deadline and radius > 1e-14:
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
        if p > 4000:
            idx = np.argpartition(-cs, 4000)[:4000]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 4000
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
            if radius < 1e-7 and p > 1500:
                break
    return xb

def solve():
    t0 = time.time()
    DEAD = t0 + 175.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = xb
    xb = _norm(np.asarray(xb, dtype=np.float64).reshape(184, 8))
    xr = _norm(np.asarray(xr, dtype=np.float64).reshape(184, 8))
    gbest = xb if _mc(xb) <= _mc(xr) else xr.copy()
    gval = _mc(gbest)
    n, d = 184, 8

    trial = 0
    # hopping phase: reserve ~32s for final LP
    while time.time() < DEAD - 32 and trial < 400:
        trial += 1
        if time.time() > DEAD - 32:
            break
        x = gbest.copy()
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        per = G.max(axis=1)
        # near-max degree for overload detection
        C = (G > m - 0.004)
        deg = C.sum(axis=1)
        worst = np.argsort(-per, kind="stable")
        loose = np.argsort(deg, kind="stable")
        # choose kick sizes
        nkick = int(rng.integers(2, 6))
        nhole = int(rng.integers(1, 3))
        sigma = float(10 ** rng.uniform(-2.3, -1.0))
        # shake worst hubs
        for k in range(nkick):
            i = int(worst[k % n])
            if rng.random() < 0.15:
                i = int(rng.integers(0, n))
            v = rng.standard_normal(d)
            v = v - np.dot(v, x[i]) * x[i]
            vn = np.linalg.norm(v) + 1e-300
            v = v / vn
            x[i] = x[i] + sigma * v
            x[i] /= np.linalg.norm(x[i]) + 1e-300
        # relocate loose points to holes
        fixed_mask = np.ones(n, dtype=bool)
        cand_hole_idx = []
        for k in range(nhole):
            # pick among loosest not already kicked much
            i = int(loose[(k + int(rng.integers(0, 8))) % n])
            cand_hole_idx.append(i)
            fixed_mask[i] = False
        # also exclude kicked? keep fixed as remaining
        xf = x[fixed_mask]
        for i in cand_hole_idx:
            y = _best_hole(xf, rng, ns=4000)
            y = _hole_refine(xf, y, steps=100)
            # small jitter
            j = rng.standard_normal(d) * 1e-3
            y = y + j - np.dot(j, y) * y
            y /= np.linalg.norm(y) + 1e-300
            x[i] = y
        # anneal
        a0 = float(rng.choice([30.0, 40.0, 60.0]))
        a1 = float(rng.choice([250.0, 320.0, 450.0]))
        lr0 = 0.03; lr1 = 0.003
        if rng.random() < 0.3:
            lr0, lr1 = 0.05, 0.005
        x = _smooth(x, steps=260, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
        x = _greedy(x, steps=110, m0=0.008, m1=0.0008, lr=0.008)
        # extra short hot smooth on failure sometimes
        v = _mc(x)
        if v < gval:
            gval = v
            gbest = x.copy()
        elif v < gval + 0.002 and rng.random() < 0.05:
            # occasional side step to diversify elite? keep hopping from best only,
            # so do a couple extra smooth steps with different temp
            pass

    gbest = _lp_polish(gbest, seconds=29, t0=time.time(), deadline=DEAD - 1.0)
    gbest = _norm(gbest)
    return gbest

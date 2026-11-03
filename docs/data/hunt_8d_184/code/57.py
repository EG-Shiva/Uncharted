import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nrm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _permax(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _descend(x, steps=170, m0=0.022, m1=0.0018, lr0=0.05, lr1=0.0035):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - margin
        W = G - thr
        np.maximum(W, 0.0, out=W)
        np.fill_diagonal(W, 0.0)
        # power anneal 2 -> ~6 without expensive pow: W=(W/margin)^2 * (1+f*W/margin...)
        W /= margin
        # clip huge
        # square then sharpen late
        W = W * W
        if f > 0.5:
            # extra sharpening late: square again partially
            W = W * (1.0 - f + f * W)
        deg = W.sum(axis=1, keepdims=True) + 1e-30
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        st = lr * Gt / deg
        sn = np.linalg.norm(st, axis=1, keepdims=True) + 1e-30
        cap = 0.07
        st *= np.minimum(1.0, cap / sn)
        x = x - st
        x = _nrm(x)
    return x

def _top(x, steps=70, k=70, lr=0.006):
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = G[iu0]
        if k < flat.size:
            thr = np.partition(flat, -k)[-k]
        else:
            thr = float(np.min(flat))
        M = (G >= thr).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True) + 1e-30
        Gr = M @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        x = x - (lr * (1.0 - 0.5 * f)) * Gt / deg
        x = _nrm(x)
    return x

def _hole(Xf, rng, ns=3500):
    d = Xf.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y = _nrm(Y)
    S = Y @ Xf.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i].copy(), float(m[i])

def _refine_hole(Xf, y, steps=80):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = 60.0 * (5.0) ** f
        lr = 0.06 * (0.005 / 0.06) ** f
        s = Xf @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ Xf
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt / sw
        y = y / (np.linalg.norm(y) + 1e-300)
    return y

def _lp_polish(x, deadline):
    x = _nrm(x.copy())
    n, d = x.shape
    best = _mc(x)
    xb = x.copy()
    radius = 1.2e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() > deadline - 2.0:
            break
        if radius < 5e-14:
            break
        G = x @ x.T
        cf = G[iu]
        m = float(cf.max())
        sel = cf > m - max(20.0 * radius, 2e-9)
        I = iu[0][sel]; J = iu[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1600:
            idx = np.argpartition(-cs, 1600)[:1600]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1600
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
        tl = max(1.0, deadline - time.time() - 0.6)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _nrm(x + res.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(20261103)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=np.float64).reshape(184, 8)
    x0 = _nrm(x0)
    gbest = x0.copy()
    gval = _mc(gbest)
    n, d = 184, 8
    iu = np.triu_indices(n, 1)

    trial = 0
    while time.time() < HOP_END and time.time() < DEADLINE - 35:
        if trial >= 120:
            break
        mode = trial % 5
        xt = gbest.copy()
        # contact info cheap
        G = xt @ xt.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        per = np.max(G, axis=1)
        # loose threshold degrees for hub detection
        thrh = mcur - 0.004
        freq = (G >= thrh).sum(axis=1)
        order_low = np.argsort(per, kind="stable")
        order_hi = np.argsort(-freq, kind="stable")
        if mode == 0:
            # collective block transplant: 3 lowest -> largest voids + hub jitter
            krel = 3
            lows = order_low[:krel].tolist()
            keepset = set(lows)
            keep = [i for i in range(n) if i not in keepset]
            Xf = xt[keep].copy()
            for li in lows:
                y0, _ = _hole(Xf, rng, ns=3500)
                y0 = _refine_hole(Xf, y0, steps=70)
                xt[li] = y0
                Xf = np.vstack([Xf, y0[None, :]])
            # shake overloaded hub neighbourhood to balance
            hub = int(order_hi[0])
            s = xt @ xt[hub]; s[hub] = -2.0
            nb = np.argsort(-s)[:10]
            xt[nb] += rng.standard_normal(size=(10, d)) * 0.045
            xt[nb] = _nrm(xt[nb])
        elif mode == 1:
            # hub-void swap: move top hub to void, jitter its old neighbours
            hub = int(order_hi[trial % 3])
            others = [i for i in range(n) if i != hub]
            Xf = xt[others].copy()
            y0, _ = _hole(Xf, rng, ns=3500)
            y0 = _refine_hole(Xf, y0, steps=70)
            old = xt[hub].copy()
            xt[hub] = y0
            # pull old neighbours toward old hole a bit to fill it
            s = Xf @ old
            idx = np.argsort(-s)[:6]
            # map idx (over others) back to real indices
            real = np.array(others)[idx]
            xt[real] += (old[None, :] - xt[real]) * 0.06
            xt[real] = _nrm(xt[real])
            xt += rng.standard_normal(size=xt.shape) * 0.002
            xt = _nrm(xt)
        elif mode == 2:
            # worst-edge surgery with orthogonal kick
            w = int(np.argmax((xt @ xt.T)[iu]))
            wi, wj = int(iu[0][w]), int(iu[1][w])
            r = rng.standard_normal(d)
            r -= float(np.dot(r, xt[wi])) * xt[wi]
            r /= (np.linalg.norm(r) + 1e-300)
            xt[wi] = _nrm((xt[wi] + 0.13 * r)[None, :])[0]
            worst = np.argsort(-per)[:6]
            xt[worst] += rng.standard_normal(size=(6, d)) * 0.03
            xt[worst] = _nrm(xt[worst])
        elif mode == 3:
            # loose-block joint shake (indices with few contacts) + global tiny noise
            loos = order_low[:14]
            xt[loos] += rng.standard_normal(size=(14, d)) * 0.075
            xt[loos] = _nrm(xt[loos])
            xt += rng.standard_normal(size=xt.shape) * 0.0025
            xt = _nrm(xt)
        else:
            idx = rng.choice(n, size=10, replace=False)
            xt[idx] += rng.standard_normal(size=(10, d)) * 0.09
            xt[idx] = _nrm(xt[idx])

        if mode in (0, 1):
            xt = _descend(xt, steps=190, m0=0.025, m1=0.0018, lr0=0.05, lr1=0.0035)
            xt = _top(xt, steps=70, k=80, lr=0.006)
        elif mode == 2:
            xt = _descend(xt, steps=170, m0=0.016, m1=0.0015, lr0=0.045, lr1=0.003)
            xt = _top(xt, steps=60, k=55, lr=0.006)
        else:
            xt = _descend(xt, steps=170, m0=0.022, m1=0.0018, lr0=0.048, lr1=0.0035)
            xt = _top(xt, steps=60, k=70, lr=0.006)

        v = _mc(xt)
        if v < gval:
            gval = float(v)
            gbest = xt.copy()
        trial += 1
        # safety: if little time left break
        if time.time() > DEADLINE - 35:
            break

    if time.time() < DEADLINE - 6:
        # short greedy extra on best before LP
        gbest = _top(gbest, steps=40, k=120, lr=0.004)
        lp_dead = min(DEADLINE - 1.0, time.time() + 32.0)
        if lp_dead - time.time() > 7:
            gbest = _lp_polish(gbest, lp_dead)
    gbest = _nrm(gbest)
    return gbest

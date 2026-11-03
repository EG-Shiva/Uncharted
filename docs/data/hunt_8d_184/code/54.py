import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nrm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _anneal(x, steps, m0, m1, p0, p1, lr0, lr1):
    n, d = x.shape
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        pw = p0 + (p1 - p0) * f
        lr = lr0 * (lr1 / lr0) ** f if lr0 > 0 and lr1 > 0 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - margin
        A = G - thr
        np.maximum(A, 0.0, out=A)
        np.fill_diagonal(A, 0.0)
        inv = 1.0 / max(margin, 1e-12)
        with np.errstate(over='ignore', under='ignore'):
            if abs(pw - 1.0) > 1e-12:
                np.multiply(A, inv, out=A)
                np.power(A, pw, out=A)
            else:
                np.multiply(A, inv, out=A)
        deg = A.sum(axis=1, keepdims=True) + 1e-30
        grad = A @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        step = (lr) * grad / deg
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-30
        cap = 0.06
        scale = np.minimum(1.0, cap / (sn + 1e-30))
        x = x - step * scale
        x = _nrm(x)
    return x

def _hole(xfix, rng, ns=2000):
    d = xfix.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y = _nrm(Y)
    S = Y @ xfix.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i].copy()

def _refine(xfix, y, steps=40):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = 60.0 * (300.0 / 60.0) ** f
        lr = 0.05 * (0.005 / 0.05) ** f
        s = xfix @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ xfix
        g = g - float(np.dot(g, y)) * y
        y = y - lr * g / sw
        nr = float(np.linalg.norm(y))
        if not np.isfinite(nr) or nr == 0:
            break
        y = y / nr
    return y

def _lp(x, deadline):
    x = _nrm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.2e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() < deadline - 2.5:
        if radius < 1e-13 or it >= 25:
            break
        it += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            idx = np.argpartition(-cs, 1500)[:1500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1500
        nv = n * d + 1
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        rows = np.concatenate([np.repeat(np.arange(p), d),
                               np.repeat(np.arange(p), d),
                               np.arange(p)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            tl = max(0.5, deadline - time.time() - 1.0)
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
        v = _max_cos(y)
        if v < best:
            best = float(v); xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEAD = t0 + 168.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64).reshape(n, d)
            c = _nrm(c)
            cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _nrm(rng.standard_normal(size=(n, d)))
        gval = _max_cos(gbest)
    xb0 = gbest.copy()
    cur = gbest.copy()
    cur_val = float(gval)

    trial = 0
    while time.time() < HOP_END and time.time() < DEAD - 35.0:
        if trial >= 300:
            break
        frac = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        frac = min(max(frac, 0.0), 1.0)
        # sideways temperature: allow ~2e-4 early -> ~1e-6 late
        Tsa = 2.0e-04 * (1.0 - frac) + 1.0e-06
        xt = cur.copy()
        mode = trial % 5
        G = xt @ xt.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        pm = np.max(G, axis=1)
        deg = (G > mcur - 1e-7).sum(axis=1)
        order_low = np.argsort(pm, kind="stable")
        order_hub = np.argsort(-deg, kind="stable")
        if mode == 0:
            # relocate single worst-low (often rattler) to void + tiny jitter
            li = int(order_low[0])
            keep = [i for i in range(n) if i != li]
            xfix = xt[keep]
            y0 = _hole(xfix, rng, ns=2000)
            y0 = _refine(xfix, y0, steps=40)
            xt[li] = y0
            # tangential kick to top hub
            hub = int(order_hub[0])
            r = rng.standard_normal(d)
            r -= float(np.dot(r, xt[hub])) * xt[hub]
            rn = float(np.linalg.norm(r))
            if rn > 1e-12:
                r /= rn
                xt[hub] = _nrm((xt[hub] + 0.05 * r)[None, :])[0]
            xt = _nrm(xt)
        elif mode == 1:
            # relocate two lows to voids sequentially
            krel = 2
            lows = [int(order_low[i]) for i in range(krel)]
            keepset = set(lows)
            keep = [i for i in range(n) if i not in keepset]
            xfix = xt[keep].copy()
            for li in lows:
                y0 = _hole(xfix, rng, ns=2000)
                y0 = _refine(xfix, y0, steps=40)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            # small global tangential noise
            xt = _nrm(xt + rng.standard_normal(size=xt.shape) * 0.002)
        elif mode == 2:
            # break worst pair + shake hubs
            iu0 = np.triu_indices(n, 1)
            cf = G[iu0]
            w = int(np.argmax(cf))
            wi, wj = int(iu0[0][w]), int(iu0[1][w])
            for ii in (wi, wj):
                r = rng.standard_normal(d)
                r -= float(np.dot(r, xt[ii])) * xt[ii]
                rn = float(np.linalg.norm(r))
                if rn > 1e-12:
                    r /= rn
                    xt[ii] = _nrm((xt[ii] + 0.10 * r)[None, :])[0]
            hubs = order_hub[:8]
            xt[hubs] += rng.standard_normal(size=(8, d)) * 0.04
            xt = _nrm(xt)
        elif mode == 3:
            idx = rng.choice(n, size=8, replace=False)
            xt[idx] += rng.standard_normal(size=(8, d)) * 0.06
            xt = _nrm(xt)
        else:
            xt = _nrm(xt + rng.standard_normal(size=xt.shape) * 0.006)

        # cheap reconverge: fewer steps to allow many trials
        if mode == 1:
            xt = _anneal(xt, steps=170, m0=0.025, m1=0.002, p0=2.0, p1=6.0, lr0=0.04, lr1=0.003)
        elif mode == 2:
            xt = _anneal(xt, steps=160, m0=0.015, m1=0.0015, p0=3.0, p1=8.0, lr0=0.04, lr1=0.003)
        else:
            xt = _anneal(xt, steps=150, m0=0.02, m1=0.002, p0=2.0, p1=6.0, lr0=0.04, lr1=0.003)

        v = _max_cos(xt)
        if v < gval:
            gval = float(v); gbest = xt.copy()
        # sideways acceptance vs cur
        if v <= cur_val or rng.random() < np.exp(-(v - cur_val) / max(Tsa, 1e-12)):
            cur = xt.copy(); cur_val = float(v)
        else:
            # occasional restart from best if walk drifts too far
            if cur_val > gval + 0.002 and (trial % 20 == 19):
                cur = gbest.copy(); cur_val = float(gval)
        trial += 1

    if time.time() < DEAD - 6.0:
        try:
            gbest = _lp(gbest, DEAD)
        except Exception:
            pass
    gbest = _nrm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = xb0
    # strict feasibility: renormalize
    return gbest

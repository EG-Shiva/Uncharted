import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _smooth(x, steps, a0, a1, lr0, lr1):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
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
        x = x - lr * (gt / S)
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
        x = x - lr * (gt / np.maximum(deg, 1.0))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _refine_hole(xf, y, steps=150, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = xf @ y
        mx = np.max(s)
        w = np.exp(alpha * (s - mx))
        sw = np.sum(w) + 1e-30
        g = w @ xf
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        n = np.linalg.norm(y)
        if n == 0 or not np.isfinite(n):
            break
        y = y / n
    return y

def _hole(xf, rng, ns=6000):
    d = xf.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    S = Y @ xf.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i]

def _lp_polish(x, seconds, t_start, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds:
        if time.time() > deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 2000:
            idx = np.argpartition(-cs, 2000)[:2000]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = float(v); xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64).reshape(184, 8)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=np.float64).reshape(184, 8)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    gbest = xb.copy()
    gval = _max_cos(gbest)
    cur = gbest.copy()
    cval = gval

    trial = 0
    # hopping phase: leave ~50s for LP
    while time.time() < DEADLINE - 55.0 and (time.time() - t0) < 118.0:
        trial += 1
        # pick base: mostly best, sometimes wanderer
        if rng.random() < 0.75:
            base = gbest
        else:
            base = cur
        xt = base.copy()
        n = 184
        # analyse base quickly
        G = xt @ xt.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        pm = np.max(G, axis=1)
        cf = G[np.triu_indices(n, 1)]
        # frequencies of near-max pairs
        thr = mcur - 0.005
        iu0 = np.triu_indices(n, 1)
        # fast freq via thresholded matrix
        B = (G >= thr)
        np.fill_diagonal(B, False)
        freq = B.sum(axis=1)
        order_hi = np.argsort(-freq, kind="stable")
        order_lo = np.argsort(pm, kind="stable")  # lowest per-max first
        mode = trial % 6
        if mode == 0:
            # shake worst + overloaded hubs
            k = 4 + int(rng.integers(0, 3))
            victims = list(order_hi[:k])
            # include worst pair endpoint
            w = int(np.argmax(cf))
            wi = int(iu0[0][w])
            if wi not in victims:
                victims[0] = wi
            s = 0.10 + 0.05 * rng.random()
            xt[victims] += rng.standard_normal(size=(len(victims), 8)) * s
            xt[victims] /= np.linalg.norm(xt[victims], axis=1, keepdims=True)
            xt = _smooth(xt, steps=260, a0=45.0, a1=300.0, lr0=0.028, lr1=0.0025)
            xt = _greedy(xt, steps=110, m0=0.007, m1=0.0007, lr=0.007)
        elif mode == 1:
            # relocate 2 rattlers/loose to holes
            k = 2 if (trial % 12 != 0) else 3
            loose = list(order_lo[:k])
            keep = [i for i in range(n) if i not in set(loose)]
            xf = xt[keep].copy()
            for li in loose:
                y0 = _hole(xf, rng, ns=6000)
                y0 = _refine_hole(xf, y0, steps=150)
                xt[li] = y0
                xf = np.vstack([xf, y0[None, :]])
            # tiny jitter others
            xt[keep] += rng.standard_normal(size=(len(keep), 8)) * 0.003
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _smooth(xt, steps=300, a0=40.0, a1=320.0, lr0=0.030, lr1=0.003)
            xt = _greedy(xt, steps=130, m0=0.008, m1=0.0007, lr=0.008)
        elif mode == 2:
            # global small jitter
            xt = xt + rng.standard_normal(size=xt.shape) * 0.012
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _smooth(xt, steps=240, a0=50.0, a1=300.0, lr0=0.025, lr1=0.0025)
            xt = _greedy(xt, steps=100, m0=0.006, m1=0.0008, lr=0.007)
        elif mode == 3:
            # block shake around random center
            c = int(rng.integers(0, n))
            grow = xt @ xt[c]
            nearest = np.argsort(-grow)[:12]
            xt[nearest] += rng.standard_normal(size=(len(nearest), 8)) * 0.07
            xt[nearest] /= np.linalg.norm(xt[nearest], axis=1, keepdims=True)
            xt = _smooth(xt, steps=260, a0=45.0, a1=300.0, lr0=0.028, lr1=0.0025)
            xt = _greedy(xt, steps=110, m0=0.007, m1=0.0007, lr=0.007)
        elif mode == 4:
            # relocate single worst to hole + shake neighbours of worst
            w = int(np.argmax(cf))
            wi, wj = int(iu0[0][w]), int(iu0[1][w])
            victim = wi if rng.random() < 0.5 else wj
            keep = [i for i in range(n) if i != victim]
            xf = xt[keep].copy()
            y0 = _hole(xf, rng, ns=6000)
            y0 = _refine_hole(xf, y0, steps=150)
            xt[victim] = y0
            # shake its old neighbours a bit
            nbrs = np.argsort(-G[victim])[:6]
            xt[nbrs] += rng.standard_normal(size=(6, 8)) * 0.02
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = _smooth(xt, steps=280, a0=42.0, a1=310.0, lr0=0.030, lr1=0.0028)
            xt = _greedy(xt, steps=120, m0=0.007, m1=0.0007, lr=0.0075)
        else:
            # medium random shake
            idx = rng.choice(n, size=8, replace=False)
            xt[idx] += rng.standard_normal(size=(8, 8)) * 0.12
            xt[idx] /= np.linalg.norm(xt[idx], axis=1, keepdims=True)
            xt = _smooth(xt, steps=250, a0=45.0, a1=300.0, lr0=0.028, lr1=0.0025)
            xt = _greedy(xt, steps=110, m0=0.007, m1=0.0008, lr=0.007)

        v = _max_cos(xt)
        if v < gval:
            gval = float(v); gbest = xt.copy()
            cur = xt.copy(); cval = float(v)
        else:
            # wandering acceptance: keep some diversity
            if v < cval or rng.random() < 0.08:
                cur = xt.copy(); cval = float(v)
        if time.time() > DEADLINE - 56.0:
            break

    remain = DEADLINE - time.time() - 1.0
    if remain > 8:
        gbest = _lp_polish(gbest, min(remain, 52.0), time.time(), DEADLINE - 0.5)
    gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(gbest)):
        gbest = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return gbest

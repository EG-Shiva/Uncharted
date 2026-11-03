import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _contacts(x, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    C = (g > m - tol)
    deg = C.sum(axis=1)
    return m, deg, g

def _anneal_power(x, steps=260, m0=0.02, m1=0.002, p0=2.0, p1=6.0, lr0=0.05, lr1=0.004):
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
        if pw != 1.0:
            # avoid overflow: normalize by margin
            # A in [0, margin]; scale
            with np.errstate(over='ignore'):
                A = np.power(A / max(margin, 1e-12), pw)
        deg = A.sum(axis=1, keepdims=True) + 1e-30
        grad = A @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt / deg
        # clip step norm for stability
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-30
        cap = 0.08
        scale = np.minimum(1.0, cap / (sn + 1e-30))
        x = x - step * scale
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

def _greedy_top(x, steps=120, k=60, lr=0.007):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # pick top-k pairs
        flat = G[iu]
        if k < flat.size:
            thr = np.partition(flat, -k)[-k]
        else:
            thr = flat.min()
        M = (G >= thr)
        np.fill_diagonal(M, False)
        grad = M.astype(np.float64) @ x
        deg = M.sum(axis=1, keepdims=True).astype(np.float64) + 1e-30
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        lrf = lr * (1.0 - 0.6 * f)
        x = x - lrf * gt / deg
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

def _best_hole(x_fixed, rng, n_sample=4000):
    d = x_fixed.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    S = Y @ x_fixed.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx], float(m[idx])

def _refine_hole(x_fixed, y, steps=120):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = 60.0 * (300.0 / 60.0) ** f
        lr = 0.06 * (0.005 / 0.06) ** f
        s = x_fixed @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ x_fixed
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt / sw
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _lp_polish(x, seconds, t_start, deadline):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds:
        if time.time() > deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argpartition(-csel, 3000)[:3000]
            I, J, csel = I[idx], J[idx], csel[idx]
            p = 3000
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-300)
        v = _max_cos(y)
        if v < best:
            best = float(v)
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64).reshape(184, 8)
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-300)
    gbest = xb.copy()
    gval = _max_cos(gbest)

    trial = 0
    # hopping phase: leave ~50s for LP
    while time.time() < DEADLINE - 55.0 and (time.time() - t0) < 115.0:
        if trial >= 90:
            break
        mode = trial % 6
        xt = gbest.copy()
        if mode == 0:
            # worst-edge surgery: break single worst pair
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            i, j = divmod(ij, 184)
            # orthogonal kick to i
            r = rng.standard_normal(8)
            r -= float(np.dot(r, xt[i])) * xt[i]
            r /= (np.linalg.norm(r) + 1e-300)
            xt[i] = xt[i] + 0.14 * r
            xt[i] /= (np.linalg.norm(xt[i]) + 1e-300)
            # small shake of j and their common neighbours
            pm = _per_max(xt)
            worst = np.argsort(-pm)[:5]
            xt[worst] += rng.standard_normal(size=(5, 8)) * 0.03
            xt[worst] /= (np.linalg.norm(xt[worst], axis=1, keepdims=True) + 1e-300)
        elif mode == 1:
            pm = _per_max(xt)
            worst = np.argsort(-pm)[:8]
            xt[worst] += rng.standard_normal(size=(8, 8)) * 0.06
            xt[worst] /= (np.linalg.norm(xt[worst], axis=1, keepdims=True) + 1e-300)
        elif mode == 2:
            # relocate rattler / low-degree point to largest hole
            m, deg, G = _contacts(xt, tol=1e-7)
            order = np.argsort(deg)
            krel = 2 if (trial % 12 < 8) else 3
            lows = order[:krel].tolist()
            keep = [i for i in range(184) if i not in set(lows)]
            xfix = xt[keep].copy()
            for li in lows:
                y0, _ = _best_hole(xfix, rng, n_sample=4000)
                y0 = _refine_hole(xfix, y0, steps=100)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            xt[keep] += rng.standard_normal(size=(len(keep), 8)) * 0.002
            xt /= (np.linalg.norm(xt, axis=1, keepdims=True) + 1e-300)
        elif mode == 3:
            # overloaded hub neighbourhood shake
            m, deg, G = _contacts(xt, tol=1e-7)
            hub = int(np.argmax(deg))
            # nearest 12 to hub
            s = (xt @ xt[hub])
            s[hub] = -2.0
            nb = np.argsort(-s)[:12]
            xt[nb] += rng.standard_normal(size=(12, 8)) * 0.07
            xt[nb] /= (np.linalg.norm(xt[nb], axis=1, keepdims=True) + 1e-300)
            # also kick hub orthogonally a bit
            r = rng.standard_normal(8)
            r -= float(np.dot(r, xt[hub])) * xt[hub]
            r /= (np.linalg.norm(r) + 1e-300)
            xt[hub] = xt[hub] + 0.05 * r
            xt[hub] /= (np.linalg.norm(xt[hub]) + 1e-300)
        elif mode == 4:
            idx = rng.choice(184, size=12, replace=False)
            xt[idx] += rng.standard_normal(size=(12, 8)) * 0.10
            xt[idx] /= (np.linalg.norm(xt[idx], axis=1, keepdims=True) + 1e-300)
        else:
            xt = xt + rng.standard_normal(size=xt.shape) * 0.008
            xt /= (np.linalg.norm(xt, axis=1, keepdims=True) + 1e-300)

        # reconverge: power anneal + top-k greedy, params vary by mode
        if mode == 2:
            xt = _anneal_power(xt, steps=280, m0=0.025, m1=0.002, p0=2.0, p1=7.0, lr0=0.05, lr1=0.004)
            xt = _greedy_top(xt, steps=130, k=80, lr=0.007)
        elif mode == 0:
            xt = _anneal_power(xt, steps=240, m0=0.015, m1=0.0015, p0=3.0, p1=8.0, lr0=0.045, lr1=0.003)
            xt = _greedy_top(xt, steps=110, k=50, lr=0.006)
        else:
            xt = _anneal_power(xt, steps=240, m0=0.02, m1=0.002, p0=2.0, p1=6.0, lr0=0.045, lr1=0.0035)
            xt = _greedy_top(xt, steps=100, k=60, lr=0.0065)

        v = _max_cos(xt)
        if v < gval:
            gval = float(v)
            gbest = xt.copy()
        trial += 1

    remain = DEADLINE - time.time() - 1.0
    if remain > 8:
        lp_t = min(remain, 50.0)
        gbest = _lp_polish(gbest, lp_t, t_start=time.time(), deadline=DEADLINE)
    gbest = gbest / (np.linalg.norm(gbest, axis=1, keepdims=True) + 1e-300)
    if not np.all(np.isfinite(gbest)):
        gbest = xb
    return gbest

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
        # power law weight
        with np.errstate(over='ignore', under='ignore'):
            if pw != 1.0:
                A = np.power(A * inv, pw)
            else:
                A *= inv
        deg = A.sum(axis=1, keepdims=True) + 1e-30
        grad = A @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        step = (lr / 1.0) * grad / deg
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-30
        cap = 0.08
        np.minimum(1.0, cap / (sn + 1e-30), out=np.empty_like(sn))  # dummy to avoid alloc issues
        scale = np.minimum(1.0, cap / (sn + 1e-30))
        x = x - step * scale
        x = _nrm(x)
    return x

def _greedy(x, steps, k, lr):
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = G[iu0]
        thr = np.partition(flat, -k)[-k] if k < flat.size else flat.min()
        M = (G >= thr)
        np.fill_diagonal(M, False)
        grad = M.astype(np.float64) @ x
        deg = M.sum(axis=1, keepdims=True).astype(np.float64) + 1e-30
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        lrf = lr * (1.0 - 0.5 * f)
        x = x - lrf * grad / deg
        x = _nrm(x)
    return x

def _hole(xfix, rng, ns=3000):
    d = xfix.shape[1]
    Y = rng.standard_normal(size=(ns, d))
    Y = _nrm(Y)
    S = Y @ xfix.T
    m = S.max(axis=1)
    i = int(np.argmin(m))
    return Y[i], float(m[i])

def _refine(xfix, y, steps=60):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = 60.0 * (300.0 / 60.0) ** f
        lr = 0.06 * (0.005 / 0.06) ** f
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

def _lp(x, seconds, t_start, deadline):
    x = _nrm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() - t_start < seconds and time.time() < deadline:
        if radius < 1e-14 or it > 40:
            break
        it += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-cs, 2500)[:2500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2500
        nv = n * d + 1
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
        y = _nrm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = float(v); xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEAD = t0 + 175.0
    rng = np.random.default_rng(11)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64).reshape(184, 8)
            c = _nrm(c)
            cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _nrm(rng.standard_normal(size=(184, 8)))
        gval = _max_cos(gbest)
    xb0 = gbest.copy()

    trial = 0
    while time.time() < DEAD - 32.0 and (time.time() - t0) < 120.0:
        if trial >= 70:
            break
        mode = trial % 6
        xt = gbest.copy()
        if mode == 0:
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            i, j = divmod(ij, 184)
            r = rng.standard_normal(8)
            r -= float(np.dot(r, xt[i])) * xt[i]
            r /= (np.linalg.norm(r) + 1e-300)
            xt[i] = _nrm((xt[i] + 0.16 * r)[None, :])[0]
            pm = _per_max(xt)
            w = np.argsort(-pm)[:5]
            xt[w] += rng.standard_normal(size=(5, 8)) * 0.035
            xt = _nrm(xt)
        elif mode == 1:
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = (G > m - 1e-7).sum(axis=1)
            hub = int(np.argmax(deg))
            s = xt @ xt[hub]; s[hub] = -2.0
            nb = np.argsort(-s)[:12]
            xt[nb] += rng.standard_normal(size=(12, 8)) * 0.075
            r = rng.standard_normal(8)
            r -= float(np.dot(r, xt[hub])) * xt[hub]
            r /= (np.linalg.norm(r) + 1e-300)
            xt[hub] = _nrm((xt[hub] + 0.06 * r)[None, :])[0]
            xt = _nrm(xt)
        elif mode == 2:
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = (G > m - 1e-7).sum(axis=1)
            order = np.argsort(deg)
            krel = 1 if (trial % 12 < 6) else 2
            lows = order[:krel].tolist()
            keepset = set(lows)
            keep = [i for i in range(184) if i not in keepset]
            xfix = xt[keep].copy()
            for li in lows:
                y0, _ = _hole(xfix, rng, ns=3000)
                y0 = _refine(xfix, y0, steps=60)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            xt[keep] += rng.standard_normal(size=(len(keep), 8)) * 0.002
            xt = _nrm(xt)
        elif mode == 3:
            pm = _per_max(xt)
            w = np.argsort(-pm)[:8]
            xt[w] += rng.standard_normal(size=(8, 8)) * 0.06
            xt = _nrm(xt)
        elif mode == 4:
            idx = rng.choice(184, size=10, replace=False)
            xt[idx] += rng.standard_normal(size=(10, 8)) * 0.09
            xt = _nrm(xt)
        else:
            xt = _nrm(xt + rng.standard_normal(size=xt.shape) * 0.008)

        if mode == 2:
            xt = _anneal(xt, steps=260, m0=0.028, m1=0.002, p0=2.0, p1=7.0, lr0=0.05, lr1=0.004)
            xt = _greedy(xt, steps=110, k=80, lr=0.007)
        elif mode == 0:
            xt = _anneal(xt, steps=230, m0=0.016, m1=0.0015, p0=3.0, p1=9.0, lr0=0.045, lr1=0.003)
            xt = _greedy(xt, steps=100, k=50, lr=0.006)
        else:
            xt = _anneal(xt, steps=220, m0=0.02, m1=0.002, p0=2.0, p1=6.5, lr0=0.045, lr1=0.0035)
            xt = _greedy(xt, steps=90, k=60, lr=0.0065)

        v = _max_cos(xt)
        if v < gval:
            gval = float(v); gbest = xt.copy()
        trial += 1
        if time.time() > DEAD - 32.0:
            break

    remain = DEAD - time.time() - 1.0
    if remain > 8:
        gbest = _lp(gbest, min(remain, 28.0), time.time(), DEAD)
    gbest = _nrm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = xb0
    return gbest

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

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _reconverge(x, steps, m0, m1, lr0, lr1):
    n = x.shape[0]
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        lr = lr0 * (lr1 / lr0) ** f if lr0 > 0 and lr1 > 0 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - margin
        A = G - thr
        np.maximum(A, 0.0, out=A)
        np.fill_diagonal(A, 0.0)
        # power 2 weighting normalized
        A /= max(margin, 1e-12)
        A *= A
        deg = A.sum(axis=1, keepdims=True) + 1e-30
        grad = A @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = (lr * gt) / deg
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-30
        cap = 0.08
        sc = np.minimum(1.0, cap / sn)
        x = x - step * sc
        x = _norm(x)
    return x

def _greedy_top(x, steps, k, lr):
    n = x.shape[0]
    iu0 = np.triu_indices(n, 1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = G[iu0]
        if k < flat.size:
            thr = np.partition(flat, -k)[-k]
        else:
            thr = float(flat.min())
        M = (G >= thr)
        np.fill_diagonal(M, False)
        grad = M.astype(np.float64) @ x
        deg = M.sum(axis=1, keepdims=True).astype(np.float64) + 1e-30
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        lrf = lr * (1.0 - 0.5 * f)
        x = x - lrf * gt / deg
        x = _norm(x)
    return x

def _best_hole(x_fixed, rng, n_sample=2500):
    d = x_fixed.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y = _norm(Y)
    S = Y @ x_fixed.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx], float(m[idx])

def _refine_hole(x_fixed, y, steps=40):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = 60.0 * (300.0 / 60.0) ** f
        lr = 0.05 * (0.006 / 0.05) ** f
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

def _lp_polish(x, budget, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    t_start = time.time()
    while time.time() - t_start < budget:
        if time.time() > deadline - 1.5:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
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
            best = float(v)
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 3e-7:
                break
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(11)
    n, d = 184, 8
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64).reshape(n, d)
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
    xb0 = gbest.copy()

    trial = 0
    # hopping budget: stop early enough for LP
    while time.time() < DEADLINE - 32.0:
        if trial >= 250:
            break
        if time.time() - t0 > 135.0:
            break
        mode = trial % 5
        xt = gbest.copy()
        if mode == 0:
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            i, j = divmod(ij, n)
            for kk, sgn in ((i, 1.0), (j, -0.5)):
                r = rng.standard_normal(d)
                r -= float(np.dot(r, xt[kk])) * xt[kk]
                rn = float(np.linalg.norm(r)) + 1e-300
                r /= rn
                amp = 0.12 if kk == i else 0.05
                xt[kk] = xt[kk] + sgn * amp * r
                xt[kk] /= (float(np.linalg.norm(xt[kk])) + 1e-300)
            pm = _per_max(xt)
            worst = np.argsort(-pm)[:5]
            xt[worst] += rng.standard_normal(size=(5, d)) * 0.025
            xt[worst] = _norm(xt[worst])
        elif mode == 1:
            pm = _per_max(xt)
            k = 6 + int(rng.integers(0, 4))
            worst = np.argsort(-pm)[:k]
            xt[worst] += rng.standard_normal(size=(k, d)) * (0.04 + 0.02 * rng.random())
            xt[worst] = _norm(xt[worst])
        elif mode == 2:
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = (G > m - 1e-7).sum(axis=1)
            order = np.argsort(deg)
            krel = 1 + int(rng.integers(0, 2))  # 1-2
            if rng.random() < 0.3:
                krel = 3
            lows = order[:krel].tolist()
            keepset = set(lows)
            keep = [i for i in range(n) if i not in keepset]
            xfix = xt[keep].copy()
            for li in lows:
                y0, _ = _best_hole(xfix, rng, n_sample=2500)
                y0 = _refine_hole(xfix, y0, steps=40)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            xt[keep] += rng.standard_normal(size=(len(keep), d)) * 0.002
            xt = _norm(xt)
        elif mode == 3:
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = (G > m - 1e-7).sum(axis=1)
            hub = int(np.argmax(deg))
            s = (xt @ xt[hub])
            s[hub] = -2.0
            nb = np.argsort(-s)[:10]
            xt[nb] += rng.standard_normal(size=(10, d)) * 0.06
            xt[nb] = _norm(xt[nb])
            r = rng.standard_normal(d)
            r -= float(np.dot(r, xt[hub])) * xt[hub]
            r /= (float(np.linalg.norm(r)) + 1e-300)
            xt[hub] = xt[hub] + 0.05 * r
            xt[hub] /= (float(np.linalg.norm(xt[hub])) + 1e-300)
        else:
            k = 8 + int(rng.integers(0, 6))
            idx = rng.choice(n, size=k, replace=False)
            xt[idx] += rng.standard_normal(size=(k, d)) * 0.08
            xt[idx] = _norm(xt[idx])

        # reconverge params vary
        if mode == 2:
            steps = 120 + int(rng.integers(0, 40))
            xt = _reconverge(xt, steps=steps, m0=0.025, m1=0.002, lr0=0.05, lr1=0.004)
            xt = _greedy_top(xt, steps=50, k=80, lr=0.007)
        elif mode == 0:
            steps = 100 + int(rng.integers(0, 40))
            xt = _reconverge(xt, steps=steps, m0=0.015, m1=0.0015, lr0=0.045, lr1=0.003)
            xt = _greedy_top(xt, steps=50, k=50, lr=0.006)
        else:
            steps = 100 + int(rng.integers(0, 40))
            xt = _reconverge(xt, steps=steps, m0=0.02, m1=0.002, lr0=0.045, lr1=0.0035)
            xt = _greedy_top(xt, steps=40, k=60, lr=0.0065)

        v = _max_cos(xt)
        if v < gval:
            gval = float(v)
            gbest = xt.copy()
        trial += 1

    remain = DEADLINE - time.time() - 1.0
    if remain > 6.0:
        lp_budget = min(remain, 28.0)
        try:
            gbest = _lp_polish(gbest, lp_budget, DEADLINE)
        except Exception:
            pass
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(xb0)
    return gbest

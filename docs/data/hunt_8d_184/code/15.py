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

def _smooth_refine(x, steps=320, a0=40.0, a1=320.0, lr0=0.030, lr1=0.0025):
    n, d = x.shape
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
        direction = gt / S
        x = x - lr * direction
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy_refine(x, steps=140, m0=0.008, m1=0.0007, lr=0.008):
    n, d = x.shape
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        thr = m - margin
        M = (G > thr).astype(np.float64)
        grad = M @ x
        deg = M.sum(axis=1, keepdims=True)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        direction = gt / np.maximum(deg, 1.0)
        x = x - lr * direction
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _best_hole_sample(x_fixed, rng, n_sample=6000):
    d = x_fixed.shape[1]
    Y = rng.standard_normal(size=(n_sample, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    S = Y @ x_fixed.T
    m = S.max(axis=1)
    idx = int(np.argmin(m))
    return Y[idx], float(m[idx])

def _refine_hole(x_fixed, y, steps=180, a0=60.0, a1=300.0, lr0=0.06, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = x_fixed @ y
        mx = np.max(s)
        w = np.exp(alpha * (s - mx))
        sw = np.sum(w) + 1e-30
        g = w @ x_fixed
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def _lp_polish(x, seconds, t_start=None, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.2e-3
    if t_start is None:
        t_start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 5000:
            # keep worst 5000 to bound time
            idx = np.argpartition(-c[sel], 5000)[:5000]
            I, J = I[idx], J[idx]
            cs = c[sel][idx]
            p = 5000
        else:
            cs = c[sel]
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (184, 8):
                a = a.reshape(184, 8)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        xb = rng.standard_normal(size=(184, 8))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    else:
        vals = [_max_cos(a) for a in cands]
        xb = cands[int(np.argmin(vals))].copy()
    gbest = xb.copy()
    gval = _max_cos(gbest)

    trial = 0
    # hopping phase, leave ~50s for LP
    while True:
        if time.time() > DEADLINE - 55:
            break
        if time.time() - t0 > 118:
            break
        if trial >= 300:
            break
        mode = trial % 6
        xt = gbest.copy()
        if mode == 0:
            xt = xt + rng.standard_normal(size=xt.shape) * 0.008
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif mode == 1:
            pm = _per_max(xt)
            worst = np.argsort(-pm)[:5]
            xt[worst] += rng.standard_normal(size=(5, 8)) * 0.08
            xt[worst] /= np.linalg.norm(xt[worst], axis=1, keepdims=True)
        elif mode == 2:
            idx = rng.choice(184, size=12, replace=False)
            xt[idx] += rng.standard_normal(size=(12, 8)) * 0.14
            xt[idx] /= np.linalg.norm(xt[idx], axis=1, keepdims=True)
        elif mode == 3:
            # destroy hub + rattler, repair into voids
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = (G > m - 0.003).sum(axis=1)
            pm = np.max(G, axis=1)
            i_hub = int(np.argmax(cnt))
            # rattler: min cnt, tie by min per-max
            order = np.argsort(cnt * 10.0 + pm)
            victims = [int(order[0]), int(i_hub)]
            # occasionally 3rd victim: second lowest
            if trial % 12 == 3:
                if order[1] not in victims:
                    victims.append(int(order[1]))
            keep = [i for i in range(184) if i not in set(victims)]
            xfix = xt[keep].copy()
            for li in victims:
                y0, _ = _best_hole_sample(xfix, rng, n_sample=6000)
                y0 = _refine_hole(xfix, y0, steps=160)
                xt[li] = y0
                xfix = np.vstack([xfix, y0[None, :]])
            xt[keep] += rng.standard_normal(size=(len(keep), 8)) * 0.0025
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif mode == 4:
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            c = int(rng.integers(0, 184))
            nearest = np.argsort(-G[c])[:12]
            xt[nearest] += rng.standard_normal(size=(12, 8)) * 0.06
            xt[nearest] /= np.linalg.norm(xt[nearest], axis=1, keepdims=True)
        else:
            # shake overloaded hubs + global jitter
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = (G > m - 0.003).sum(axis=1)
            top = np.argsort(-cnt)[:4]
            xt[top] += rng.standard_normal(size=(4, 8)) * 0.10
            xt[top] /= np.linalg.norm(xt[top], axis=1, keepdims=True)
            xt += rng.standard_normal(size=xt.shape) * 0.006
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)

        if mode == 3:
            xt = _smooth_refine(xt, steps=360, a0=38.0, a1=340.0, lr0=0.032, lr1=0.0028)
            xt = _greedy_refine(xt, steps=160, m0=0.009, m1=0.0006, lr=0.008)
        elif mode == 5:
            xt = _smooth_refine(xt, steps=300, a0=50.0, a1=280.0, lr0=0.026, lr1=0.0022)
            xt = _greedy_refine(xt, steps=120, m0=0.006, m1=0.0008, lr=0.007)
        else:
            xt = _smooth_refine(xt, steps=320, a0=45.0, a1=310.0, lr0=0.028, lr1=0.0025)
            xt = _greedy_refine(xt, steps=130, m0=0.007, m1=0.0007, lr=0.0075)
        v = _max_cos(xt)
        if v < gval:
            gval = float(v)
            gbest = xt.copy()
        trial += 1

    remain = DEADLINE - time.time() - 1.0
    if remain > 8:
        lp_t = min(remain, 52.0)
        gbest = _lp_polish(gbest, lp_t, t_start=time.time(), deadline=DEADLINE)
    gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(gbest)):
        gbest = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return gbest

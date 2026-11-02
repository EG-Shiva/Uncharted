import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, ci)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T = 168.0
    rng = np.random.RandomState(7)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for z in (xb, xr):
        if z is not None and z.shape == (n, d):
            zn = z / np.linalg.norm(z, axis=1, keepdims=True)
            cands.append(((_max_cos(zn)), zn))
    if len(cands) == 0:
        zn = rng.randn(n, d); zn /= np.linalg.norm(zn, axis=1, keepdims=True)
        cands.append((_max_cos(zn), zn))
    cands.sort(key=lambda t: t[0])
    best_val, Xbest = cands[0][0], cands[0][1].copy()
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0[0], iu0[1]

    def smooth_run(X0, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m1 = np.zeros_like(X); v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        grad = np.empty_like(X)
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            c = G[I0, J0]
            cmax = c.max()
            e = np.exp(s * (c - cmax))
            ss = e.sum()
            if ss == 0 or not np.isfinite(ss):
                w = (c >= cmax - 1e-12).astype(float)
                ssum = w.sum()
                if ssum == 0:
                    w[:] = 1.0 / len(w)
                else:
                    w /= ssum
            else:
                w = e / ss
            grad.fill(0.0)
            np.add.at(grad, I0, w[:, None] * X[J0])
            np.add.at(grad, J0, w[:, None] * X[I0])
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 *= b1; m1 += (1 - b1) * gt
            v1 *= b2; v1 += (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
            b1p *= b1; b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            # mild clip
            nr = np.sqrt((step * step).sum(axis=1, keepdims=True)) + 1e-12
            clip = np.minimum(1.0, 3.0 / nr)
            step = step * clip
            X -= lr * step
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X /= nn
            if (k & 255) == 255 and time.time() - t0 > T - 22:
                break
        return X

    def greedy_reinsert(Y, idxs, ntry=6000):
        Y = Y.copy()
        for idx in idxs:
            C = rng.randn(ntry, d)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            others = np.delete(Y, idx, axis=0)
            dots = C @ others.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            Y[idx] = C[bi]
        return Y

    it = 0
    # quick contact estimate of best
    while time.time() - t0 < T - 22:
        it += 1
        r = rng.rand()
        X0 = None
        if r < 0.35:
            sigma = 10 ** rng.uniform(-2.6, -0.9)
            X0 = Xbest + sigma * rng.randn(n, d)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            ntry = 0
        elif r < 0.65:
            k = int(rng.choice([1, 2, 3, 4, 5, 7]))
            idx = rng.choice(n, k, replace=False)
            X0 = Xbest.copy()
            for ii in idx:
                vv = rng.randn(d); vv /= np.linalg.norm(vv)
                X0[ii] = vv
            # greedy refine
            X0 = greedy_reinsert(X0, idx, ntry=6000)
        else:
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            thr = 10 ** rng.uniform(-3.5, -2.0)
            Cmat = G > m - thr
            cnt = Cmat.sum(axis=1)
            order_desc = np.argsort(-cnt)
            order_asc = np.argsort(cnt)
            k = int(rng.randint(2, 8))
            # mix overloaded + rattlers + random
            idx_hi = order_desc[:k]
            idx_lo = order_asc[:1]
            idx = np.unique(np.concatenate([idx_hi, idx_lo]))
            X0 = Xbest.copy()
            sigma = 10 ** rng.uniform(-2.2, -0.9)
            X0[idx] += sigma * rng.randn(len(idx), d)
            # occasionally also jitter one random point hard
            if rng.rand() < 0.4:
                j = int(rng.randint(0, n))
                vv = rng.randn(d); vv /= np.linalg.norm(vv)
                mix = rng.rand() * 0.8 + 0.2
                X0[j] = (1 - mix) * X0[j] + mix * vv
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        steps = int(rng.choice([700, 1200, 2000, 2800]))
        s0 = float(10 ** rng.uniform(1.3, 1.9))
        s1 = float(10 ** rng.uniform(2.2, 2.9))
        if s1 < s0:
            s0, s1 = s1, s0
        lr0 = float(10 ** rng.uniform(-2.3, -1.7))
        lr1 = float(10 ** rng.uniform(-3.8, -3.0))
        X1 = smooth_run(X0, steps, s0, s1, lr0, lr1)
        v1 = _max_cos(X1)
        # only LP-polish if within striking distance to save time
        if v1 < best_val + 0.006:
            budget = 1.2 if v1 > best_val else 2.5
            # leave time for final polish
            if time.time() - t0 > T - 30:
                budget = 0.6
            X1p = _lp_polish(X1, seconds=budget)
            v1p = _max_cos(X1p)
        else:
            X1p, v1p = X1, v1
        if v1p < best_val:
            best_val = v1p
            Xbest = X1p.copy()
        # occasional re-anchor to second best basin: small random walk even if worse
        elif rng.rand() < 0.02 and v1 < best_val + 0.002:
            # accept sideways to diversify anchor for next kicks (elitist with small prob)
            pass
        if time.time() - t0 >= T - 22:
            break
    # final squeeze of best
    remain = T - (time.time() - t0)
    if remain > 3:
        Xbest = _lp_polish(Xbest, seconds=min(remain - 1, 20.0), r0=2e-3)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

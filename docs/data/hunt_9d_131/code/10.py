import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def quench(X, steps, lr0, lr1, m0, m1, rng=None):
    # exponential-weighted tangential repulsion
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        T = max(margin / 3.0, 1e-6)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        th = m - margin
        Mask = G >= th
        # exp weight: 1.0 at max, ~0.05 at threshold
        W = np.zeros_like(G)
        W[Mask] = np.exp((G[Mask] - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X /= nrm
    return X


def refine_void(c, Xk, steps=400, lr0=0.03, lr1=0.002):
    # maximize minimal angle to Xk: minimize max(c @ Xk.T)
    c = c / np.linalg.norm(c)
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f)
        g = Xk @ c  # (m,)
        mg = float(np.max(g))
        # softmax weight on nearest
        k = 60.0
        w = np.exp(k * (g - mg))
        w /= np.sum(w)
        grad = w @ Xk  # (d,) points toward neighbours
        # move away: c <- c - lr * tangential(grad)
        d = float(grad @ c)
        gt = grad - d * c
        c = c - lr * gt * 10.0
        c /= max(np.linalg.norm(c), 1e-12)
    return c


def find_void(Xk, rng, n_sample=15000):
    d = Xk.shape[1]
    cand = rng.standard_normal((n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.max(cand @ Xk.T, axis=1)
    idx = np.argpartition(scores, 6)[:6]
    best_c = None
    best_s = 1e9
    for ii in idx:
        c0 = cand[ii].copy()
        c1 = refine_void(c0, Xk, steps=300)
        s = float(np.max(Xk @ c1))
        if s < best_s:
            best_s = s
            best_c = c1
    return best_c


def hub_order(X, mv):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    # strict contacts for hub ranking
    C = G >= (mv - 1e-7)
    deg = C.sum(axis=1)
    order = np.argsort(-deg)
    return order, deg, G


def rattler_idx(deg):
    return np.where(deg == 0)[0]


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)

    it = 0
    # leave time for final LP
    while time.time() - t0 < 135:
        it += 1
        cycle = it % 7
        order, deg, Gmat = hub_order(best, best_val)
        Y = best.copy()

        if cycle == 0:
            # delete 1 hub, reinsert at void
            k = 1
            drop = order[:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[drop] = False
            Xk = Y[keep_mask]
            for ddi in drop:
                nv = find_void(Xk, rng)
                Xk = np.vstack([Xk, nv.reshape(1, -1)])
            # Xk has n rows but order changed; map back: keep rows + new
            # preserve indexing simplicity: put new points at dropped positions
            Y[drop] = Xk[-k:]
            # tiny noise to avoid exact symmetry
            Y += 0.002 * rng.standard_normal(Y.shape)
        elif cycle == 1:
            k = 2
            # pick 2 hubs (top hub + random among top-15)
            top = order[:15]
            drop = np.concatenate([[order[0]], rng.choice(top[1:], k - 1, replace=False)])
            keep_mask = np.ones(n, dtype=bool); keep_mask[drop] = False
            Xk = Y[keep_mask]
            for ddi in drop:
                nv = find_void(Xk, rng)
                Xk = np.vstack([Xk, nv.reshape(1, -1)])
            Y[drop] = Xk[-k:]
            Y += 0.003 * rng.standard_normal(Y.shape)
        elif cycle == 2:
            # coherent block shake of hub neighbourhood
            h = int(order[0])
            g = Gmat[h].copy()
            # neighbours with high cosine
            mv = float(np.max(Gmat))
            neigh = np.where(g >= mv - 0.015)[0]
            if len(neigh) < 4:
                neigh = np.argsort(-g)[:8]
            shift = rng.standard_normal(d); shift /= np.linalg.norm(shift)
            amp = float(rng.uniform(0.05, 0.12))
            Y[neigh] += amp * shift[None, :] + 0.01 * rng.standard_normal((len(neigh), d))
        elif cycle == 3:
            # single hub Gaussian kick
            h = int(rng.choice(order[:10]))
            Y[h] += 0.14 * rng.standard_normal(d)
            Y += 0.004 * rng.standard_normal(Y.shape)
        elif cycle == 4:
            # global shake + hub kick
            Y += 0.013 * rng.standard_normal(Y.shape)
            h = int(order[int(rng.integers(0, 5))])
            Y[h] += 0.10 * rng.standard_normal(d)
        elif cycle == 5:
            # delete 3 hubs
            k = 3
            top = order[:20]
            drop = np.array([order[0], order[1], rng.choice(top[2:], 1, replace=False)[0]])
            keep_mask = np.ones(n, dtype=bool); keep_mask[drop] = False
            Xk = Y[keep_mask]
            for ddi in drop:
                nv = find_void(Xk, rng)
                Xk = np.vstack([Xk, nv.reshape(1, -1)])
            Y[drop] = Xk[-k:]
            Y += 0.003 * rng.standard_normal(Y.shape)
        else:
            # rattler relocation to voids
            ridx = rattler_idx(deg)
            if len(ridx) == 0:
                # use lowest-degree points as pseudo-rattlers
                ridx = order[-4:]
            keep_mask = np.ones(n, dtype=bool); keep_mask[ridx] = False
            Xk = Y[keep_mask]
            # relocate each rattler sequentially
            for r in ridx:
                nv = find_void(Xk, rng, n_sample=8000)
                Xk = np.vstack([Xk, nv.reshape(1, -1)])
            Y[ridx] = Xk[-len(ridx):]
            Y += 0.005 * rng.standard_normal(Y.shape)

        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        lr0 = float(rng.uniform(0.014, 0.030))
        lr1 = float(rng.uniform(0.0002, 0.0009))
        m0 = float(rng.uniform(0.025, 0.055))
        m1 = float(rng.uniform(0.0015, 0.004))
        steps = int(rng.integers(1400, 2200))
        if cycle in (1, 5):
            steps = int(steps * 1.25)
        Y = quench(Y, steps, lr0, lr1, m0, m1)
        Y = quench(Y, 350, 0.0006, 0.00015, 0.002, 0.001)
        v = max_cos(Y)
        if v < best_val - 1e-12:
            best = Y
            best_val = v
        if time.time() - t0 > 136:
            break

    remain = 178 - (time.time() - t0)
    if remain > 6:
        best = lp_polish(best, min(remain - 2, 28))
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

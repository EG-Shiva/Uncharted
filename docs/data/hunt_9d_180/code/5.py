import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = c_full.max()
        sel = c_full > m - max(20 * radius, 1e-9)
        # cap constraints to tightest 6000 for speed
        if np.count_nonzero(sel) > 6000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:6000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def adam_stage(x, alpha, steps, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        E = np.exp(alpha * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S == 0:
            break
        grad = (E @ x) * (n / S)
        nd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - nd * x
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        dn = (delta * x).sum(axis=1, keepdims=True)
        delta = delta - dn * x
        # clip per-point step to avoid blowup
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        clip = 0.05
        scale = np.minimum(1.0, clip / (rn + 1e-12))
        # scale is (n,1), broadcast
        delta = delta * scale
        x = x - delta
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def smooth_refine(x):
    x = adam_stage(x, 25.0, 300, 0.006)
    x = adam_stage(x, 70.0, 350, 0.004)
    x = adam_stage(x, 180.0, 450, 0.002)
    x = adam_stage(x, 450.0, 250, 0.0009)
    return x


def solve():
    np.random.seed(0)
    t0 = time.time()
    deadline = t0 + 176.0
    # load start
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = 180, 9
    rng = np.random.RandomState(0)
    hop_deadline = t0 + 144.0
    sigmas = [0.008, 0.02, 0.04, 0.08, 0.15]
    it = 0
    lp_calls = 0
    # quick initial snap to ensure feasible local optimum
    # (best already polished, skip)
    while time.time() < hop_deadline and it < 120:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        y = best.copy()
        mode = it % 7
        if mode == 0:
            # global shake
            y = best + sigma * rng.randn(n, d)
        elif mode == 6:
            # heavy re-init of few points (topology change)
            k = int(rng.randint(3, 7))
            # pick points with largest per-point max + random
            try:
                ppm = per_point_max(best)
                order = np.argsort(-ppm)
                # mix worst and random
                idx = list(order[:k // 2 + 1])
                while len(idx) < k:
                    c = int(rng.randint(0, n))
                    if c not in idx:
                        idx.append(c)
                idx = np.array(idx[:k])
            except Exception:
                idx = rng.choice(n, k, replace=False)
            y[idx] = rng.randn(k, d) * 1.0
            y += 0.005 * rng.randn(n, d)
        else:
            # subset shake biased to jammed points
            k = int(rng.randint(6, 24))
            try:
                ppm = per_point_max(best)
                # probability proportional to rank of ppm
                # take worst 60 + random
                order = np.argsort(-ppm)
                pool = order[:60]
                idx1 = pool[rng.choice(len(pool), min(k // 2, len(pool)), replace=False)]
                idx2 = rng.choice(n, k - len(idx1), replace=False)
                idx = np.unique(np.concatenate([idx1, idx2]))[:k]
            except Exception:
                idx = rng.choice(n, k, replace=False)
            y[idx] = y[idx] + sigma * 1.5 * rng.randn(len(idx), d)
            y = y + 0.002 * rng.randn(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth annealing
        try:
            y2 = smooth_refine(y)
        except Exception:
            continue
        try:
            v2 = max_cos(y2)
        except Exception:
            continue
        # decide to LP-polish
        if v2 < best_val + 0.0008 and lp_calls < 30:
            # only polish if promising and time left
            if v2 < best_val or (it % 3 == 0 and v2 < best_val + 0.0004):
                remain = hop_deadline - time.time()
                if remain < 3.0:
                    # save time for final polish
                    if v2 < best_val:
                        best = y2
                        best_val = v2
                    continue
                secs = 1.2 if v2 >= best_val else 2.0
                secs = min(secs, max(0.6, remain - 1.0))
                try:
                    y3 = lp_polish(y2, secs, radius_init=0.004)
                    v3 = max_cos(y3)
                    lp_calls += 1
                    if v3 < best_val:
                        best = y3
                        best_val = v3
                    elif v2 < best_val:
                        best = y2
                        best_val = v2
                except Exception:
                    if v2 < best_val:
                        best = y2
                        best_val = v2
                    continue
        else:
            if v2 < best_val:
                # keep smooth optimum even without LP for further hops
                # quick tiny LP if it is a new record to snap it
                best = y2
                best_val = v2
        if time.time() >= hop_deadline:
            break
    # final long polish
    remain = deadline - time.time() - 1.0
    if remain > 3.0:
        try:
            best = lp_polish(best, min(remain, 28.0), radius_init=1e-3)
            best_val = max_cos(best)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

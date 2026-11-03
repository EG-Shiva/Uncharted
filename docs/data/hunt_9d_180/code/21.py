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
        if np.count_nonzero(sel) > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:4000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
        p = len(I)
        if p == 0:
            break
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
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        clip = 0.05
        scale = np.minimum(1.0, clip / (rn + 1e-12))
        delta = delta * scale
        x = x - delta
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def fast_refine(x):
    x = adam_stage(x, 30.0, 120, 0.008)
    x = adam_stage(x, 120.0, 140, 0.004)
    x = adam_stage(x, 400.0, 140, 0.0015)
    return x

def void_replace(y, idx, rng, n_cand=4000):
    n, d = y.shape
    idx = np.atleast_1d(np.asarray(idx, dtype=int))
    keep = np.ones(n, dtype=bool)
    keep[idx] = False
    fixed = y[keep]
    cand = rng.randn(n_cand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ fixed.T
    worst = dots.max(axis=1)
    for k, j in enumerate(idx):
        # pick best candidate far from fixed + already placed replacements
        if k > 0:
            prev = y[idx[:k]]
            d2 = cand @ prev.T
            worst2 = np.maximum(worst, d2.max(axis=1))
        else:
            worst2 = worst
        b = int(np.argmin(worst2))
        y[j] = cand[b]
        # update worst to include this new point for next k
        nd = cand @ y[j]
        worst = np.maximum(worst, nd)
    return y

def solve():
    np.random.seed(1)
    t0 = time.time()
    deadline = t0 + 176.0
    hop_deadline = t0 + 138.0
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
        r0 = np.random.RandomState(1)
        best = r0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = 180, 9
    rng = np.random.RandomState(1)
    it = 0
    lp_calls = 0
    sigmas = [0.008, 0.02, 0.04, 0.08]
    while time.time() < hop_deadline and it < 600:
        it += 1
        if time.time() >= hop_deadline:
            break
        y = best.copy()
        mode = it % 5
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            ppm = G.max(axis=1)
            thr = best_val - 1e-7
            contacts = (G > thr).sum(axis=1)
            order_hi = np.argsort(-ppm)
            order_lo = np.argsort(ppm)
        except Exception:
            order_hi = rng.permutation(n)
            order_lo = order_hi[::-1]
            contacts = np.ones(n)
            ppm = np.ones(n) * best_val
        sigma = sigmas[it % len(sigmas)]
        if mode == 0:
            # void re-insertion: replace 1-2 loosest (rattlers) by largest-void points
            k = 1 if (it % 2 == 0) else 2
            # loosest = smallest ppm / fewest contacts
            cand_idx = order_lo[:k]
            y = void_replace(y, cand_idx, rng, n_cand=4000)
            y += 0.002 * rng.randn(n, d)
        elif mode == 1:
            # coherent kick of jammed cluster: pick overloaded neighborhood
            k = int(rng.randint(8, 16))
            pool = order_hi[:40]
            sel = pool[rng.choice(len(pool), min(k, len(pool)), replace=False)]
            u = rng.randn(d)
            u /= np.linalg.norm(u) + 1e-12
            s = sigma * 2.0
            y[sel] += s * u[None, :] + sigma * rng.randn(len(sel), d)
            # also jitter a couple rattlers
            y[order_lo[:2]] += 0.01 * rng.randn(2, d)
        elif mode == 2:
            # overloaded single/cluster shake with larger noise
            k = int(rng.randint(4, 10))
            pool = order_hi[:60]
            idx1 = pool[rng.choice(len(pool), min(k // 2, len(pool)), replace=False)]
            idx2 = rng.choice(n, k - len(idx1), replace=False)
            idx = np.unique(np.concatenate([idx1, idx2]))
            y[idx] += sigma * 1.8 * rng.randn(len(idx), d)
            y += 0.002 * rng.randn(n, d)
        elif mode == 3:
            # global small shake
            y = best + sigma * 0.6 * rng.randn(n, d)
        else:
            # random re-init of 2-4 worst + jitter
            k = int(rng.randint(2, 5))
            idx = order_hi[rng.choice(30, k, replace=False)]
            y[idx] = rng.randn(k, d)
            y += 0.003 * rng.randn(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        try:
            y2 = fast_refine(y)
        except Exception:
            continue
        try:
            v2 = max_cos(y2)
        except Exception:
            continue
        if v2 < best_val + 0.0012:
            do_lp = False
            if v2 < best_val:
                do_lp = True
            elif (it % 4 == 0) and v2 < best_val + 0.0004 and lp_calls < 22:
                do_lp = True
            if do_lp and lp_calls < 25:
                remain = hop_deadline - time.time()
                if remain < 2.5:
                    if v2 < best_val:
                        best = y2
                        best_val = v2
                    continue
                secs = 1.6 if v2 >= best_val else 2.2
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
            else:
                if v2 < best_val:
                    best = y2
                    best_val = v2
        if time.time() >= hop_deadline:
            break
    remain = deadline - time.time() - 0.8
    if remain > 3.0:
        try:
            best = lp_polish(best, min(remain, 26.0), radius_init=8e-4)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

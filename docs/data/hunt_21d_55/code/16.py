import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def lp_polish(x, seconds, r0=3e-3, max_it=40):
    x = norm_rows(x)
    n, d = x.shape
    best = max_cos(x)
    xb = x.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < max_it:
        it += 1
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        # cap LP size: keep tightest K to keep HiGHS fast
        if p > 500:
            k = 500
            idx = np.argpartition(-cs, k - 1)[:k]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            x, best, xb = y, v, y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def adam_subset(x, idx, target, temp, steps, lr, t_end):
    n, d = x.shape
    m = np.zeros((len(idx), d))
    vv = np.zeros((len(idx), d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > t_end:
            break
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        # soft-max weights
        D = (G - target) / temp
        dm = D.max()
        W = np.exp(D - dm)
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            break
        Grad = 2.0 * (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        g = Gt[idx]
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / sn)
        step = step * scale
        x[idx] = x[idx] - step
        x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True) + 1e-16
        if not np.all(np.isfinite(x)):
            break
    return x

def find_hole(Y, rng, n_cand=3000):
    d = Y.shape[1]
    cand = rng.randn(n_cand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-16
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def solve():
    t0 = time.time()
    DEADLINE = 168.0
    SEARCH_END = 150.0
    rng = np.random.RandomState(7)
    n, d = 55, 21
    cands = []
    try:
        cands.append(np.load("best.npy"))
    except Exception:
        pass
    try:
        cands.append(np.load("record.npy"))
    except Exception:
        pass
    best_val = 1e9
    x_best = None
    for c in cands:
        try:
            c = np.asarray(c, dtype=float)
            if c.shape != (55, 21):
                continue
            c = norm_rows(c)
            v = max_cos(c)
            if v < best_val:
                best_val = v
                x_best = c.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = norm_rows(rng.randn(n, d))
        best_val = max_cos(x_best)
    # quick initial squeeze
    if time.time() - t0 < 20:
        x_best = lp_polish(x_best, seconds=2.0, r0=1e-3)
        best_val = max_cos(x_best)
    it = 0
    # precompute triu for speed reuse in diagnostics
    while time.time() - t0 < SEARCH_END and it < 70:
        it += 1
        remain = SEARCH_END - (time.time() - t0)
        if remain < 2.0:
            break
        cand = x_best.copy()
        G = cand @ cand.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        cnt = (G > mx - 1e-3).sum(axis=1)
        rowmx = G.max(axis=1)
        mode = it % 6
        if mode == 0:
            # hole reinsertion of 1-2 most overloaded
            k = 1 if rng.rand() < 0.6 else 2
            victims = np.argsort(-cnt)[:k]
            if rng.rand() < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = cand[keep]
            for vi in victims:
                h = find_hole(Y, rng, n_cand=3000)
                cand[vi] = h
                Y = np.vstack([Y, h[None, :]])
            cand += 0.008 * rng.randn(n, d)
            cand = norm_rows(cand)
            idx = np.arange(n)
            tgt = best_val - (2e-4 + rng.rand() * 5e-4)
            temp = 10.0 ** rng.uniform(-3.0, -2.4)
            t_end = t0 + SEARCH_END - 1.5
            cand = adam_subset(cand, idx, tgt, temp, steps=350, lr=0.006 + rng.rand() * 0.008, t_end=t_end)
        elif mode == 1 or mode == 2:
            # subset shake of overloaded hubs
            k = 8 + int(rng.rand() * 5)  # 8-12
            idx = np.argsort(-cnt)[:k]
            # sometimes include worst-rowmax points
            if rng.rand() < 0.4:
                idx = np.argsort(-rowmx)[:k]
            sigma = 0.015 + rng.rand() * 0.03
            cand[idx] += sigma * rng.randn(k, d)
            cand[idx] = norm_rows(cand[idx])
            tgt = best_val - (2e-4 + rng.rand() * 6e-4)
            temp = 10.0 ** rng.uniform(-3.0, -2.3)
            t_end = t0 + SEARCH_END - 1.5
            steps = 300 + int(rng.rand() * 300)
            lr = 0.005 + rng.rand() * 0.010
            cand = adam_subset(cand, idx, tgt, temp, steps=steps, lr=lr, t_end=t_end)
        elif mode == 3:
            # global small kick + full soft-max
            sigma = 0.006 + rng.rand() * 0.010
            cand = norm_rows(cand + sigma * rng.randn(n, d))
            idx = np.arange(n)
            tgt = best_val - (1.5e-4 + rng.rand() * 4e-4)
            temp = 10.0 ** rng.uniform(-3.0, -2.4)
            t_end = t0 + SEARCH_END - 1.5
            cand = adam_subset(cand, idx, tgt, temp, steps=300, lr=0.006, t_end=t_end)
        elif mode == 4:
            # single-point hole move + local refine around it
            w = int(np.argmax(rowmx))
            keep = np.ones(n, dtype=bool); keep[w] = False
            h = find_hole(cand[keep], rng, n_cand=4000)
            # blend toward hole to keep some memory
            cand[w] = norm_rows(0.5 * cand[w][None, :] + 0.5 * h[None, :])[0]
            # neighbours most conflicting with w
            gw = (cand @ cand[w])
            gw[w] = -2
            idx = np.argsort(-gw)[:10]
            idx = np.union1d(idx, [w])
            tgt = best_val - (2e-4 + rng.rand() * 5e-4)
            temp = 10.0 ** rng.uniform(-2.9, -2.3)
            t_end = t0 + SEARCH_END - 1.5
            cand = adam_subset(cand, idx, tgt, temp, steps=350, lr=0.008, t_end=t_end)
        else:
            # medium global perturbation, no adam (pure LP jump)
            sigma = 0.02 + rng.rand() * 0.03
            cand = norm_rows(cand + sigma * rng.randn(n, d))
        # short LP squeeze
        rem2 = SEARCH_END - (time.time() - t0)
        if rem2 < 0.8:
            break
        secs = 1.4 if rem2 > 2.5 else max(0.6, rem2 * 0.5)
        cand = lp_polish(cand, seconds=secs, r0=0.005)
        v = max_cos(cand)
        if v < best_val - 1e-12:
            best_val = v
            x_best = cand.copy()
    # final polish
    rem = DEADLINE - (time.time() - t0)
    if rem > 1.2:
        x_best = lp_polish(x_best, seconds=min(rem - 0.6, 10.0), r0=1e-3)
    x_best = norm_rows(x_best)
    x_best[~np.isfinite(x_best)] = 0.01
    x_best = norm_rows(x_best)
    return x_best

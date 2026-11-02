import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def anneal_relax(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if S <= 0 or not np.isfinite(S):
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        # renormalize (inlined for speed)
        nr = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        X /= nr
    return X

def find_hole(Y, rng, n_cand=2000):
    d = Y.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def greedy_max_descent(x, iters, lr=0.005):
    x = norm_rows(x)
    best = max_cos(x)
    bx = x.copy()
    for it in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        # pairs near max
        mask = G > M - 5e-4
        # gradient of mean of near-max pairs
        # use weighting
        W = np.exp(800.0 * (G - M)) * mask
        np.fill_diagonal(W, 0.0)
        S = np.sum(W)
        if S <= 0 or not np.isfinite(S):
            break
        grad = (W @ x) / S
        proj = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - proj * x
        # decay lr
        cur_lr = lr / (1.0 + it / 500.0)
        y = x - cur_lr * gt
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            # increase a bit on success
            lr *= 1.02
        else:
            # try smaller step from x
            y2 = x - (cur_lr * 0.3) * gt
            y2 = norm_rows(y2)
            v2 = max_cos(y2)
            if v2 < best:
                best = v2
                bx = y2.copy()
                x = y2
            else:
                # accept sideways to keep moving (only if close)
                if v < best + 1e-6:
                    x = y
                lr *= 0.98
                if lr < 1e-5:
                    break
    return bx

def lp_polish(x, seconds, t0):
    x = norm_rows(x)
    n, d = x.shape
    best = max_cos(x)
    bx = x.copy()
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        # keep only top-K near-max to keep LP fast
        # threshold
        thr = m - max(10 * radius, 2e-7)
        sel = cfull > thr
        # cap to top 600
        if np.sum(sel) > 600:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:600]
            sel2 = np.zeros_like(sel)
            sel2[idx[order]] = True
            sel = sel2
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(1.0, seconds - (time.time() - t0))})
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
            best = v
            bx = y.copy()
            x = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # keep x at best for next linearization
        x = bx.copy() if best < max_cos(x) else x
        if time.time() - t0 >= seconds:
            break
    return bx

def solve():
    t_start = time.time()
    TIME_LIMIT = 170.0
    SEARCH_BUDGET = 132.0
    rng = np.random.default_rng(7)
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = norm_rows(a)
                s = max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = norm_rows(rng.standard_normal(size=(55, 21)))
        best_score = max_cos(best)
    cur = best.copy()
    cur_score = best_score
    n, d = 55, 21
    T = 0.0012
    trial = 0
    while time.time() - t_start < SEARCH_BUDGET:
        trial += 1
        if trial % 30 == 0:
            cur = best.copy()
            cur_score = best_score
        prop = cur.copy()
        mode = trial % 6
        if mode == 0 or mode == 1:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = np.max(G)
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 1 if mode == 0 else 2
            victims = np.argsort(-cnt)[:k]
            if rng.random() < 0.35:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = find_hole(Y, rng, n_cand=2000)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.008 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 800, 30.0, 500.0, 0.02, 0.002
        elif mode == 2:
            prop = prop + 0.05 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 700, 20.0, 400.0, 0.015, 0.002
        elif mode == 3:
            prop = prop + 0.20 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 900, 15.0, 400.0, 0.02, 0.002
        elif mode == 4:
            # worst-pair split: move one endpoint to hole
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            ij = np.argmax(G)
            ii, jj = divmod(ij, n)
            # pick the more loaded of the two
            cnt = np.sum(G > np.max(G) - 1e-3, axis=1)
            victim = ii if cnt[ii] >= cnt[jj] else jj
            mask = np.ones(n, dtype=bool); mask[victim] = False
            h = find_hole(prop[mask], rng, n_cand=2000)
            prop[victim] = h
            prop = prop + 0.01 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 800, 30.0, 500.0, 0.02, 0.002
        else:
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            for ii in idx:
                vv = rng.standard_normal(size=(d,))
                vv /= np.linalg.norm(vv)
                prop[ii] = vv
            prop = prop + 0.015 * rng.standard_normal(size=prop.shape)
            prop = norm_rows(prop)
            steps, a0, a1, lr0, lr1 = 800, 25.0, 500.0, 0.015, 0.002
        if time.time() - t_start > SEARCH_BUDGET - 8:
            steps = min(steps, 400)
        new = anneal_relax(prop, steps, a0, a1, lr0, lr1)
        s = max_cos(new)
        if s < best_score - 1e-12:
            best, best_score = new.copy(), s
            cur, cur_score = new.copy(), s
        elif s < cur_score:
            cur, cur_score = new.copy(), s
        else:
            if rng.random() < np.exp(-(s - cur_score) / T):
                cur, cur_score = new.copy(), s
        # anneal temperature slowly
        T *= 0.9995
        T = max(T, 0.0004)
        # time guard for final polish
        if time.time() - t_start >= SEARCH_BUDGET:
            break
    # final polish
    remain = TIME_LIMIT - (time.time() - t_start)
    if remain > 6:
        # short high-alpha smooth squeeze
        try:
            best = anneal_relax(best, 500, 80.0, 800.0, 0.008, 0.0008)
        except Exception:
            pass
    remain = TIME_LIMIT - (time.time() - t_start)
    if remain > 5:
        try:
            best = greedy_max_descent(best, iters=min(800, int(remain * 40)), lr=0.004)
        except Exception:
            pass
    remain = TIME_LIMIT - (time.time() - t_start)
    if remain > 4:
        try:
            best = lp_polish(best, min(remain - 1.0, 18.0), t0=time.time())
        except Exception:
            pass
    best = norm_rows(best)
    return best

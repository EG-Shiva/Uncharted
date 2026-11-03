import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_opt(x, steps, T0, T1, lr):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = float(lr)
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mval = float(np.max(G))
        if steps > 1:
            T = T0 * (T1 / T0) ** ((t - 1) / (steps - 1))
        else:
            T = T0
        # weights, <=1
        W = np.exp((G - mval) / T)
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Grad = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Grad
        v1 = b2 * v1 + (1 - b2) * (Grad * Grad)
        mhat = m1 / (1 - b1 ** t)
        vhat = v1 / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # safety clip per row
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.03 / rn)
        step = step * scale
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def greedy_reinsert(x_keep, k, rng, n_cand=15000, refine_steps=300, refine_sigma=0.02):
    d = x_keep.shape[1]
    placed = x_keep.copy()
    for _ in range(k):
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        scores = np.max(cand @ placed.T, axis=1)
        bi = int(np.argmin(scores))
        best = cand[bi].copy()
        best_score = float(scores[bi])
        for _ in range(refine_steps):
            prop = best + refine_sigma * rng.standard_normal(size=d)
            nrm = float(np.linalg.norm(prop))
            if nrm == 0.0 or not np.isfinite(nrm):
                continue
            prop /= nrm
            sc = float(np.max(placed @ prop))
            if sc < best_score:
                best = prop
                best_score = sc
        placed = np.vstack([placed, best.reshape(1, -1)])
    return placed


def solve():
    t_start = time.time()
    deadline = t_start + 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    # ensure shape (154,10), if not, tile/cut deterministically
    if xb.shape != (154, 10):
        # fallback: random補完
        tmp = rng.standard_normal(size=(154, 10))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m = min(n, 154)
        tmp[:m] = xb[:m, :10] if d >= 10 else np.concatenate([xb[:m], np.zeros((m, 10 - d))], axis=1)
        tmp[:m] /= np.linalg.norm(tmp[:m], axis=1, keepdims=True)
        xb = tmp
        n, d = 154, 10
    x_best = xb.copy()
    best_val = max_cos(x_best)

    trial = 0
    # reserve ~12s for final polish
    while True:
        trial += 1
        now = time.time()
        if now > deadline - 12.0:
            break
        if trial > 1000:
            break
        op = trial % 5
        try:
            if op == 1 or op == 2:
                sigma = 0.03 if op == 1 else 0.12
                y = x_best + sigma * rng.standard_normal(size=x_best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                steps = 600
                if deadline - 12.0 - time.time() < 30:
                    steps = 350
                y = smooth_opt(y, steps=steps, T0=0.025, T1=0.004, lr=0.004)
            elif op == 3 or op == 4:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                per = np.max(G, axis=1)
                order = np.argsort(-per)
                kk = 2 + ((trial // 5) % 4)  # 2..5
                pool = order[:12]
                idx_remove = np.sort(rng.choice(pool, size=kk, replace=False))
                mask = np.ones(n, dtype=bool)
                mask[idx_remove] = False
                x_keep = x_best[mask]
                y = greedy_reinsert(x_keep, int(kk), rng, n_cand=15000,
                                    refine_steps=300, refine_sigma=0.02)
                y = y + 0.01 * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = smooth_opt(y, steps=650, T0=0.02, T1=0.003, lr=0.004)
            else:  # op == 0 : rattler reinsertion
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                mval = float(np.max(G))
                cnt = np.sum(G > (mval - 5e-4), axis=1)
                ratt = np.where(cnt == 0)[0]
                if len(ratt) >= 2:
                    # cap to 11 to bound cost
                    if len(ratt) > 11:
                        ratt = np.sort(rng.choice(ratt, size=11, replace=False))
                    mask = np.ones(n, dtype=bool)
                    mask[ratt] = False
                    x_keep = x_best[mask]
                    y = greedy_reinsert(x_keep, len(ratt), rng, n_cand=12000,
                                        refine_steps=250, refine_sigma=0.025)
                    y = y + 0.008 * rng.standard_normal(size=y.shape)
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    y = smooth_opt(y, steps=600, T0=0.02, T1=0.0035, lr=0.004)
                else:
                    y = x_best + 0.05 * rng.standard_normal(size=x_best.shape)
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    y = smooth_opt(y, steps=500, T0=0.025, T1=0.004, lr=0.004)
        except Exception:
            continue
        try:
            v = max_cos(y)
        except Exception:
            continue
        # promising? allow small regression since LP can recover
        if v < best_val + 0.015:
            budget = min(4.0, (deadline - 12.0) - time.time())
            if budget > 1.2:
                try:
                    y2 = lp_polish(y, seconds=budget)
                    v2 = max_cos(y2)
                except Exception:
                    v2 = v
                    y2 = y
                if v2 < best_val - 1e-12:
                    x_best = y2
                    best_val = v2
                elif v < best_val - 1e-12:
                    # smooth alone beat best but LP did not
                    x_best = y / np.linalg.norm(y, axis=1, keepdims=True)
                    best_val = v
            else:
                if v < best_val - 1e-12:
                    x_best = y / np.linalg.norm(y, axis=1, keepdims=True)
                    best_val = v
        # early stop if excellent? keep going to use time

    remaining = deadline - time.time() - 0.5
    if remaining > 2.0:
        try:
            x_best = lp_polish(x_best, seconds=min(remaining, 10.0))
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # final safety: replace non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = xb
    return x_best

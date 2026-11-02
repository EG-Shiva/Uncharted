import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    xb = x.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best, xb = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    # return best found in this call (xb) but keep last x for chaining? return xb
    return xb


def adam_push(x, target, temp, steps, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        D = (G - target) / temp
        dm = D.max()
        W = np.exp(D - dm)
        # kill diagonal explicitly
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            break
        Grad = 2.0 * (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # clip extreme steps for stability
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = 0.05
        scale = np.minimum(1.0, clip / (sn + 1e-12))
        step = step * scale
        x = x - step
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
        if not np.all(np.isfinite(x)):
            break
    return x


def row_worst_idx(x, k):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return np.argsort(-rm)[:k]


def solve():
    t0 = time.time()
    DEADLINE = 172.0  # leave margin under 180
    rng = np.random.RandomState(0)
    # load incumbents
    cands = []
    try:
        cands.append(np.load("best.npy"))
    except Exception:
        pass
    try:
        cands.append(np.load("record.npy"))
    except Exception:
        pass
    if len(cands) == 0:
        cands.append(rng.randn(55, 21))
    best_val = 1e9
    x_best = None
    for c in cands:
        c = np.asarray(c, dtype=float)
        if c.shape != (55, 21):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    if x_best is None:
        x_best = rng.randn(55, 21)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = max_cos(x_best)
    n, d = 55, 21

    # tiny initial squeeze (already jammed, don't waste time)
    if time.time() - t0 < DEADLINE - 5:
        x_best = lp_polish(x_best, seconds=1.5, r0=1e-3)
        best_val = max_cos(x_best)

    it = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.015, 0.035, 0.07, 0.005]
    while time.time() - t0 < DEADLINE - 8:
        it += 1
        sigma = sigmas[it % len(sigmas)] * (0.7 + 0.6 * rng.rand())
        cand = x_best + sigma * rng.randn(n, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)

        # every few trials: remove-and-reinsert worst points
        if it % 4 == 0:
            k = 1 + int(rng.rand() * 2)  # 1-2 points
            worst = row_worst_idx(cand, k)
            for w in worst:
                v = rng.randn(d)
                v /= np.linalg.norm(v)
                cand[w] = v
        elif it % 4 == 1:
            # targeted extra shake on overloaded points
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            thr = G.max() - 2e-3
            cnt = (G > thr).sum(axis=1)
            overload = np.argsort(-cnt)[:6]
            cand[overload] += (sigma * 1.5) * rng.randn(len(overload), d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)

        # annealed soft-max push below current best
        if it % 3 != 2:
            delta = 2e-4 + rng.rand() * 6e-4
            target = best_val - delta
            ltemp = rng.uniform(-3.0, -2.3)  # 0.001 - 0.005
            temp = 10.0 ** ltemp
            steps = 700 + int(rng.rand() * 900)
            lr = 0.004 + rng.rand() * 0.012
            # don't exceed time
            cand = adam_push(cand, target, temp, steps, lr)

        # short LP squeeze to nearest optimum
        remain = DEADLINE - 8 - (time.time() - t0)
        if remain < 0.5:
            break
        secs = 2.0 if remain > 3 else max(0.5, remain * 0.5)
        cand = lp_polish(cand, seconds=secs, r0=0.006)
        v = max_cos(cand)
        if v < best_val - 1e-12:
            best_val = v
            x_best = cand.copy()
            # extra squeeze of new basin if time permits
            remain2 = DEADLINE - 8 - (time.time() - t0)
            if remain2 > 2.0:
                sq = lp_polish(x_best, seconds=min(2.5, remain2 * 0.4), r0=1e-3)
                vs = max_cos(sq)
                if vs < best_val:
                    best_val = vs
                    x_best = sq
        # opportunistic: keep time for final polish
        if it > 120:
            break

    # final high-precision polish with remaining time
    rem = DEADLINE - (time.time() - t0)
    if rem > 1.0:
        x_best = lp_polish(x_best, seconds=min(rem - 0.5, 12.0), r0=1e-3)
        best_val = max_cos(x_best)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # shrink margin not needed; ensure finite
    x_best[~np.isfinite(x_best)] = 0.01
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

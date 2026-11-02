import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn = np.maximum(yn, 1e-12)
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def softmax_polish(X0, steps=800, T0=0.02, T1=0.004, lr0=0.01):
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, steps + 1):
        t = it / steps
        T = T0 * (T1 / T0) ** t
        lr = lr0 * (1.0 - 0.5 * t)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        amax = float(G.max())
        # scaled exp weights, focus on largest dots
        W = np.exp((G - amax) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * gt
        v1 = b2 * v1 + (1.0 - b2) * (gt * gt)
        mhat = m1 / (1.0 - b1 ** it)
        vhat = v1 / (1.0 - b2 ** it)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        stept = step - sdot * X
        # clip extreme steps
        sn = np.linalg.norm(stept, axis=1, keepdims=True)
        cap = 0.05
        scale = np.minimum(1.0, cap / (sn + 1e-12))
        stept = stept * scale
        X = X - stept
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = xb
    xb = np.asarray(xb, dtype=float)
    xr = np.asarray(xr, dtype=float)
    xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    xr = xr / np.maximum(np.linalg.norm(xr, axis=1, keepdims=True), 1e-12)
    vb = max_cos(xb)
    vr = max_cos(xr)
    if vr < vb:
        best, best_val = xr, vr
    else:
        best, best_val = xb, vb
    n, d = best.shape
    # safety: ensure shape (163,4)
    it = 0
    while time.time() < deadline - 12.0:
        it += 1
        # time left guard
        if time.time() > deadline - 12.0:
            break
        cand = best.copy()
        # find worst points
        G = cand @ cand.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        worst = np.argsort(rowmax)[-8:]
        mode = it % 5
        if mode == 1:
            cand = cand + 0.02 * rng.standard_normal(cand.shape)
        elif mode == 2:
            k = 5
            idx = rng.choice(worst, size=k, replace=False)
            cand[idx] = cand[idx] + 0.15 * rng.standard_normal((k, d))
        elif mode == 3:
            k = 3
            idx = rng.choice(n, size=k, replace=False)
            z = rng.standard_normal((k, d))
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            cand[idx] = z
        elif mode == 4:
            # largest-hole reinsertion of single worst point
            w = int(worst[-1])
            S = 3000
            Sp = rng.standard_normal((S, d))
            Sp = Sp / np.maximum(np.linalg.norm(Sp, axis=1, keepdims=True), 1e-12)
            M = Sp @ cand.T
            M[:, w] = -2.0
            mh = M.max(axis=1)
            bh = int(np.argmin(mh))
            cand[w] = Sp[bh]
            cand = cand + 0.008 * rng.standard_normal(cand.shape)
        else:
            cand = cand + 0.05 * rng.standard_normal(cand.shape)
        cand = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        # smooth repulsion to change basin
        steps = 700 if (it % 2 == 0) else 1000
        cand_s = softmax_polish(cand, steps=steps, T0=0.025, T1=0.004, lr0=0.012)
        v = max_cos(cand_s)
        # only LP-polish promising candidates to save time
        if v < best_val + 0.003:
            remain = deadline - 12.0 - time.time()
            if remain < 0.5:
                # no time for LP, just check direct
                if v < best_val - 1e-12:
                    best, best_val = cand_s, v
                continue
            lp_time = min(1.4, max(0.6, remain / 8.0))
            cand_p = lp_polish(cand_s, seconds=lp_time)
            v2 = max_cos(cand_p)
            if v2 < best_val - 1e-12:
                best, best_val = cand_p, v2
        # opportunistic direct update without LP
        elif v < best_val - 1e-12:
            best, best_val = cand_s, v
    # final long polish of champion
    rem = deadline - time.time() - 1.0
    if rem > 1.0:
        best = lp_polish(best, seconds=min(rem, 10.0))
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best

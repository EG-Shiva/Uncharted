import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_quench(X, steps, lr0, lr1, T0, T1, iu0, iu1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1p = 1.0
    b2p = 1.0
    lr_ratio = lr1 / lr0
    T_ratio = T1 / T0
    for t in range(1, steps + 1):
        frac = t / steps
        lr = lr0 * (lr_ratio ** frac)
        TT = T0 * (T_ratio ** frac)
        G = X @ X.T
        g = G[iu0, iu1]
        mx = g.max()
        e = np.exp((g - mx) / TT)
        s = e.sum()
        e /= s
        W = np.zeros((n, n), dtype=X.dtype)
        W[iu0, iu1] = e
        W[iu1, iu0] = e
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad -= dot * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * grad
        v *= b2
        v += (1.0 - b2) * (grad * grad)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X -= step
        nr = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
    return X

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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = 175.0
    np.random.seed(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float)
    # safety: shape fix
    if xb.shape != (54, 21):
        # try to reshape/crop if needed
        r = np.random.randn(54, 21)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        # fill as much as possible
        m = min(xb.shape[0], 54)
        d = min(xb.shape[1], 21)
        r[:m, :d] = xb[:m, :d]
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        xb = r
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 54, 21
    iu0, iu1 = np.triu_indices(n, 1)
    best = max_cos(xb)
    x_best = xb.copy()

    # helper to get worst points
    def worst_points(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        idx = np.argsort(-mx)[:k]
        return idx

    trial = 0
    # hopping phase until ~115s
    HOP_END = t0 + 115.0
    # include a few pure random quenches first for diversity
    while time.time() < HOP_END and time.time() - t0 < HARD - 50:
        trial += 1
        rtype = np.random.rand()
        if trial <= 6 and trial % 2 == 0:
            # pure random start
            x_try = np.random.randn(n, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif rtype < 0.45:
            sigma = float(np.random.choice([0.005, 0.01, 0.02, 0.04, 0.08]))
            x_try = x_best + np.random.randn(n, d) * sigma
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif rtype < 0.85:
            k = int(np.random.choice([1, 2, 3]))
            x_try = x_best.copy()
            if np.random.rand() < 0.6:
                idx = worst_points(x_best, k)
            else:
                idx = np.random.choice(n, k, replace=False)
            rnd = np.random.randn(k, d)
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            # small shake of rest to break contacts
            x_try += np.random.randn(n, d) * 0.004
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            sigma = 0.12 + 0.05 * np.random.rand()
            x_try = x_best + np.random.randn(n, d) * sigma
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        steps = int(np.random.choice([1500, 2500, 3500]))
        # adapt steps to remaining time
        if time.time() + steps * 0.00012 > HOP_END:
            steps = max(800, int((HOP_END - time.time()) / 0.00012))
            if steps < 800:
                break
        T0 = float(np.random.choice([0.008, 0.01, 0.015]))
        T1 = float(np.random.choice([0.0005, 0.001, 0.002]))
        lr0 = float(np.random.choice([0.005, 0.01, 0.02]))
        lr1 = lr0 * 0.1
        x_q = adam_quench(x_try.copy(), steps, lr0, lr1, T0, T1, iu0, iu1)
        v = max_cos(x_q)
        if v < best:
            best = v
            x_best = x_q.copy()
            # quick squeeze with short LP if time allows
            if time.time() < HOP_END - 5:
                x_p = lp_polish(x_best, 1.5)
                vp = max_cos(x_p)
                if vp < best:
                    best = vp
                    x_best = x_p

    # medium refinement with low T Adam
    if time.time() - t0 < HARD - 35:
        try:
            x_ref = adam_quench(x_best.copy(), 2500, 0.006, 0.0006, 0.003, 0.00025, iu0, iu1)
            vr = max_cos(x_ref)
            if vr < best:
                best = vr
                x_best = x_ref
        except Exception:
            pass

    # final LP polish with remaining time
    remain = HARD - (time.time() - t0) - 2.0
    if remain > 5:
        remain = min(remain, 45.0)
        try:
            x_best = lp_polish(x_best, remain)
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    x_best /= nr
    return x_best

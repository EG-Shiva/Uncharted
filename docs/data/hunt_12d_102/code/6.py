import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_refine(x, steps, beta0, beta1, lr):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x = x.copy()
    best_v = max_cos(x)
    for t in range(1, steps + 1):
        beta = beta0 + (beta1 - beta0) * (t / steps)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        W = np.exp(beta * (G - mval))
        np.fill_diagonal(W, 0.0)
        s = W.sum()
        if s <= 0 or not np.isfinite(s):
            break
        W /= s
        grad = (W @ x) * 0.5
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        m1h = m1 / (1.0 - b1 ** t)
        m2h = m2 / (1.0 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1e-12
        x = x / nr
        if (t % 50 == 0) or (t == steps):
            v = max_cos(x)
            if v < best_v:
                best_v = v
                best_x = x.copy()
    return best_x


def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            k = 2500
            idx = np.argpartition(c, len(c) - k)[len(c) - k:]
            sel = np.zeros_like(sel)
            sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(nr)):
            radius *= 0.5
            continue
        nr[nr < 1e-12] = 1e-12
        y = y / nr
        v = max_cos(y)
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def solve():
    np.random.seed(0)
    t0 = time.time()
    rng = np.random.RandomState(12345)
    n, d = 102, 12
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = rng.randn(n, d)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        # try to reshape/transpose defensively
        if xb.size == n * d:
            xb = xb.reshape(n, d)
        else:
            xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (n, d):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                xb = xr
                best_val = vr
    except Exception:
        pass
    x_best = xb.copy()
    sigmas = [0.04, 0.025, 0.015, 0.008, 0.004, 0.002]
    trial = 0
    # exploration phase, reserve time for final polish
    loop_end = t0 + 150.0
    while time.time() < loop_end:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        Gtmp = x_best @ x_best.T
        np.fill_diagonal(Gtmp, -2.0)
        mtmp = float(Gtmp.max())
        cnt = (Gtmp > mtmp - 1e-9).sum(axis=1)
        x_try = x_best + sigma * rng.randn(n, d)
        if trial % 2 == 0:
            k = 6 + (trial % 5)
            idx = np.argsort(-cnt)[:k]
            x_try[idx] += (sigma * 2.0) * rng.randn(k, d)
        if trial % 8 == 0:
            worst = int(np.argmax(cnt))
            x_try[worst] = rng.randn(d)
        x_try = x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
        steps = 400
        beta0 = 25 + (trial % 4) * 10
        beta1 = 150 + (trial % 3) * 30
        lr = 0.006 + 0.004 * ((trial * 7) % 5) / 4.0
        x_ref = adam_refine(x_try, steps, beta0, beta1, lr)
        remain = loop_end - time.time()
        if remain < 0.4:
            vr = max_cos(x_ref)
            if vr < best_val - 1e-12:
                x_best = x_ref
                best_val = vr
            break
        secs = min(2.0, remain)
        r0 = min(5e-3, max(1e-3, sigma * 0.3))
        x_pol = lp_polish(x_ref, seconds=secs, r0=r0)
        v = max_cos(x_pol)
        if v < best_val - 1e-12:
            x_best = x_pol.copy()
            best_val = v
        else:
            vr = max_cos(x_ref)
            if vr < best_val - 1e-12 and vr < v:
                x_best = x_ref.copy()
                best_val = vr
        if time.time() > t0 + 172.0:
            break
    secs_final = max(0.5, (t0 + 175.0) - time.time())
    # cap to stay within 180s
    secs_final = min(secs_final, 25.0)
    try:
        x_best = lp_polish(x_best, seconds=secs_final, r0=1e-3)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

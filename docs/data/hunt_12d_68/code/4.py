import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    """Trust-region LP polish, time-limited."""
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
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def adam_refine(x, steps, k0, k1, lr):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute growth factor
    for t in range(1, steps + 1):
        frac = t / steps
        k = k0 * (k1 / k0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        # exp in stable way
        E = np.exp(k * (G - m0))
        # zero diag (was exp(k*(-2-m0)) tiny but remove)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ x) / S
        dot = (Gr * x).sum(axis=1, keepdims=True)
        Gt = Gr - dot * x
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1 ** t)
        vhat = v1 / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # safety clip per-row
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        clip = 0.02
        scale = np.ones_like(sn)
        over = sn > clip
        scale[over] = clip / sn[over]
        step = step * scale
        x = x - step
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def solve():
    t_start = time.time()
    BUDGET = 165.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(68, 12))
    xb = np.asarray(xb, dtype=np.float64)
    # ensure shape (68,12)
    if xb.shape != (68, 12):
        # try to reshape / truncate / pad deterministically
        flat = xb.ravel()
        need = 68 * 12
        if flat.size < need:
            extra = rng.normal(size=(need - flat.size,))
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(68, 12)
    xn = np.linalg.norm(xb, axis=1, keepdims=True)
    xn[xn == 0] = 1.0
    xb = xb / xn
    best = max_cos(xb)
    xbest = xb.copy()

    trial = 0
    sigmas = [0.008, 0.015, 0.025, 0.005, 0.04]
    while True:
        now = time.time() - t_start
        remain = BUDGET - now
        if remain < 3.0:
            break
        trial += 1

        # choose perturbation mode
        mode = trial % 6
        x = xbest.copy()
        if mode == 0:
            # target overloaded points: re-init 2 worst + small noise
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cont = (G > m - 0.005).sum(axis=1)
            idx = np.argsort(cont)[-2:]
            R = rng.normal(size=(2, 12))
            R = R / np.linalg.norm(R, axis=1, keepdims=True)
            x[idx] = R
            x = x + 0.005 * rng.normal(size=x.shape)
        elif mode == 3:
            # re-init 1 worst + medium shake
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cont = (G > m - 0.003).sum(axis=1)
            idx = int(np.argmax(cont))
            R = rng.normal(size=(12,))
            R = R / np.linalg.norm(R)
            x[idx] = R
            x = x + 0.008 * rng.normal(size=x.shape)
        elif mode == 5:
            # random subset shake harder
            kdrop = 6
            sel = rng.choice(68, size=kdrop, replace=False)
            x[sel] = x[sel] + 0.08 * rng.normal(size=(kdrop, 12))
        else:
            sigma = sigmas[trial % len(sigmas)]
            x = x + sigma * rng.normal(size=x.shape)

        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn

        # annealed smooth params vary by trial
        if mode == 0:
            steps, k0, k1, lr = 1500, 80.0, 700.0, 0.004
        elif mode == 3:
            steps, k0, k1, lr = 1200, 60.0, 500.0, 0.003
        else:
            # cycle lr/k1
            k1c = [600.0, 900.0, 400.0][trial % 3]
            lrc = [0.003, 0.005, 0.002][(trial // 3) % 3]
            steps = 900 if remain < 60 else 1300
            k0, k1, lr = 70.0, k1c, lrc

        # shorten if little time left
        if remain < 25:
            steps = min(steps, 600)

        x = adam_refine(x, steps, k0, k1, lr)
        v = max_cos(x)

        # only LP-polish if promising (avoid wasting time)
        thresh = best + 0.015
        if v < thresh:
            pt = min(4.0, max(0.8, remain * 0.25))
            # don't exceed budget
            if time.time() - t_start + pt > BUDGET + 5:
                pt = max(0.5, BUDGET - (time.time() - t_start) - 1.0)
            x2 = lp_polish(x, pt)
            v2 = max_cos(x2)
            if v2 < best - 1e-13:
                best = v2
                xbest = x2.copy()
            elif v < best - 1e-13:
                best = v
                xbest = x.copy()
        else:
            if v < best - 1e-13:
                best = v
                xbest = x.copy()

        # occasional direct LP on shaken point without smooth (cheap basin hop)
        if trial % 10 == 9 and remain > 20:
            xs = xbest + 0.003 * rng.normal(size=xbest.shape)
            xs = xs / np.linalg.norm(xs, axis=1, keepdims=True)
            xs2 = lp_polish(xs, 1.5)
            vs = max_cos(xs2)
            if vs < best - 1e-13:
                best = vs
                xbest = xs2.copy()

    # final polish with leftover time (up to ~10s but respect 178s total)
    elapsed = time.time() - t_start
    left = 176.0 - elapsed
    if left > 1.0:
        xbest = lp_polish(xbest, min(left, 12.0))

    xbest = xbest / np.linalg.norm(xbest, axis=1, keepdims=True)
    return xbest

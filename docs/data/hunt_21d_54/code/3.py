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
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def adam_smooth(X_init, steps, K0, K1, lr):
    X = X_init.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        K = K0 * (K1 / K0) ** frac
        lr_t = lr * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        gmax = G.max()
        A = np.exp(K * (G - gmax))
        np.fill_diagonal(A, 0.0)
        S = A.sum()
        if not np.isfinite(S) or S <= 0:
            break
        P = A / S
        grad = 2.0 * (P @ X)
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * vv + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        step = lr_t * mhat / (np.sqrt(vhat) + eps)
        # clip extreme steps
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / (sn + 1e-12))
        step = step * clip
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = np.asarray(xb, dtype=np.float64)
    # ensure shape (54,21)
    if xb.shape != (54, 21):
        # try to reshape/truncate deterministically
        flat = xb.ravel()
        need = 54 * 21
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(54, 21)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = xb
    best_val = max_cos(best)

    # short squeeze of starting basin
    best = lp_polish(best, 8)
    best_val = max_cos(best)

    sigmas = [0.02, 0.05, 0.09, 0.15, 0.25, 0.35, 0.07, 0.12]
    K1s = [400.0, 600.0, 900.0, 1300.0]
    lrs = [0.003, 0.005, 0.008]

    n_iter = 28
    for it in range(n_iter):
        if time.time() - t0 > 95:
            break
        sigma = sigmas[it % len(sigmas)]
        # find overloaded points in current best
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m0 = G.max()
        cnt = (G > m0 - 0.002).sum(axis=1)
        order = np.argsort(cnt)
        # candidate from best
        Y = best.copy()
        if it % 5 == 4:
            # single-point reinsertion of worst point
            worst = int(order[-1])
            Y[worst] = rng.randn(21)
            Y[worst] /= np.linalg.norm(Y[worst]) + 1e-12
            Y += 0.02 * rng.randn(*Y.shape)
        else:
            noise = rng.randn(*Y.shape)
            Y = Y + sigma * noise
            # extra kick to top-6 overloaded hubs
            big = order[-6:]
            Y[big] += (sigma * 1.0) * rng.randn(len(big), 21)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        K1 = K1s[it % len(K1s)]
        lr = lrs[(it // 2) % len(lrs)]
        steps = 1800
        Y2 = adam_smooth(Y, steps, 80.0, K1, lr)
        v = max_cos(Y2)
        if v < best_val:
            # verify with a short LP polish to get to bottom of basin
            Y3 = lp_polish(Y2, 4)
            v3 = max_cos(Y3)
            if v3 < best_val:
                best, best_val = Y3, v3
            elif v < best_val:
                best, best_val = Y2, v
        # also try a pure random-restart smooth candidate every 7th iter
        if it % 7 == 6 and time.time() - t0 < 90:
            Yr = rng.randn(54, 21)
            Yr /= np.linalg.norm(Yr, axis=1, keepdims=True)
            # seed it near best by averaging (basin bridge)
            Yr = 0.5 * Yr + 0.5 * (best + 0.3 * rng.randn(*best.shape))
            Yr /= np.linalg.norm(Yr, axis=1, keepdims=True)
            Yr2 = adam_smooth(Yr, 1500, 60.0, 800.0, 0.005)
            vv = max_cos(Yr2)
            if vv < best_val:
                Yr3 = lp_polish(Yr2, 4)
                vv3 = max_cos(Yr3)
                if vv3 < best_val:
                    best, best_val = Yr3, vv3

    remain = 172 - (time.time() - t0)
    if remain > 8:
        # leave 2s buffer
        best = lp_polish(best, remain - 2)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

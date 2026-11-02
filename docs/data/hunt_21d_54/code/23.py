import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _adam_smooth(X_init, steps, K0, K1, lr):
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
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / sn)
        step = step * clip
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_all = (x @ x.T)[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        cs = c_all[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1800:
            idx = np.argpartition(-cs, 1800)[:1800]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1800
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                      np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (54, 21):
        flat = xb.ravel()
        need = 54 * 21
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(54, 21)
    xb = _proj(xb)
    try:
        xr = _proj(np.asarray(np.load("record.npy"), dtype=np.float64))
        if xr.shape == (54, 21):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr
    except Exception:
        pass

    best = xb.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n, d = 54, 21

    sigmas = [0.02, 0.05, 0.09, 0.15, 0.25, 0.12, 0.07, 0.35]
    K1s = [600.0, 900.0, 1300.0, 500.0]
    lrs = [0.005, 0.008, 0.003]

    # hopping phase, reserve time for final polish
    max_iter = 300
    for it in range(max_iter):
        if time.time() - t0 > 132:
            break
        mode = it % 6
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        m0 = float(Gc.max())
        cnt = (Gc > m0 - 0.003).sum(axis=1)
        order = np.argsort(cnt)

        Y = cur.copy()
        if mode == 0:
            sigma = sigmas[(it // 6) % len(sigmas)]
            Y = Y + sigma * rng.randn(*Y.shape)
        elif mode == 1:
            big = order[-6:]
            Y[big] += 0.15 * rng.randn(6, d)
            Y += 0.02 * rng.randn(*Y.shape)
        elif mode == 2:
            worst = int(order[-1])
            Y[worst] = rng.randn(d)
            Y += 0.02 * rng.randn(*Y.shape)
        elif mode == 3:
            # best-void reinsertion of a bad point
            if (it % 2) == 0:
                worst = int(order[-1])
            else:
                worst = int(rng.choice(order[-5:]))
            K = 2000
            C = rng.randn(K, d)
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
            D = C @ Y.T
            D[:, worst] = -2.0
            md = D.max(axis=1)
            bi = int(np.argmin(md))
            Y[worst] = C[bi]
            Y += 0.01 * rng.randn(*Y.shape)
        elif mode == 4:
            w2 = order[-2:]
            for w in w2:
                Y[int(w)] = rng.randn(d)
            Y += 0.02 * rng.randn(*Y.shape)
        else:
            subset = rng.choice(n, 8, replace=False)
            Y[subset] += 0.20 * rng.randn(8, d)
            Y += 0.02 * rng.randn(*Y.shape)

        Y = _proj(Y)
        K1 = K1s[it % len(K1s)]
        lr = lrs[(it // 2) % len(lrs)]
        steps = 1000 if mode in (0, 1, 5) else 1200
        # shorten steps when time is short
        if time.time() - t0 > 110:
            steps = 700
        Y2 = _adam_smooth(Y, steps, 80.0, K1, lr)
        v = _max_cos(Y2)
        if v < best_val - 1e-12:
            best = Y2.copy()
            best_val = v
            cur = Y2.copy()
            cur_val = v
        else:
            elapsed = time.time() - t0
            tol = 0.003 * max(0.0, 1.0 - elapsed / 132.0)
            if v < cur_val + tol:
                cur = Y2.copy()
                cur_val = v
        if it % 60 == 59:
            # pull walk back to best region if drifting
            if cur_val > best_val + 0.004:
                cur = best.copy()
                cur_val = best_val

    remain = 172.0 - (time.time() - t0)
    if remain > 6:
        best = _lp_polish(best, remain - 2.0)
    best = _proj(best)
    return best

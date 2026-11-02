import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    # keep best copy
    x_best = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
            x, best, x_best, radius = y, v, y.copy(), min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x_best


def smooth_polish(x, steps=1000, beta0=60.0, beta1=600.0, lr=0.012):
    n, d = x.shape
    X = x / np.linalg.norm(x, axis=1, keepdims=True)
    m = np.zeros_like(X)
    s = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    ratio = beta1 / max(beta0, 1e-12)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = beta0 * (ratio ** frac)
        lr_t = lr * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        M = np.max(G)
        W = np.exp(beta * (G - M))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True)
        Avg = (W @ X) / (row + 1e-12)
        dot = np.sum(Avg * X, axis=1, keepdims=True)
        Gt = Avg - dot * X
        m = b1 * m + (1.0 - b1) * Gt
        s = b2 * s + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1 ** t)
        shat = s / (1.0 - b2 ** t)
        upd = mhat / (np.sqrt(shat) + eps)
        rdot = np.sum(upd * X, axis=1, keepdims=True)
        upd_t = upd - rdot * X
        X = X - lr_t * upd_t
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    T_HARD = 165.0
    T_SEARCH = 138.0
    rng = np.random.default_rng(0)
    n, d = 54, 23
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best = float(vals[bi])

    sigmas = np.array([0.008, 0.015, 0.03, 0.06, 0.10])
    ks_sub = np.array([2, 3, 4, 6, 8])
    ks_worst = np.array([2, 3, 5])
    steps_arr = np.array([600, 900, 1300])
    b0_arr = np.array([40.0, 60.0, 80.0])
    b1_arr = np.array([400.0, 700.0, 1100.0])
    lr_arr = np.array([0.008, 0.012, 0.018])

    trial = 0
    while time.time() - t0 < T_SEARCH:
        trial += 1
        if time.time() - t0 > T_SEARCH - 2.0:
            break
        u = rng.random()
        if u < 0.35:
            sigma = float(rng.choice(sigmas))
            x0 = x_best + sigma * rng.standard_normal((n, d))
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
        elif u < 0.70:
            k = int(rng.choice(ks_sub))
            idx = rng.choice(n, size=k, replace=False)
            x0 = x_best + 0.008 * rng.standard_normal((n, d))
            rnew = rng.standard_normal((k, d))
            rnew = rnew / np.linalg.norm(rnew, axis=1, keepdims=True)
            x0[idx] = rnew
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -10.0)
            per_max = G.max(axis=1)
            k = int(rng.choice(ks_worst))
            idx = np.argsort(per_max)[-k:]
            x0 = x_best + 0.008 * rng.standard_normal((n, d))
            rnew = rng.standard_normal((k, d))
            rnew = rnew / np.linalg.norm(rnew, axis=1, keepdims=True)
            x0[idx] = rnew
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)

        steps = int(rng.choice(steps_arr))
        beta0 = float(rng.choice(b0_arr))
        beta1 = float(rng.choice(b1_arr))
        lr = float(rng.choice(lr_arr))

        # adjust steps if little time left to allow more trials
        time_left_search = T_SEARCH - (time.time() - t0)
        if time_left_search < 10 and steps > 600:
            steps = 600

        x1 = smooth_polish(x0, steps=steps, beta0=beta0, beta1=beta1, lr=lr)
        v1 = max_cos(x1)
        if v1 > best + 0.012:
            continue
        time_left = T_SEARCH - (time.time() - t0)
        if time_left < 0.5:
            break
        lp_t = min(1.5, time_left)
        x2 = lp_polish(x1, seconds=lp_t)
        v2 = max_cos(x2)
        if v2 < best - 1e-12:
            best = v2
            x_best = x2.copy()

    time_left = T_HARD - (time.time() - t0)
    if time_left > 3.0:
        x_best = lp_polish(x_best, seconds=min(time_left - 1.0, 20.0))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    if seconds <= 0:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time left check
        if time.time() - start >= seconds:
            break
        try:
            Gfull = x @ x.T
            c = Gfull[iu0]
        except Exception:
            break
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
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
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            if not np.all(np.isfinite(y)):
                radius *= 0.5
                continue
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_descent(x, steps=120, beta0=20.0, beta1=90.0, lr0=0.002):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        nrm = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        if not np.all(np.isfinite(grad)):
            break
        step = lr0 * (1.0 - 0.5 * frac)
        x = x - step * grad / nrm
        nrm2 = np.linalg.norm(x, axis=1, keepdims=True)
        nrm2[nrm2 == 0] = 1.0
        x = x / nrm2
    return x


def solve():
    t0 = time.time()
    HOP_END = t0 + 148.0
    HARD_END = t0 + 174.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    # load start
    xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d) and np.all(np.isfinite(v)):
                xb = v
                break
        except Exception:
            continue
    if xb is None:
        xb = rng.standard_normal((n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)

    def contacts_of(x, tol=2e-7):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu = np.triu_indices(n, 1)
        c = G[iu]
        sel = c > m - tol
        I = iu[0][sel]
        J = iu[1][sel]
        rowmax = G.max(axis=1)
        ratt = np.where(rowmax < m - tol)[0]
        return m, I, J, ratt

    try:
        _, cI, cJ, rattlers = contacts_of(xb)
    except Exception:
        cI, cJ, rattlers = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    trial = 0
    # hopping loop
    while time.time() < HOP_END:
        time_left = HOP_END - time.time()
        if time_left < 1.2:
            break
        mod = trial % 6
        y = xb.copy()
        try:
            if mod == 0:
                sig = 0.008
                y = y + sig * rng.standard_normal((n, d))
            elif mod == 1:
                sig = 0.020
                y = y + sig * rng.standard_normal((n, d))
            elif mod == 2:
                # subset shake: most small, few large
                y = y + 0.003 * rng.standard_normal((n, d))
                k = 12
                idx = rng.choice(n, size=k, replace=False)
                y[idx] = y[idx] + 0.06 * rng.standard_normal((k, d))
            elif mod == 3:
                # contact-breaking kick
                y = y + 0.004 * rng.standard_normal((n, d))
                if len(cI) > 0:
                    e = rng.integers(0, len(cI))
                    i, j = int(cI[e]), int(cJ[e])
                    diff = y[i] - y[j]
                    nd = np.linalg.norm(diff) + 1e-12
                    diff = diff / nd
                    push = 0.04 + 0.02 * rng.random()
                    y[i] = y[i] + push * diff
                    y[j] = y[j] - push * diff
                else:
                    y = y + 0.01 * rng.standard_normal((n, d))
            elif mod == 4:
                # rattler relocation
                y = y + 0.005 * rng.standard_normal((n, d))
                if len(rattlers) > 0:
                    k = min(3, len(rattlers))
                    idx = rng.choice(rattlers, size=k, replace=False)
                    rnd = rng.standard_normal((k, d))
                    rnd = rnd / np.linalg.norm(rnd, axis=1, keepdims=True)
                    # blend random to keep some continuity half the time
                    if trial % 2 == 0:
                        y[idx] = rnd
                    else:
                        y[idx] = y[idx] + 0.15 * rnd
                else:
                    idx = rng.choice(n, size=3, replace=False)
                    rnd = rng.standard_normal((3, d))
                    rnd = rnd / np.linalg.norm(rnd, axis=1, keepdims=True)
                    y[idx] = rnd
            else:
                sig = 0.05
                y = y + sig * rng.standard_normal((n, d))
        except Exception:
            trial += 1
            continue
        # normalize
        try:
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
        except Exception:
            trial += 1
            continue
        # smooth phase (cheap, topology-changing descent)
        try:
            # vary smooth params by trial
            if mod in (3, 4):
                y = smooth_descent(y, steps=150, beta0=15.0, beta1=100.0, lr0=0.0025)
            else:
                y = smooth_descent(y, steps=100, beta0=20.0, beta1=80.0, lr0=0.0018)
        except Exception:
            pass
        # short LP polish
        budget = min(1.6, HOP_END - time.time() - 0.05)
        if budget < 0.3:
            trial += 1
            continue
        try:
            y = lp_polish(y, seconds=budget)
            v = max_cos(y)
        except Exception:
            trial += 1
            continue
        if v < best_val - 1e-12 and np.all(np.isfinite(y)):
            xb = y
            best_val = v
            try:
                _, cI, cJ, rattlers = contacts_of(xb)
            except Exception:
                pass
            # extra exploitation when promising: a bit longer polish immediately
            if time.time() < HOP_END - 3.0:
                try:
                    y2 = lp_polish(xb, seconds=min(3.0, HOP_END - time.time()))
                    v2 = max_cos(y2)
                    if v2 < best_val:
                        xb = y2
                        best_val = v2
                        _, cI, cJ, rattlers = contacts_of(xb)
                except Exception:
                    pass
        trial += 1

    # final long polish of best
    try:
        remain = HARD_END - time.time() - 0.5
        if remain > 1.0:
            xb = lp_polish(xb, seconds=remain)
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb

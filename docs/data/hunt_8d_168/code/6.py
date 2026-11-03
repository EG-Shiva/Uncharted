import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _quench(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    # local refs for speed
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        # bias correction
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
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
        if p == 0:
            break
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_BUDGET = 175.0
    PHASE1 = 128.0  # leave rest for LP

    # load start
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    n, d = best.shape
    rng = np.random.RandomState(0)

    deadline1 = t0 + PHASE1
    it = 0
    while time.time() < deadline1:
        it += 1
        # time check for remaining budget
        if time.time() + 2.0 > t0 + TIME_BUDGET:
            break
        X = best.copy()
        choice = rng.rand()
        if choice < 0.50:
            # global Gaussian kick
            sigmas = (0.008, 0.015, 0.030, 0.060)
            sigma = sigmas[rng.randint(len(sigmas))]
            noise = rng.randn(n, d) * sigma
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 30.0 + rng.rand() * 30.0
            beta1 = 200.0 + rng.rand() * 400.0
            steps = 1200 + int(rng.randint(800))
            lr0 = 0.003 + rng.rand() * 0.005
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        elif choice < 0.82:
            # block replacement by voids
            k = 1 + int(rng.randint(4))  # 1..4
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            indmax = G.max(axis=1)
            if rng.rand() < 0.5:
                idx = np.argsort(indmax)[-k:]
            else:
                idx = rng.choice(n, k, replace=False)
            for j in idx:
                others = np.delete(X, j, axis=0)
                cand = rng.randn(4000, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ others.T
                mm = dots.max(axis=1)
                y = cand[np.argmin(mm)].copy()
                # quick hill climb for the single point
                lr = 0.02
                for _ in range(200):
                    dts = others @ y
                    sel = np.argpartition(dts, -3)[-3:]
                    g = others[sel].mean(axis=0)
                    gt = g - (g @ y) * y
                    nn = np.linalg.norm(gt)
                    if nn < 1e-12:
                        break
                    gt /= nn
                    y = y - lr * gt
                    y /= np.linalg.norm(y)
                    lr *= 0.995
                X[j] = y
            beta0 = 40.0
            beta1 = 300.0 + rng.rand() * 300.0
            steps = 1400 + int(rng.randint(900))
            lr0 = 0.004 + rng.rand() * 0.004
            lr1 = lr0 * 0.12
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        else:
            # local cluster shake
            c = rng.randn(d)
            c /= np.linalg.norm(c)
            dots = X @ c
            k = 8 + int(rng.randint(8))
            idx = np.argsort(dots)[-k:]
            sigma = 0.05 + rng.rand() * 0.05
            noise = np.zeros((n, d))
            noise[idx] = rng.randn(k, d) * sigma
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 45.0
            beta1 = 400.0
            steps = 1600
            lr0 = 0.006
            lr1 = 0.0006
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)

        vq = _max_cos(Xq)
        if vq + 1e-12 < best_val and np.all(np.isfinite(Xq)):
            best = Xq
            best_val = vq

    remaining = (t0 + TIME_BUDGET) - time.time()
    # final LP polish with leftover time
    if remaining > 6.0:
        secs = min(remaining - 2.0, 42.0)
        try:
            polished = _lp_polish(best, secs)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: replace any bad row
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(bad.sum(), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    norms = np.linalg.norm(best, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best = best / norms
    return best

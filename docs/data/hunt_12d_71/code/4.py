import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_opt(X0, steps=3000, deadline=1e18):
    """Minimize softmax max-cosine with Riemannian Adam, annealed temperature."""
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    for t in range(1, steps + 1):
        if t < 1000:
            s, lr = 50.0, 0.012
        elif t < 2000:
            s, lr = 150.0, 0.008
        else:
            s, lr = 400.0, 0.005
        # stable softmax over off-diagonal
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        cmax = float(G.max())
        E = np.exp(s * (G - cmax))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        grad = W @ X
        # tangent projection
        r = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - r * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        b1p *= b1
        b2p *= b2
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        # keep update tangential
        dr = np.sum(delta * X, axis=1, keepdims=True)
        delta = delta - dr * X
        X = X - delta
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if (t % 400) == 0 and time.time() > deadline:
            break
    return X


def solve():
    np.random.seed(0)
    t0 = time.time()
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            rng0 = np.random.default_rng(0)
            Xb = rng0.standard_normal((71, 12))
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    n, d = Xb.shape
    # enforce shape (71,12)
    if Xb.shape != (71, 12):
        rng0 = np.random.default_rng(0)
        Xb = rng0.standard_normal((71, 12))
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = max_cos(Xb)
    X_best = Xb.copy()

    rng = np.random.default_rng(0)
    sigmas = [0.008, 0.015, 0.025, 0.04, 0.06, 0.10, 0.15, 0.02, 0.03, 0.05]

    # phase 1: basin hopping until ~125s
    PHASE1 = 122.0
    trial = 0
    while time.time() - t0 < PHASE1:
        sigma = sigmas[trial % len(sigmas)]
        if trial % 6 == 5:
            # targeted single-point reinsertion: worst-contact point
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            thr = best_val - 1e-7
            cnt = np.sum(G > thr, axis=1)
            worst = int(np.argmax(cnt))
            if (trial // 6) % 2 == 1:
                idx = int(rng.integers(n))
            else:
                idx = worst
            Xp = X_best.copy()
            rv = rng.standard_normal(d)
            nr = np.linalg.norm(rv)
            if nr == 0:
                rv[0] = 1.0
                nr = 1.0
            rv /= nr
            # sometimes place opposite-ish to diversify
            if (trial // 6) % 4 == 3:
                # far from others: try a few random and pick best
                best_rv = rv
                best_sc = 10.0
                for _ in range(8):
                    cand = rng.standard_normal(d)
                    cand /= np.linalg.norm(cand)
                    sc = float(np.max(X_best @ cand))
                    if sc < best_sc:
                        best_sc = sc
                        best_rv = cand
                rv = best_rv
            Xp[idx] = rv
            # tiny jitter others to break symmetry
            Xp = Xp + 0.002 * rng.standard_normal(size=Xp.shape)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            noise = rng.standard_normal(size=(n, d))
            Xp = X_best + sigma * noise
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)

        deadline = t0 + PHASE1 + 2.0
        Xo = smooth_opt(Xp, steps=3000, deadline=deadline)
        val = max_cos(Xo)
        if val < best_val:
            # quick LP refine to squeeze local optimum
            tleft = (t0 + PHASE1) - time.time()
            if tleft > 4.0:
                Xo2 = lp_polish(Xo, min(3.0, tleft - 1.0))
                val2 = max_cos(Xo2)
                if val2 < val:
                    Xo, val = Xo2, val2
            if val < best_val:
                X_best, best_val = Xo, val
        trial += 1
        # safety: stop if approaching hard limit
        if time.time() - t0 > 170:
            break

    # final polish with remaining time
    remain = 178.0 - (time.time() - t0)
    if remain > 6.0:
        X_best = lp_polish(X_best, min(remain - 2.0, 38.0))

    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_refine(X, iu0, iu1, schedule):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    t = 0
    # local refs for speed
    for beta, lr, steps in schedule:
        for _ in range(steps):
            t += 1
            G = X @ X.T
            g = G[iu0, iu1]
            mx = np.max(g)
            e = np.exp(beta * (g - mx))
            s = np.sum(e)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
            m *= b1
            m += (1.0 - b1) * gt
            v *= b2
            v += (1.0 - b2) * (gt * gt)
            bc1 = 1.0 - (b1 ** t)
            bc2 = 1.0 - (b2 ** t)
            denom = np.sqrt(v / bc2) + eps
            step = (lr / bc1) * m / denom
            X = X - step
            nrm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
            nrm[nrm == 0] = 1.0
            X /= nrm
    return X


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mxx = c.max()
        sel = c > mxx - max(20 * radius, 1e-9)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 176, 9
    # load start
    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                X0 = a.astype(float)
                break
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal((n, d))
    X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    best_X = X0.copy()
    best_val = max_cos(best_X)
    iu0, iu1 = np.triu_indices(n, 1)

    sched_short = [(30.0, 0.02, 150), (80.0, 0.01, 150), (200.0, 0.005, 250), (500.0, 0.002, 250)]
    sched_long = [(30.0, 0.02, 200), (80.0, 0.01, 250), (200.0, 0.005, 350), (600.0, 0.0015, 400)]

    # quick intensify of starting point with short smooth (in case LP stuck)
    # only if we have time; do one short refine and keep if better
    try:
        _t = best_X.copy()
        _t = smooth_refine(_t, iu0, iu1, [(80.0, 0.005, 200), (300.0, 0.002, 200)])
        _v = max_cos(_t)
        if _v < best_val:
            best_X, best_val = _t, _v
    except Exception:
        pass

    explore_budget = 118.0  # seconds for hopping, leave rest for LP
    trial = 0
    # precompute for worst search
    while time.time() - t0 < explore_budget:
        trial += 1
        mode = trial % 5
        Xt = None
        if mode == 0:
            sigma = 0.008 + 0.008 * rng.random()
            Xt = best_X + sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            sched = sched_short
        elif mode == 1:
            sigma = 0.025 + 0.02 * rng.random()
            Xt = best_X + sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            sched = sched_short
        elif mode == 2:
            # reinsert worst k fully random
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            indiv = G.max(axis=1)
            k = int(rng.choice([2, 3, 4, 5]))
            worst = np.argsort(indiv)[-k:]
            Xt = best_X.copy()
            Xt[worst] = rng.standard_normal((k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            sched = sched_short
        elif mode == 3:
            # heavy kick to worst + neighbours
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            indiv = G.max(axis=1)
            k = int(rng.choice([4, 6, 8]))
            worst = np.argsort(indiv)[-k:]
            Xt = best_X.copy()
            Xt[worst] += 0.35 * rng.standard_normal((k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            sched = sched_short
        else:
            # random subset moderate shake
            k = int(rng.choice([6, 10, 16]))
            idx = rng.choice(n, k, replace=False)
            Xt = best_X.copy()
            Xt[idx] += 0.12 * rng.standard_normal((k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            sched = sched_short

        # occasional long refine for promising diversity
        if trial % 25 == 0:
            sched = sched_long
        try:
            Xt = smooth_refine(Xt, iu0, iu1, sched)
        except Exception:
            continue
        v = max_cos(Xt)
        if v < best_val:
            best_val = v
            best_X = Xt
            # extra intensification immediately around new best
            try:
                if time.time() - t0 < explore_budget - 5:
                    Xt2 = smooth_refine(best_X.copy(), iu0, iu1, [(600.0, 0.0015, 300)])
                    v2 = max_cos(Xt2)
                    if v2 < best_val:
                        best_val = v2
                        best_X = Xt2
            except Exception:
                pass
        # stop early if time nearly up
        if time.time() - t0 > explore_budget:
            break

    # final intensify + LP polish with remaining time
    remain = 180.0 - (time.time() - t0) - 2.0
    if remain < 5:
        remain = 5
    # one more high-beta smooth before LP (fast, ~1-2s)
    try:
        if remain > 12:
            best_X = smooth_refine(best_X.copy(), iu0, iu1, [(800.0, 0.001, 250)])
            best_val = max_cos(best_X)
            remain = 180.0 - (time.time() - t0) - 2.0
    except Exception:
        pass
    if remain > 3:
        try:
            best_X = lp_polish(best_X, seconds=max(3.0, min(remain, 45.0)))
        except Exception:
            pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X

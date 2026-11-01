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
        obj = np.zeros(nv); obj[-1] = 1.0
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


def gd_optimize(X_init, steps, a0, a1, s0, s1):
    X = X_init.copy()
    n, d = X.shape
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        step = s0 * (s1 / s0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        # stable softmax weights
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        # tangent projection
        d = np.sum(grad * X, axis=1, keepdims=True)
        grad -= d * X
        nm = np.linalg.norm(grad, axis=1).max()
        if nm < 1e-14:
            continue
        grad *= (step / nm)
        X -= grad
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    TIME_BUDGET = 168.0
    SEARCH_TIME = 115.0  # rest for LP polish

    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal(size=(173, 5))
    xb = np.asarray(xb, dtype=float)
    # ensure shape (173,5)
    if xb.shape != (173, 5):
        rng0 = np.random.default_rng(0)
        tmp = rng0.standard_normal(size=(173, 5))
        # try to fit
        n = min(xb.shape[0], 173)
        tmp[:n, :min(xb.shape[1], 5)] = xb[:n, :min(xb.shape[1], 5)]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    cur_best = xb.copy()
    cur_val = max_cos(cur_best)

    rng = np.random.default_rng(0)

    # pool of promising raw codes: list of (val, X)
    pool = []
    POOL_SIZE = 12

    def pool_add(val, X):
        # keep diverse pool sorted
        pool.append((val, X.copy()))
        pool.sort(key=lambda t: t[0])
        del pool[POOL_SIZE:]

    trial = 0
    # seed pool with starting point
    pool_add(cur_val + 1e-9, cur_best)

    while time.time() - t0 < SEARCH_TIME:
        trial += 1
        # choose base
        if len(pool) > 1 and rng.random() < 0.40:
            bi = rng.integers(len(pool))
            baseX = pool[bi][1]
        else:
            baseX = cur_best

        mode = rng.random()
        Xp = baseX.copy()
        if mode < 0.38:
            # global shake, log-uniform sigma
            sigma = 10.0 ** rng.uniform(-2.6, -0.7)
            Xp = Xp + sigma * rng.standard_normal(size=Xp.shape)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode < 0.70:
            # local cluster shake around random center
            i0 = int(rng.integers(n))
            G0 = baseX @ baseX[i0]
            k = int(rng.integers(8, 32))
            idx = np.argpartition(-G0, k)[:k]
            sigma_big = 10.0 ** rng.uniform(-1.35, -0.45)
            sigma_small = 0.003
            noise = rng.standard_normal(size=Xp.shape) * sigma_small
            noise[idx] = rng.standard_normal(size=(len(idx), d)) * sigma_big
            # occasionally also shake second cluster
            if rng.random() < 0.3:
                i1 = int(rng.integers(n))
                G1 = baseX @ baseX[i1]
                idx2 = np.argpartition(-G1, k)[:k]
                noise[idx2] += rng.standard_normal(size=(len(idx2), d)) * sigma_big * 0.5
            Xp = Xp + noise
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            # surgery: replace 1-3 worst points by void maxima
            mrep = int(rng.integers(1, 4))
            G = baseX @ baseX.T
            np.fill_diagonal(G, -2.0)
            worst = np.argsort(-G.max(axis=1))[:mrep]
            Xtmp = baseX.copy()
            for w in worst:
                cand = rng.standard_normal(size=(2500, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xtmp.T
                dots[:, w] = -10.0
                md = dots.max(axis=1)
                Xtmp[w] = cand[np.argmin(md)]
            # small shake of rest
            Xtmp = Xtmp + 0.004 * rng.standard_normal(size=Xtmp.shape)
            Xtmp /= np.linalg.norm(Xtmp, axis=1, keepdims=True)
            Xp = Xtmp

        steps = int(rng.integers(220, 520))
        a0 = 10.0 ** rng.uniform(0.7, 1.6)
        a1 = 10.0 ** rng.uniform(2.0, 2.65)
        s0 = 10.0 ** rng.uniform(-1.7, -1.0)
        s1 = s0 * (10.0 ** rng.uniform(-1.0, -0.3))
        # guard s1
        s1 = max(s1, 3e-4)

        Xo = gd_optimize(Xp, steps, a0, a1, s0, s1)
        raw = max_cos(Xo)
        # update pool if good enough (keep best raws)
        if len(pool) < POOL_SIZE or raw < pool[-1][0]:
            pool_add(raw, Xo)
        # direct update of cur_best on raw (rare, but saves time)
        # note cur_best is polished value; raw beating it is significant
        if raw < cur_val:
            # treat as new best without polish yet
            cur_best = Xo.copy()
            cur_val = raw

        # time check for loop
        if time.time() - t0 >= SEARCH_TIME:
            break

    # Final stage: LP-polish most promising candidates
    # sort pool by raw value
    pool.sort(key=lambda t: t[0])
    # candidates to polish: top few unique + cur_best
    cands = []
    seen = []
    for v, X in pool[:6]:
        cands.append(X)
    # ensure cur_best included
    cands.append(cur_best)
    # also try original file again to not lose record
    try:
        orig = np.load("best.npy").astype(float)
        orig /= np.linalg.norm(orig, axis=1, keepdims=True)
        cands.append(orig)
    except Exception:
        pass

    # deduplicate roughly by value? just keep first 5
    # polish budget
    overall_best = cur_best
    overall_val = cur_val
    # check orig value in case GD never beat it
    for X in cands:
        v = max_cos(X)
        if v < overall_val:
            overall_val = v
            overall_best = X.copy()

    remaining = TIME_BUDGET - (time.time() - t0) - 2.0
    if remaining < 5:
        remaining = 5.0
    # allocate: give more to current overall best
    # polish top 3 then final long polish of winner
    top = sorted(cands, key=lambda X: max_cos(X))[:3]
    per = max(4.0, remaining * 0.22)
    polished = []
    for X in top:
        if time.time() - t0 > TIME_BUDGET - 12:
            break
        Xp2 = lp_polish(X, per)
        polished.append(Xp2)
        v = max_cos(Xp2)
        if v < overall_val:
            overall_val = v
            overall_best = Xp2.copy()

    # final long polish of winner with leftover time
    leftover = TIME_BUDGET - (time.time() - t0) - 1.0
    if leftover > 3:
        overall_best = lp_polish(overall_best, leftover)
        overall_val = max_cos(overall_best)

    overall_best /= np.linalg.norm(overall_best, axis=1, keepdims=True)
    return overall_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 2500:
            # keep tightest 2500
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            p = 2500
            csv = c[sel][idx]
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        tl = max(0.3, seconds - (time.time() - start))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X


def _whiten(X):
    n, d = X.shape
    C = (X.T @ X) / n
    C = 0.5 * (C + C.T) + 1e-9 * np.eye(d)
    w, U = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    Cinvh = (U * (1.0 / np.sqrt(w))) @ U.T
    A = Cinvh / np.sqrt(float(d))
    Y = X @ A
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def _anneal_adam(X0, betas, steps_each, lr0, lam0, rng):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    iu0, iu1 = np.triu_indices(n, 1)
    I15 = np.eye(d) / float(d)
    m1 = np.zeros((n, d))
    m2 = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for bi, beta in enumerate(betas):
        lam = lam0 * (0.35 ** bi) if lam0 > 0 else 0.0
        lr = lr0 * (0.7 ** bi)
        for s in range(steps_each):
            t += 1
            G = X @ X.T
            gv = G[iu0, iu1]
            mm = gv.max()
            e = np.exp(beta * (gv - mm))
            S = e.sum()
            w = e / S
            # W @ X without full matrix
            # build W sparse action: (W@X)_i = sum_j w_ij X_j
            Wx = np.zeros((n, d))
            # accumulate using vectorization via dense W (78x78 is tiny)
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Grad = Gr - dot * X
            if lam > 0:
                C = (X.T @ X) / n
                E = C - I15
                Gi = (4.0 / n) * (X @ E)
                Gi = Gi - ((Gi * X).sum(axis=1, keepdims=True)) * X
                Grad = Grad + lam * Gi
            m1 = b1 * m1 + (1 - b1) * Grad
            m2 = b2 * m2 + (1 - b2) * (Grad * Grad)
            m1h = m1 / (1 - b1 ** t)
            m2h = m2 / (1 - b2 ** t)
            delta = m1h / (np.sqrt(m2h) + eps)
            delta = delta - ((delta * X).sum(axis=1, keepdims=True)) * X
            # clip
            dn = np.linalg.norm(delta, axis=1, keepdims=True)
            delta = delta / np.maximum(1.0, dn / 1.0)
            X = X - lr * delta
            X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(7)
    n, d = 78, 15

    def left():
        return LIMIT - (time.time() - t0)

    # load incumbent
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incumbent = bestX.copy()

    # quick snap of incumbent to make sure we hold lab best
    if left() > 60:
        incumbent = _lp_polish(incumbent, min(4.0, left() - 50))
        v = _max_cos(incumbent)
        if v < bestV:
            bestV, bestX = v, incumbent.copy()

    # build diverse starts: whitened incumbent, shaken, pure random
    starts = []
    try:
        starts.append(_whiten(incumbent))
    except Exception:
        pass
    for sig in (0.35, 0.6):
        try:
            y = incumbent + sig * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            starts.append(_whiten(y))
        except Exception:
            pass
    for _ in range(5):
        y = rng.standard_normal((n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        starts.append(y)

    betas = [40.0, 90.0, 170.0, 280.0, 420.0]
    cands = []
    for si, s0 in enumerate(starts):
        if left() < 30:
            break
        # fewer steps for speed, more for first (whitened) starts
        steps = 700 if si < 3 else 500
        try:
            y = _anneal_adam(s0, betas, steps, 0.02, 1.5, rng)
        except Exception:
            continue
        v = _max_cos(y)
        cands.append((v, y))
        # keep global best even before polish (unlikely)
        if v < bestV:
            bestV, bestX = v, y.copy()

    # polish top candidates briefly
    if cands:
        cands.sort(key=lambda t: t[0])
        for k in range(min(3, len(cands))):
            if left() < 22:
                break
            _, y0 = cands[k]
            # extra shake+whiten for diversity on worse ones
            y = _lp_polish(y0, min(6.0, max(1.5, left() - 18)))
            v = _max_cos(y)
            if v < bestV:
                bestV, bestX = v, y.copy()
            # try one more anneal at high beta from polished point
            if left() > 25 and v < 0.19:
                try:
                    y2 = _anneal_adam(y, [350.0, 550.0], 400, 0.008, 0.0, rng)
                    y2 = _lp_polish(y2, min(4.0, max(1.0, left() - 18)))
                    v2 = _max_cos(y2)
                    if v2 < bestV:
                        bestV, bestX = v2, y2.copy()
                except Exception:
                    pass

    # hopping around current best with void reinsertion + whiten
    it = 0
    while left() > 20:
        it += 1
        if it > 6:
            break
        try:
            # pick k points with worst personal maxima to reinsert
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = G.max(axis=1)
            k = int(rng.integers(3, 7))
            idx = np.argpartition(-pmax, k)[:k]
            y = bestX.copy()
            # largest-hole directions for reinsertion
            trials = rng.standard_normal((800, d))
            trials /= np.linalg.norm(trials, axis=1, keepdims=True)
            sims = trials @ y.T
            score = sims.max(axis=1)
            order = np.argsort(score)
            for ii, pi in enumerate(idx):
                y[pi] = trials[order[ii % len(order)]]
            # small global shake + whiten every other iter
            if it % 2 == 0:
                y = y + 0.08 * rng.standard_normal((n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _whiten(y)
            y = _anneal_adam(y, [120.0, 260.0, 450.0], 500, 0.012, 0.3, rng)
            y = _lp_polish(y, min(5.0, max(1.0, left() - 14)))
            v = _max_cos(y)
            if v < bestV:
                bestV, bestX = v, y.copy()
        except Exception:
            continue

    # final snap of global best with remaining time
    if left() > 3:
        try:
            y = _lp_polish(bestX, max(1.0, left() - 2.0))
            v = _max_cos(y)
            if v < bestV:
                bestV, bestX = v, y.copy()
        except Exception:
            pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

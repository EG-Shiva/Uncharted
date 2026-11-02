import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def anneal_refine(X, deadline, stages, mu=0.9):
    n, d = X.shape
    v = np.zeros_like(X)
    for beta, iters, lr0, lr1 in stages:
        v.fill(0.0)
        for it in range(iters):
            if (it % 20 == 0) and time.time() > deadline:
                return X
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            m = float(np.max(G))
            # exp(beta*(G-m)) <=1, safe
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            tg = grad - dot * X
            nrm = float(np.max(np.sqrt(np.sum(tg * tg, axis=1))))
            if not np.isfinite(nrm) or nrm < 1e-14:
                continue
            tg /= nrm
            v = mu * v + (1.0 - mu) * tg
            frac = it / max(1, iters - 1)
            lr = lr0 + (lr1 - lr0) * frac
            X = X - lr * v
            X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def micro_break(X, cur, rng, deadline):
    n, d = X.shape
    sigma = 0.002
    fail = 0
    # scratch for speed
    while time.time() < deadline:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        # propose jitter of both i,j tangentially
        Y = X.copy()
        for k in (i, j):
            nz = rng.normal(scale=sigma, size=d)
            nz -= np.dot(nz, Y[k]) * Y[k]
            Y[k] += nz
            nn = np.linalg.norm(Y[k])
            if nn == 0 or not np.isfinite(nn):
                Y[k] = X[k]
            else:
                Y[k] /= nn
        # occasionally also nudge a random third point to relieve 2nd tier
        if (fail % 7) == 6:
            r = int(rng.integers(0, n))
            if r != i and r != j:
                nz = rng.normal(scale=sigma * 0.5, size=d)
                nz -= np.dot(nz, Y[r]) * Y[r]
                Y[r] += nz
                nn = np.linalg.norm(Y[r])
                if nn != 0 and np.isfinite(nn):
                    Y[r] /= nn
        G2 = Y @ Y.T
        np.fill_diagonal(G2, -2.0)
        v = float(np.max(G2))
        if v < cur:
            X = Y
            cur = v
            fail = 0
            sigma = min(sigma * 1.05, 0.003)
        else:
            fail += 1
            if fail >= 250:
                sigma *= 0.6
                fail = 0
                if sigma < 2e-6:
                    sigma = 2e-6
                    # kick a random point to escape flat region (keep best tracking outside)
                    # do a small random walk step that may worsen, but only if stuck long
                    # we do it on X and update cur accordingly
                    r = int(rng.integers(0, n))
                    nz = rng.normal(scale=0.004, size=d)
                    nz -= np.dot(nz, X[r]) * X[r]
                    X[r] += nz
                    X[r] /= np.linalg.norm(X[r])
                    G3 = X @ X.T
                    np.fill_diagonal(G3, -2.0)
                    cur = float(np.max(G3))
                    sigma = 0.001
    return X, cur


def lp_polish(X, cur, deadline, r0=8e-4):
    n, d = X.shape
    radius = float(r0)
    iu_r, iu_c = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-13:
        if deadline - time.time() < 2.2:
            break
        Gfull = X @ X.T
        c = Gfull[iu_r, iu_c]
        m = float(np.max(c))
        tol = 20.0 * radius
        if tol < 1e-9:
            tol = 1e-9
        # select near-max
        if np.count_nonzero(c > m - tol) > 6000:
            idx = np.argpartition(c, -6000)[-6000:]
            I = iu_r[idx]
            J = iu_c[idx]
            cs = c[idx]
        else:
            sel = c > m - tol
            I = iu_r[sel]
            J = iu_c[sel]
            cs = c[sel]
        p = int(I.size)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        G2 = y @ y.T
        np.fill_diagonal(G2, -2.0)
        vv = float(np.max(G2))
        if vv < cur:
            X = y
            cur = vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, cur


def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPLORE_END = t0 + 124.0
    rng = np.random.default_rng(0)
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=np.float64))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=np.float64))
    except Exception:
        pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                cc = np.asarray(cc, dtype=np.float64).reshape(453, 5)
            nrm = np.linalg.norm(cc, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            cc = cc / nrm
            if not np.all(np.isfinite(cc)):
                continue
            vv = max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.normal(size=(453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    n, d = bestX.shape

    # immediate shave of isolated max pair
    if time.time() < HARD - 5:
        bestX, bestV = micro_break(bestX, bestV, rng, min(t0 + 9.0, HARD - 5))

    # basin-hopping: perturb worst points + anneal + micro
    sig_cycle = [0.02, 0.01, 0.005, 0.0025]
    K_cycle = [14, 8, 4, 2]
    it = 0
    X_cur = bestX.copy()
    curV = bestV
    while time.time() < EXPLORE_END:
        it += 1
        if time.time() > HARD - 8:
            break
        # find worst points of best
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        row_max = G.max(axis=1)
        order = np.argsort(-row_max)
        ci = (it - 1) % 4
        K = int(K_cycle[ci])
        sigma = float(sig_cycle[ci])
        # sometimes hop from current instead of best
        base = bestX
        if (it % 7 == 0) and curV < bestV + 0.002:
            base = X_cur
            Gc = X_cur @ X_cur.T
            np.fill_diagonal(Gc, -2.0)
            order = np.argsort(-Gc.max(axis=1))
        X_try = base.copy()
        idx = order[:K]
        if it % 3 == 0:
            extra = rng.integers(0, n, size=2)
            idx = np.unique(np.concatenate([idx, extra]))
        X_try[idx] += rng.normal(scale=sigma, size=(len(idx), d))
        X_try += rng.normal(scale=0.001, size=X_try.shape)
        X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        # anneal
        dl = min(HARD - 6, time.time() + 3.5, EXPLORE_END)
        stages = [(60.0, 180, 0.004, 0.001), (180.0, 180, 0.002, 0.0005), (500.0, 180, 0.001, 0.0003)]
        if ci == 3:
            stages = [(100.0, 150, 0.003, 0.0008), (400.0, 150, 0.0012, 0.0003)]
        X_try = anneal_refine(X_try, dl, stages)
        v = max_cos(X_try)
        # quick micro shave
        dl2 = min(HARD - 6, time.time() + 1.8, EXPLORE_END + 1.0)
        if dl2 - time.time() > 0.4:
            X_try, v = micro_break(X_try, v, rng, dl2)
        if v < bestV:
            bestV = v
            bestX = X_try.copy()
            X_cur = X_try.copy()
            curV = v
        else:
            if v < curV:
                X_cur = X_try.copy()
                curV = v
            elif v < curV + 0.001 and rng.random() < 0.12:
                X_cur = X_try.copy()
                curV = v

    # fine anneal
    if time.time() < HARD - 12:
        bestX = anneal_refine(bestX, HARD - 11,
                              [(80.0, 200, 0.002, 0.0006), (250.0, 200, 0.001, 0.0003),
                               (600.0, 200, 0.0006, 0.00015)])
        bestV = max_cos(bestX)
    # LP polish
    if time.time() < HARD - 4:
        bestX, bestV = lp_polish(bestX, bestV, HARD - 2.5, r0=8e-4)
    # final micro shave + tiny LP if time remains
    if time.time() < HARD - 2:
        bestX, bestV = micro_break(bestX, bestV, rng, HARD - 1.2)
    if time.time() < HARD - 0.8:
        # one more very small LP attempt
        try:
            bestX, bestV = lp_polish(bestX, bestV, HARD - 0.3, r0=2e-4)
        except Exception:
            pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.nan_to_num(bestX, nan=1.0 / np.sqrt(5.0))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

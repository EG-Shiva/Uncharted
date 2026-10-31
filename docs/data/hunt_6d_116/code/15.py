import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    return m, cnt, G


def _smooth(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    for it in range(1, iters + 1):
        frac = it / iters
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        s = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - s * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * Gt * Gt
        mhat = m / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        try:
            Gfull = X @ X.T
            c = Gfull[iu]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            I = iu[0][sel]
            J = iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def _hole_best(others, rng, n_cand):
    d = others.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    mx = dots.max(axis=1)
    return cand[int(np.argmin(mx))]


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        Xbest = rng.standard_normal((116, 6))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    try:
        Xr = np.load("record.npy").astype(float)
        Xr = Xr / np.linalg.norm(Xr, axis=1, keepdims=True)
        if _max_cos(Xr) < _max_cos(Xbest):
            Xbest = Xr
    except Exception:
        pass
    n, d = Xbest.shape
    vbest = _max_cos(Xbest)
    # wandering state (allows sideways moves)
    Xcur = Xbest.copy()
    vcur = vbest

    SEARCH = 122.0
    it = 0
    while time.time() - t0 < SEARCH:
        it += 1
        elapsed = time.time() - t0
        prog = elapsed / SEARCH
        # temperature for Metropolis on wandering state
        Temp = 0.0025 * (1.0 - prog) + 0.00015 * prog
        try:
            _, cnt, G = _contacts(Xcur, tol=1e-7)
        except Exception:
            cnt = np.zeros(n, dtype=int)
        Y = Xcur.copy()
        mode = int(rng.integers(0, 6))
        # occasional full random restart to find new basin
        if it % 47 == 0:
            mode = 6
        if mode == 0:
            sigma = 0.012 + 0.020 * float(rng.random())
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, iters=500, a0=50.0, a1=500.0, lr0=0.012, lr1=0.002)
        elif mode == 1:
            k = int(rng.integers(4, 11))
            order = np.argsort(-cnt)
            idx = order[:k]
            sig = 0.10 + 0.15 * float(rng.random())
            Y[idx] = Y[idx] + sig * rng.standard_normal((k, d))
            Y = Y + 0.006 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, iters=700, a0=60.0, a1=650.0, lr0=0.014, lr1=0.002)
        elif mode == 2:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:5]
            # limit number reinserted
            if len(ratt) > 6:
                ratt = ratt[rng.choice(len(ratt), 6, replace=False)]
            for r in ratt:
                others = np.delete(Y, r, axis=0)
                Y[r] = _hole_best(others, rng, n_cand=3500)
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, iters=800, a0=70.0, a1=700.0, lr0=0.013, lr1=0.002)
        elif mode == 3:
            # worst-pair surgery: reinsert one end of worst pairs into hole
            Gu = Y @ Y.T
            np.fill_diagonal(Gu, -2.0)
            # find worst 3 pairs
            flat = Gu[np.triu_indices(n, 1)]
            order = np.argsort(-flat)[:3]
            iu0, iu1 = np.triu_indices(n, 1)
            for o in order:
                a = int(iu0[o])
                b = int(iu1[o])
                victim = b if float(rng.random()) < 0.5 else a
                others = np.delete(Y, victim, axis=0)
                Y[victim] = _hole_best(others, rng, n_cand=3000)
                other = a if victim == b else b
                Y[other] = Y[other] + 0.03 * rng.standard_normal(d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, iters=700, a0=60.0, a1=600.0, lr0=0.013, lr1=0.002)
        elif mode == 4:
            sigma = 0.05 + 0.04 * float(rng.random())
            Y = Y + sigma * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, iters=600, a0=45.0, a1=450.0, lr0=0.012, lr1=0.0025)
        elif mode == 5:
            # hub hole-reinsert: move overloaded points to voids
            order = np.argsort(-cnt)
            k = int(rng.integers(2, 5))
            idx = order[:k]
            for r in idx:
                others = np.delete(Y, int(r), axis=0)
                Y[int(r)] = _hole_best(others, rng, n_cand=3000)
            Y = Y + 0.005 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, iters=800, a0=70.0, a1=700.0, lr0=0.012, lr1=0.002)
        else:
            # fresh random start, coarsely polished
            Y = rng.standard_normal((n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _smooth(Y, iters=900, a0=40.0, a1=500.0, lr0=0.015, lr1=0.002)
        try:
            v = _max_cos(Y)
        except Exception:
            continue
        # Metropolis update of wandering state
        if v < vcur or float(rng.random()) < np.exp((vcur - v) / max(Temp, 1e-9)):
            Xcur = Y
            vcur = v
        # LP-refine promising candidates toward true local optimum
        if v < vbest + 0.004:
            budget = 0.8
            if v < vbest - 1e-6:
                budget = 1.2
            if time.time() - t0 > SEARCH - 2.0:
                break
            try:
                Y2 = _lp_polish(Y, seconds=budget)
                v2 = _max_cos(Y2)
            except Exception:
                continue
            if v2 < vbest - 1e-12:
                Xbest = Y2
                vbest = v2
                # also pull wandering state to new best sometimes
                if float(rng.random()) < 0.7:
                    Xcur = Y2.copy()
                    vcur = v2
            elif v2 < vcur and float(rng.random()) < 0.3:
                Xcur = Y2.copy()
                vcur = v2
        # keep time guard
        if time.time() - t0 > SEARCH - 0.5:
            break
    # final intensification: high-temperature smooth then long LP
    try:
        if time.time() - t0 < 168:
            Xbest = _smooth(Xbest, iters=700, a0=500.0, a1=1500.0, lr0=0.004, lr1=0.0004)
        remain = 171.0 - (time.time() - t0)
        remain = float(max(4.0, min(38.0, remain)))
        if remain > 1.0:
            Xbest = _lp_polish(Xbest, seconds=remain)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

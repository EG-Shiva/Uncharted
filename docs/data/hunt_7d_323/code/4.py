import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu0]
        m = float(c_full.max())
        tol = max(20.0 * radius, 1e-9)
        sel = c_full > m - tol
        # cap LP size for speed
        if int(sel.sum()) > 4000:
            thr = np.partition(c_full, -4000)[-4000]
            sel = c_full >= thr
        c = c_full[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def _hinge_smooth(X, target, iters, lr0):
    n, d = X.shape
    for it in range(iters):
        lr = lr0 * (1.0 - 0.7 * it / max(1, iters))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        W = G - target
        np.maximum(W, 0.0, out=W)
        # gradient step of 0.5*sum W^2
        F = W @ X
        s = np.sum(F * X, axis=1, keepdims=True)
        F -= s * X
        X -= lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _best_holes(X, k, rng, n_cand=20000):
    n, d = X.shape
    C = rng.randn(n_cand, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ X.T
    mx = S.max(axis=1)
    idx = np.argpartition(mx, k)[:k]
    H = C[idx].copy()
    # refine holes away from X with soft-max repulsion
    T = 0.01
    for _ in range(100):
        S = H @ X.T
        m = S.max(axis=1, keepdims=True)
        E = np.exp((S - m) / T)
        E /= E.sum(axis=1, keepdims=True)
        F = E @ X
        F -= np.sum(F * H, axis=1, keepdims=True) * H
        H -= 0.15 * F
        H /= np.linalg.norm(H, axis=1, keepdims=True)
    return H


def _degrees(X, best, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    A = G >= best - tol
    return A.sum(axis=1).astype(int)


def solve():
    rng = np.random.RandomState(0)
    np.random.seed(0)
    t0 = time.time()
    DEADLINE = 172.0  # leave margin for 180s limit

    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xbest = np.load("record.npy").astype(float)
        except Exception:
            Xbest = rng.randn(323, 7)
    if Xbest.shape != (323, 7):
        # fallback: tile/truncate
        tmp = rng.randn(323, 7)
        m = min(len(Xbest), 323)
        tmp[:m] = Xbest[:m]
        Xbest = tmp
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best = _max_cos(Xbest)

    n, d = Xbest.shape
    trial = 0

    # main basin-hopping loop, reserve ~30s for final polish
    while time.time() - t0 < DEADLINE - 32.0:
        trial += 1
        rem = DEADLINE - 32.0 - (time.time() - t0)
        if rem < 3.0:
            break
        X = Xbest.copy()
        cur_best = best
        deg = _degrees(X, cur_best)
        order = np.argsort(-deg)
        ratt = np.where(deg == 0)[0]

        mode = trial % 4
        if mode == 1:
            k = 6
            hubs = order[:k]
            H = _best_holes(X, k, rng)
            H += 0.01 * rng.randn(*H.shape)
            H /= np.linalg.norm(H, axis=1, keepdims=True)
            X[hubs] = H
            X += 0.004 * rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 2:
            X += 0.020 * rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if len(ratt) > 0:
                k = len(ratt)
                H = _best_holes(X, k, rng)
                X[ratt] = H
            else:
                # relocate worst point to biggest hole
                w = order[0:1]
                H = _best_holes(X, 1, rng)
                X[w] = H
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 3:
            k = 12
            idx = rng.choice(n, k, replace=False)
            C = rng.randn(k, d)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            X[idx] = C
        else:
            X += 0.008 * rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # break current max pair strongly
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            ii, jj = ij // n, ij % n
            X[ii] += 0.08 * rng.randn(d)
            X[jj] += 0.08 * rng.randn(d)
            X[[ii, jj]] /= np.linalg.norm(X[[ii, jj]], axis=1, keepdims=True)

        # hinge repulsion toward just below current best
        target = cur_best - 0.003 - 0.002 * rng.rand()
        X = _hinge_smooth(X, target, iters=700, lr0=0.8)

        # short LP snap
        lp_t = min(4.0, max(1.5, rem * 0.15))
        # don't overshoot deadline reserve
        if time.time() + lp_t > t0 + DEADLINE - 30.0:
            lp_t = max(1.0, t0 + DEADLINE - 30.0 - time.time())
        X = _lp_polish(X, seconds=float(lp_t))
        v = _max_cos(X)
        if v < best - 1e-12:
            best = v
            Xbest = X

    # final long polish of champion
    rem = DEADLINE - (time.time() - t0) - 1.0
    if rem > 2.0:
        Xbest = _lp_polish(Xbest, seconds=float(min(rem, 28.0)))
        best = _max_cos(Xbest)

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

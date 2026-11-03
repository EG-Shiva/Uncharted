import time
import numpy as np
from scipy.optimize import minimize, linprog
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
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    TIME_BUDGET = 175.0
    # load seeds
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (162, 8):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (162, 8):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        rng0 = np.random.default_rng(0)
        a = rng0.standard_normal(size=(162, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    X_best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = 162, 8
    rng = np.random.default_rng(12345)

    SHIFT = 0.5

    def eg_fn(Yf, alpha):
        Y = Yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # exp(alpha*(G-SHIFT)), safe: G<=1, SHIFT=0.5
        M = alpha * (G - SHIFT)
        # clip lower to avoid underflow warnings (still 0)
        np.clip(M, -700, 200, out=M)
        Emat = np.exp(M)
        E = float(Emat.sum() * 0.5)
        GradX = alpha * (Emat @ X)
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        GradY = (GradX - dot * X) / nr
        return E, GradY.ravel()

    def staged_opt(X_init, alphas, maxiters):
        Yf = X_init.ravel().copy()
        for alpha, mi in zip(alphas, maxiters):
            if time.time() - t_start > TIME_BUDGET - 35:
                break
            try:
                res = minimize(lambda y: eg_fn(y, alpha), Yf, jac=True,
                               method="L-BFGS-B",
                               options={"maxiter": mi, "ftol": 1e-12,
                                        "gtol": 1e-5, "maxfun": mi * 20})
                Yf = res.x
            except Exception:
                break
        Y = Yf.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def greedy_replace(Xcur, idx_list, rng):
        Xnew = Xcur.copy()
        for w in idx_list:
            cand = rng.standard_normal(size=(2500, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            others = np.delete(Xnew, w, axis=0)
            scores = (cand @ others.T).max(axis=1)
            Xnew[w] = cand[np.argmin(scores)]
        return Xnew

    trial = 0
    no_improve = 0
    # explore until ~130s, reserve rest for LP
    while time.time() - t_start < 128:
        trial += 1
        mode = trial % 5
        X_try = X_best.copy()
        if mode == 1:
            sigma = 0.015 if no_improve < 12 else 0.045
            X_try = X_try + sigma * rng.standard_normal(size=X_try.shape)
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        elif mode == 2:
            k = 8
            idx = rng.choice(n, k, replace=False)
            X_try[idx] += 0.18 * rng.standard_normal(size=(k, d))
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        elif mode == 3:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            per_max = G.max(axis=1)
            worst = np.argsort(per_max)[-2:]
            X_try = greedy_replace(X_try, worst, rng)
        elif mode == 4:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            thresh = best_val - 0.006
            cont = (G > thresh).sum(axis=1)
            low = np.argsort(cont)[:3]
            X_try = greedy_replace(X_try, low, rng)
        else:
            sigma = 0.085
            X_try = X_try + sigma * rng.standard_normal(size=X_try.shape)
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)

        # choose smoothing strength
        if mode in (3, 4):
            alphas, mis = [40, 110], [70, 70]
        else:
            alphas, mis = [45, 130], [55, 55]
        X_opt = staged_opt(X_try, alphas, mis)
        v = _max_cos(X_opt)
        if v < best_val - 1e-12:
            X_best = X_opt
            best_val = v
            no_improve = 0
            # quick squeeze to lock in
            if time.time() - t_start < 105:
                X_lp = _lp_polish(X_best, 2.5)
                v2 = _max_cos(X_lp)
                if v2 < best_val:
                    X_best, best_val = X_lp, v2
        else:
            no_improve += 1
        # safety: if very little time left break
        if time.time() - t_start > TIME_BUDGET - 40:
            break

    remain = TIME_BUDGET - (time.time() - t_start) - 2.0
    if remain > 5:
        X_best = _lp_polish(X_best, remain)
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    # sanitize
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    return X_best

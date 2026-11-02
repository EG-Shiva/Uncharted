import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def top_pair(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    n = X.shape[0]
    i = k // n
    j = k % n
    return i, j, float(G[i, j])

def smooth_polish(X0, tau, maxiter):
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]
    def fun(y_flat):
        Y = y_flat.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        g = G[I0, J0]
        mmax = g.max()
        e = np.exp((g - mmax) / tau)
        S = e.sum()
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        Wx = np.zeros_like(X)
        # vectorized scatter
        Wx[I0] += w[:, None] * X[J0]
        Wx[J0] += w[:, None] * X[I0]
        dot = np.sum(Wx * X, axis=1, keepdims=True)
        grad_Y = (Wx - X * dot) / r
        return F, grad_Y.ravel()
    y0 = X0.ravel().copy()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-9, "maxfun": 10000})
        Y = res.x.reshape(n, d)
    except Exception:
        return X0
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    X = Y / nrm
    if not np.all(np.isfinite(X)):
        return X0
    return X

def lp_polish(X, seconds, radius_init=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Cfull = (X @ X.T)[iu]
        m = Cfull.max()
        sel = Cfull > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) > 2500:
            thr = np.sort(Cfull)[-2500]
            sel = Cfull >= thr
        I = iu[0][sel]; J = iu[1][sel]
        c = Cfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def directed_squeeze(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    best = max_cos(X)
    start = time.time()
    n, d = X.shape
    step = 2e-6
    while time.time() - start < seconds:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # find worst pair
        k = int(np.argmax(G))
        i = k // n; j = k % n
        c = float(G[i, j])
        xi = X[i]; xj = X[j]
        gi = xj - c * xi
        gj = xi - c * xj
        ni = np.linalg.norm(gi); nj = np.linalg.norm(gj)
        if ni < 1e-12 or nj < 1e-12:
            break
        gi /= ni; gj /= nj
        improved = False
        # try 3 moves: both, i only, j only
        for mode in (0, 1, 2):
            Y = X.copy()
            if mode == 0:
                Y[i] = xi - step * gi
                Y[j] = xj - step * gj
            elif mode == 1:
                Y[i] = xi - step * 2.0 * gi
            else:
                Y[j] = xj - step * 2.0 * gj
            Y[i] /= np.linalg.norm(Y[i]); Y[j] /= np.linalg.norm(Y[j])
            v = max_cos(Y)
            if v < best:
                X, best = Y, v
                improved = True
                break
        if improved:
            step = min(step * 1.2, 5e-5)
        else:
            step *= 0.5
            if step < 1e-11:
                # random tiny kick to change active pair
                break
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (74, 13):
        tmp = rng.standard_normal((74, 13))
        m = min(xb.shape[0], 74); k = min(xb.shape[1], 13)
        tmp[:m, :k] = xb[:m, :k]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    Xbest = xb.copy()

    n, d = 74, 13
    # initial micro squeeze of the single contact
    if DEADLINE - time.time() > 10:
        try:
            Xs = directed_squeeze(Xbest, 4.0)
            vs = max_cos(Xs)
            if vs < best_val:
                Xbest, best_val = Xs, vs
        except Exception:
            pass

    sigmas = [0.02, 0.05, 0.10, 0.005, 0.15, 0.01, 0.03, 0.20]
    taus = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    it = 0
    EXPLORE_END = t0 + 138.0
    while time.time() < EXPLORE_END:
        it += 1
        remain = DEADLINE - time.time()
        if remain < 8:
            break
        # choose strategy
        if it % 4 == 0:
            # hole relocation of worst point
            try:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                w = int(np.argmax(rowmax))
                # remaining points
                mask = np.ones(n, dtype=bool); mask[w] = False
                Xr = Xbest[mask]
                # sample holes
                K = 2500
                C = rng.standard_normal((K, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                # max cosine to remaining
                S = C @ Xr.T
                smax = S.max(axis=1)
                bh = int(np.argmin(smax))
                X0 = Xbest.copy()
                X0[w] = C[bh]
                # small jitter around hole
                X0[w] += 1e-3 * rng.standard_normal(d)
                X0[w] /= np.linalg.norm(X0[w])
                tau = taus[it % len(taus)]
                X1 = smooth_polish(X0, tau, 120)
                v1 = max_cos(X1)
                if v1 < best_val:
                    Xbest, best_val = X1, v1
                    continue
                # also try LP short on this proposal if promising
                if v1 < best_val + 0.005:
                    X2 = lp_polish(X1, 1.5, radius_init=5e-4)
                    v2 = max_cos(X2)
                    if v2 < best_val:
                        Xbest, best_val = X2, v2
            except Exception:
                pass
        else:
            try:
                sigma = sigmas[it % len(sigmas)]
                # targeted perturbation: more noise on high-contact points
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                mx = G.max()
                # points involved in near-max pairs
                cnt = np.sum(G > mx - 2e-4, axis=1)
                scale = (np.ones(n) + 4.0 * (cnt > 0))[:, None] * sigma
                if it % 9 == 0:
                    # global restart from random occasionally
                    X0 = rng.standard_normal((n, d))
                    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
                else:
                    noise = rng.standard_normal((n, d))
                    # project noise to tangent to avoid radial waste
                    dot = np.sum(noise * Xbest, axis=1, keepdims=True)
                    noise = noise - Xbest * dot
                    X0 = Xbest + scale * noise
                    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
                tau = taus[(it + (it // 3)) % len(taus)]
                maxiter = 150 if sigma < 0.06 else 100
                X1 = smooth_polish(X0, tau, maxiter)
                v1 = max_cos(X1)
                if v1 < best_val:
                    Xbest, best_val = X1, v1
                elif it % 3 == 0 and v1 < best_val + 0.01:
                    # squeeze attempt
                    X2 = directed_squeeze(X1, 1.0)
                    v2 = max_cos(X2)
                    if v2 < best_val:
                        Xbest, best_val = X2, v2
            except Exception:
                pass
        # periodic greedy acceptance of tiny jitter hill-climb
        if it % 6 == 0 and time.time() < EXPLORE_END:
            try:
                for _ in range(20):
                    s = 1e-5 * (0.5 ** (rng.random() * 4))
                    Y = Xbest + s * rng.standard_normal((n, d))
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                    v = max_cos(Y)
                    if v < best_val:
                        Xbest, best_val = Y, v
                        break
            except Exception:
                pass

    # final polishing: squeeze + LP + smooth with tiny tau
    try:
        if time.time() < DEADLINE - 25:
            Xbest = smooth_polish(Xbest, 0.0003, 300)
            best_val = max_cos(Xbest)
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 20:
            Xbest = directed_squeeze(Xbest, 6.0)
            best_val = max_cos(Xbest)
    except Exception:
        pass
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 3:
            Xp = lp_polish(Xbest, remain, radius_init=1e-3)
            vp = max_cos(Xp)
            if vp < best_val:
                Xbest, best_val = Xp, vp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # tiny shrink guard not needed; ensure finite
    Xbest = np.where(np.isfinite(Xbest), Xbest, rng.standard_normal((74, 13)))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

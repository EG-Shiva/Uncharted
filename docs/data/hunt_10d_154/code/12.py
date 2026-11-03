import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def lp_polish(x, seconds):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = _normalize(np.asarray(xb, dtype=float))
    n, d = xb.shape
    # ensure shape (154,10)
    if xb.shape != (154, 10):
        # fallback: random
        r = rng.normal(size=(154, 10))
        xb = _normalize(r)
        n, d = xb.shape

    bestX = xb.copy()
    best_score = _max_cos(bestX)

    iu0, iu1 = np.triu_indices(n, 1)
    lam = 1.0

    def find_hole(X, exclude, n_cand=3000):
        R = rng.normal(size=(n_cand, d))
        R = _normalize(R)
        D = R @ X.T
        D[:, exclude] = -2.0
        m = D.max(axis=1)
        return R[int(np.argmin(m))]

    def smooth_refine(X_init, shift, Ts, iters):
        Y_flat = X_init.ravel().copy()
        cur_shift = float(shift)
        for T, mit in zip(Ts, iters):
            # closure capturing T, cur_shift
            def fun(Yf, T=T, sh=cur_shift):
                Y = Yf.reshape(n, d)
                norms = np.linalg.norm(Y, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X = Y / norms
                G = X @ X.T
                g = G[iu0, iu1]
                z = (g - sh) / T
                mz = float(z.max())
                e = np.exp(z - mz)
                s = float(e.sum())
                f = sh + T * (mz + np.log(s + 1e-300))
                w = e / (s + 1e-300)
                W = np.zeros((n, n))
                W[iu0, iu1] = w
                W = W + W.T
                GX = W @ X
                dot = (GX * X).sum(axis=1, keepdims=True)
                grad = (GX - dot * X) / norms
                diff = norms[:, 0] - 1.0
                f = f + 0.5 * lam * float(np.mean(diff ** 2))
                grad = grad + ((lam * diff / n)[:, None] * X)
                return f, grad.ravel()
            try:
                res = minimize(fun, Y_flat, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mit), "ftol": 1e-12, "gtol": 1e-6, "maxcor": 10})
                Y_flat = res.x
            except Exception:
                break
            # update shift to current max
            try:
                Y = Y_flat.reshape(n, d)
                Xc = _normalize(Y)
                Gc = Xc @ Xc.T
                np.fill_diagonal(Gc, -2.0)
                cur_shift = float(Gc.max())
            except Exception:
                pass
            if time.time() - t0 > 120:
                break
        Y = Y_flat.reshape(n, d)
        return _normalize(Y)

    Ts = [0.02, 0.008, 0.003]
    Iters = [40, 40, 60]

    trial = 0
    sigmas = [0.03, 0.07, 0.12, 0.20]
    # quick first refine without shake (may help marginally)
    # do it as trial 0 inside loop
    while time.time() - t0 < 115:
        trial += 1
        if time.time() - t0 > 112:
            break
        G0 = bestX @ bestX.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        rmax = G0.max(axis=1)
        is_rattler = rmax < (m0 - 0.005)
        sigma = sigmas[(trial // 4) % len(sigmas)]
        mod = trial % 4
        X0 = bestX.copy()
        if mod == 0:
            noise = rng.normal(size=X0.shape) * sigma
            noise[is_rattler] *= 0.2
            X0 = _normalize(X0 + noise)
        elif mod == 1:
            idx = np.argsort(-rmax)[:10]
            noise = rng.normal(size=X0.shape) * (sigma * 0.5)
            noise[idx] *= 3.0
            noise[is_rattler] *= 0.2
            X0 = _normalize(X0 + noise)
        elif mod == 2:
            top5 = np.argsort(-rmax)[:5]
            worst = int(top5[rng.randint(len(top5))])
            hole = find_hole(X0, worst, n_cand=3000)
            hole = hole + rng.normal(size=d) * 0.01
            hole = hole / (np.linalg.norm(hole) + 1e-12)
            X0[worst] = hole
            noise = rng.normal(size=X0.shape) * (sigma * 0.3)
            noise[worst] = 0.0
            noise[is_rattler] *= 0.2
            X0 = _normalize(X0 + noise)
            # re-enforce hole (avoid drift of relocated point)
            # keep as is after normalize; already perturbed slightly
        else:
            # worst pair relocation
            k = int(np.argmax(G0[iu0, iu1]))
            a, b = int(iu0[k]), int(iu1[k])
            # randomly pick one of the two, biased to higher rmax
            pick = a if (rmax[a] >= rmax[b]) else b
            if rng.rand() < 0.25:
                pick = b if pick == a else a
            hole = find_hole(X0, pick, n_cand=3000)
            hole = hole + rng.normal(size=d) * 0.01
            hole = hole / (np.linalg.norm(hole) + 1e-12)
            X0[pick] = hole
            noise = rng.normal(size=X0.shape) * (sigma * 0.3)
            noise[pick] = 0.0
            X0 = _normalize(X0 + noise)

        Xopt = smooth_refine(X0, m0, Ts, Iters)
        sc = _max_cos(Xopt)
        if sc < best_score and np.all(np.isfinite(Xopt)):
            best_score = sc
            bestX = Xopt.copy()
        # early exit if time low
        if time.time() - t0 > 115:
            break

    # final LP polish with remaining budget
    elapsed = time.time() - t0
    remain = 168.0 - elapsed
    if remain > 6:
        secs = min(38.0, remain - 2.0)
        try:
            bestX = lp_polish(bestX, secs)
        except Exception:
            pass
    bestX = _normalize(bestX)
    # safety: replace any bad rows
    norms = np.linalg.norm(bestX, axis=1)
    bad = ~(np.isfinite(norms) & (norms > 1e-8))
    if np.any(bad):
        R = rng.normal(size=(int(bad.sum()), d))
        bestX[bad] = _normalize(R)
    return bestX

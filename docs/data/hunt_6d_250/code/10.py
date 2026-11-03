import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_opt(X0, beta, maxiter):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        D = beta * (G - M)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0:
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                            "maxfun": maxiter * 5, "maxcor": 15})
    Y = res.x.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20 * radius, 2e-9)
        if np.count_nonzero(sel) > 5000:
            thr = np.sort(iu_c[sel])[-5000]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((250, 6))
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (250, 6):
        tmp = rng.standard_normal((250, 6))
        try:
            m = min(x0.shape[0], 250)
            dd = min(x0.shape[1], 6)
            tmp[:m, :dd] = x0[:m, :dd]
        except Exception:
            pass
        x0 = tmp
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=np.float64)
        if rec.shape == (250, 6):
            rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
            if _max_cos(rec) < _max_cos(x0):
                x0 = rec.copy()
    except Exception:
        pass

    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    def remaining():
        return TIME_BUDGET - (time.time() - t_start)

    # quick initial polish at moderate-high beta
    if remaining() > 40:
        try:
            c = _smooth_opt(best, 8000.0, 200)
            v = _max_cos(c)
            if v < best_val:
                best, best_val = c.copy(), v
                cur, cur_val = c.copy(), v
        except Exception:
            pass

    n, d = 250, 6
    trial = 0
    sigmas = [0.012, 0.006, 0.003, 0.020, 0.008, 0.0015]
    while remaining() > 28 and trial < 80:
        trial += 1
        mode = trial % 6
        # base: mostly from best, sometimes from cur to walk
        base = best if (trial % 3 != 0) else cur
        try:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
        except Exception:
            order = np.arange(n)
        Y = base.copy()
        if mode == 4 or mode == 2 and trial % 2 == 0:
            # void reinsertion of 1-2 worst points
            nw = 1 + (trial % 2)
            victims = list(order[:nw])
            for w in victims:
                cand = rng.standard_normal((48, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                sc = cand @ Y.T
                # ignore self column
                sc[:, w] = -2.0
                mx = sc.max(axis=1)
                bi = int(np.argmin(mx))
                Y[w] = cand[bi]
            # tiny jitter rest
            Y += rng.standard_normal(Y.shape) * 0.0015
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            beta1, it1 = 4000.0, 120
        elif mode == 5:
            K = 10 + (trial % 12)
            idx = list(order[:K])
            sigma = sigmas[trial % len(sigmas)] * 2.0
            Y[idx] += rng.standard_normal((len(idx), d)) * sigma
            # move rattlers slightly too
            Y += rng.standard_normal(Y.shape) * 0.0008
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            beta1, it1 = 3000.0, 120
        else:
            sigma = sigmas[trial % len(sigmas)]
            K = 3 + (trial % 9)
            victims = list(order[:K])
            if trial % 2 == 1:
                victims += list(rng.integers(0, n, size=4))
            victims = list(set(victims))
            Y[victims] += rng.standard_normal((len(victims), d)) * sigma
            if trial % 7 == 0:
                Y += rng.standard_normal(Y.shape) * 0.001
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            beta1 = 8000.0 if (trial % 3 == 0) else 3500.0
            it1 = 130
        try:
            Y1 = _smooth_opt(Y, beta1, it1)
        except Exception:
            continue
        v1 = _max_cos(Y1)
        # update walk state
        if v1 < cur_val:
            cur, cur_val = Y1.copy(), v1
        elif v1 < best_val + 0.002 and rng.random() < 0.15:
            cur, cur_val = Y1.copy(), v1
        if v1 < best_val:
            best, best_val = Y1.copy(), v1
            cur, cur_val = Y1.copy(), v1
            continue
        # second-stage high-beta squeeze if promising
        if v1 < best_val + 0.0015 and remaining() > 30:
            try:
                Y2 = _smooth_opt(Y1, 20000.0, 70)
            except Exception:
                continue
            v2 = _max_cos(Y2)
            if v2 < best_val:
                best, best_val = Y2.copy(), v2
                cur, cur_val = Y2.copy(), v2
            elif v2 < cur_val:
                cur, cur_val = Y2.copy(), v2
        if remaining() < 4:
            break

    # final chain on best
    if remaining() > 10:
        try:
            c = _smooth_opt(best, 15000.0, 150)
            v = _max_cos(c)
            if v < best_val:
                best, best_val = c.copy(), v
        except Exception:
            pass
    if remaining() > 8:
        try:
            c = _smooth_opt(best, 40000.0, 100)
            v = _max_cos(c)
            if v < best_val:
                best, best_val = c.copy(), v
        except Exception:
            pass
    if remaining() > 6:
        try:
            polished = _lp_polish(best, max(remaining() - 2.0, 1.0), time.time())
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best

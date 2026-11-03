import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_opt(X0, beta, maxiter, timelimit=1e9):
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
        # threshold tiny exps for speed / stability
        # exp(-30) ~ 9e-14 negligible
        E = np.exp(D, out=np.empty_like(D))
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

    # L-BFGS-B with tight tolerances
    res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                            "maxfun": maxiter * 5, "maxcor": 20})
    Y = res.x.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20 * radius, 2e-9)
        # cap number of constraints for speed: keep tightest 4000
        if np.count_nonzero(sel) > 4000:
            thr = np.sort(iu_c[sel])[-4000]
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    np.random.seed(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            x0 = rng0.standard_normal((250, 6))
    x0 = np.asarray(x0, dtype=np.float64)
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    # ensure shape (250,6)
    if x0.shape != (250, 6):
        rng0 = np.random.default_rng(0)
        tmp = rng0.standard_normal((250, 6))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        # copy what we can
        m = min(x0.shape[0], 250)
        d = min(x0.shape[1], 6)
        tmp[:m, :d] = x0[:m, :d]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        x0 = tmp

    best = x0.copy()
    best_val = _max_cos(best)

    def remaining():
        return TIME_BUDGET - (time.time() - t_start)

    # Stage 1: annealed smoothing chain
    # betas increasing, iters moderate
    schedule = [
        (300.0, 250),
        (1000.0, 300),
        (3000.0, 300),
        (8000.0, 300),
        (20000.0, 250),
    ]
    cur = best.copy()
    for beta, iters in schedule:
        if remaining() < 25:
            break
        try:
            cur = _smooth_opt(cur, beta, iters)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best_val = v
            best = cur.copy()
        # small random kick to avoid exact saddle when beta jumps
        if remaining() < 5:
            break

    # Stage 2: shake-and-reoptimize basin hopping
    rng = np.random.default_rng(0)
    # identify worst-involved points from best for targeted shakes
    trial = 0
    while remaining() > 30 and trial < 60:
        trial += 1
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # top pairs
        # count involvement
        thresh = m - 5e-4 if trial % 3 == 0 else m - 2e-3
        # fallback: use top 2000 pairs
        flat = G[np.triu_indices(250, 1)]
        # adaptive threshold to get ~2000 pairs
        if trial % 2 == 0:
            kth = np.sort(flat)[-2000]
            thresh = min(thresh, kth)
        W = (G >= thresh).astype(np.float64)
        np.fill_diagonal(W, 0)
        deg = W.sum(axis=1)
        order = np.argsort(-deg)
        # choose sigma and k
        sigmas = [0.015, 0.007, 0.003, 0.0015]
        sigma = sigmas[trial % len(sigmas)]
        # number of points to shake: shake worst + random rattlers
        k = 4 + (trial % 8)
        victims = list(order[:k])
        # add some random indices to move rattlers
        if trial % 2 == 1:
            victims += list(rng.integers(0, 250, size=4))
        victims = list(set(victims))
        Y = best.copy()
        noise = rng.standard_normal((len(victims), 6)) * sigma
        Y[victims] += noise
        # occasionally global tiny shake
        if trial % 7 == 0:
            Y += rng.standard_normal(Y.shape) * 0.001
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # re-polish with moderate beta (short)
        beta = 3000.0 if trial % 3 == 0 else 8000.0
        try:
            Y2 = _smooth_opt(Y, beta, 150)
        except Exception:
            continue
        # optional second polish at higher beta
        if remaining() > 35:
            try:
                Y2 = _smooth_opt(Y2, 20000.0, 80)
            except Exception:
                pass
        v2 = _max_cos(Y2)
        if v2 < best_val:
            best_val = v2
            best = Y2.copy()
            cur = Y2.copy()
        if remaining() < 5:
            break

    # Stage 3: final LP trust-region squeeze
    if remaining() > 8:
        try:
            polished = _lp_polish(best, min(remaining() - 2.0, 25.0), time.time())
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # guard against nan
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best

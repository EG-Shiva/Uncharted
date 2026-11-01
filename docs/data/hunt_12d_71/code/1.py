import time
import numpy as np

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end_limit=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 3e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_end_limit is not None and time.time() > t_end_limit:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_opt(X0, betas, iters=150):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = X0.copy()
    # normalize to start
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    for beta in betas:
        def fun(Yflat, beta=beta):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            gv = G[iu]
            gm = gv.max()
            e = np.exp(beta * (gv - gm))
            S = e.sum()
            F = gm + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            gradX = W @ X
            dot = (gradX * X).sum(axis=1, keepdims=True)
            gradY = (gradX - dot * X) / rn
            # keep norms ~1
            r1 = rn.squeeze() - 1.0
            pen = float((r1 ** 2).mean())
            Ftot = float(F) + pen
            # grad of pen: 2(r-1)/n * x
            gradY = gradY + (2.0 / n) * r1[:, None] * X / 1.0
            # note X already = Ym/rn, correction for pen grad w.r.t Ym is same direction
            return Ftot, gradY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 20})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 172.0
    # load start
    try:
        Xbest = np.load("best.npy")
    except Exception:
        try:
            Xbest = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            Xbest = rng0.standard_normal(size=(71, 12))
    Xbest = np.asarray(Xbest, dtype=float)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best = max_cos(Xbest)
    n, d = Xbest.shape
    rng = np.random.default_rng(0)
    # precompute contact info helper
    trial = 0
    # explore until ~140s
    EXPLORE_END = t0 + 140.0
    sigmas = [0.04, 0.10, 0.22, 0.02, 0.15, 0.06]
    while _time.time() < EXPLORE_END and _time.time() < HARD_END - 25 and trial < 200:
        trial += 1
        X0 = Xbest.copy()
        rtype = rng.random()
        if rtype < 0.55:
            sigma = sigmas[trial % len(sigmas)]
            # boost overloaded points
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            cnt = (G > best - 1e-9).sum(axis=1)
            w = np.ones(n)
            # top overloaded get 2x kick
            thresh = np.sort(cnt)[-8] if n >= 8 else cnt.max()
            w[cnt >= thresh] = 2.0
            w[mx > best - 5e-7] *= 1.5
            noise = rng.standard_normal(size=X0.shape)
            X0 = X0 + sigma * noise * w[:, None]
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif rtype < 0.85:
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            cnt = (G > best - 1e-9).sum(axis=1)
            # pick victim among top-5 by cnt (randomized)
            order = np.argsort(-cnt)
            cand_v = order[:5]
            victim = cand_v[rng.integers(0, len(cand_v))]
            R = np.delete(X0, victim, axis=0)
            cand = rng.standard_normal(size=(2500, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            scores = (cand @ R.T).max(axis=1)
            X0[victim] = cand[np.argmin(scores)]
            X0 = X0 + 0.015 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            k = 4 + int(rng.integers(0, 3))  # 4-6 points
            idx = rng.choice(n, k, replace=False)
            keep = np.delete(np.arange(n), idx)
            R = X0[keep]
            for j in idx:
                cand = rng.standard_normal(size=(2500, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                scores = (cand @ R.T).max(axis=1)
                X0[j] = cand[np.argmin(scores)]
                R = np.vstack([R, X0[j][None, :]])
            X0 = X0 + 0.01 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        if trial % 3 == 0:
            betas = [40.0, 120.0, 300.0]
        elif trial % 3 == 1:
            betas = [60.0, 200.0]
        else:
            betas = [30.0, 80.0, 160.0, 350.0]
        # fewer iters for large sigma trials to save time
        iters = 120 if trial % 2 == 0 else 180
        Xopt = _smooth_opt(X0, betas, iters=iters)
        v = max_cos(Xopt)
        if v < best - 1e-12:
            # quick polish to lock in
            Xopt2 = _lp_polish(Xopt, seconds=2.0, t_end_limit=HARD_END - 12)
            v2 = max_cos(Xopt2)
            if v2 < best:
                Xbest, best = Xopt2, v2
            else:
                Xbest, best = Xopt, v
        elif v < best + 0.002:
            # promising: short polish
            # only polish every other to save time
            if trial % 2 == 0 or v < best + 0.0005:
                Xopt2 = _lp_polish(Xopt, seconds=1.5, t_end_limit=HARD_END - 12)
                v2 = max_cos(Xopt2)
                if v2 < best:
                    Xbest, best = Xopt2, v2
        # timeout guard
        if _time.time() > HARD_END - 25:
            break
    # final polish with remaining time
    remain = HARD_END - _time.time() - 2.0
    if remain > 3:
        Xbest = _lp_polish(Xbest, seconds=min(remain, 28.0))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

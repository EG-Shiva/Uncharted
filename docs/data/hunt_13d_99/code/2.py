import time
import numpy as np

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-15)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, time_budget, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < time_budget and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _lbfgs_smooth(x_init, k, maxiter, deadline):
    from scipy.optimize import minimize
    n, d = x_init.shape
    Y0 = x_init.ravel().copy()
    iu0 = np.triu_indices(n, 1)

    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = Y / norms
        G = X @ X.T
        gv = G[iu0]
        m = gv.max()
        e = np.exp(k * (gv - m))
        S = e.sum()
        loss = m + np.log(S) / k
        wv = e / S
        W = np.zeros((n, n))
        W[iu0] = wv
        W = W + W.T
        grad_X = W @ X
        dot = (grad_X * X).sum(axis=1, keepdims=True)
        grad_Y = (grad_X - dot * X) / norms
        return float(loss), grad_Y.ravel()

    try:
        res = minimize(fg, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12,
                                "gtol": 1e-8, "maxfun": maxiter * 5})
        Yopt = res.x.reshape(n, d)
        Xopt = _normalize(Yopt)
        return Xopt
    except Exception:
        return x_init

def _hole_improve(x, rng, n_try=4000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    cur_max = float(G.max())
    per = G.max(axis=1)
    order = np.argsort(-per)[:5]
    for idx in order:
        ii = int(idx)
        others = np.delete(x, ii, axis=0)
        cand = rng.randn(n_try, d)
        cand = _normalize(cand)
        dots = cand @ others.T
        s = dots.max(axis=1)
        j = int(np.argmin(s))
        if s[j] >= per[ii] - 1e-7:
            continue
        v = cand[j].copy()
        for it in range(200):
            dts = others @ v
            m = dts.max()
            mask = dts > m - 0.02
            if not np.any(mask):
                break
            grad = others[mask].sum(axis=0)
            grad_t = grad - (grad @ v) * v
            gn = np.linalg.norm(grad_t)
            if gn < 1e-12:
                break
            grad_t /= gn
            step = 0.01 / (1 + it * 0.02)
            v = v - step * grad_t
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                break
            v /= nv
        fmax = float((others @ v).max())
        Go = others @ others.T
        np.fill_diagonal(Go, -2.0)
        omax = float(Go.max())
        nglob = fmax if fmax > omax else omax
        if nglob < cur_max - 1e-9:
            x[ii] = v
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cur_max = float(G.max())
            per = G.max(axis=1)
    return x, cur_max

def solve():
    import time
    t_start = time.time()
    deadline = t_start + 165.0
    np.random.seed(0)
    rng = np.random.RandomState(1)

    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(99, 13)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        # reshape / pad deterministically if needed
        flat = xb.ravel()
        need = 99 * 13
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(99, 13)
    xb = _normalize(xb)
    best = _max_cos(xb)
    x_best = xb.copy()

    # quick check: if lots of time, keep best
    sigmas_all = [0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.12, 0.20]
    k_schedule = [[40, 120, 300], [80, 250], [30, 100, 250], [60, 200]]
    trial = 0
    # reserve ~15s for final polish
    while time.time() < deadline - 15.0 and trial < 200:
        sig = sigmas_all[trial % len(sigmas_all)]
        ks = k_schedule[trial % len(k_schedule)]
        # perturb
        xp = x_best + sig * rng.randn(*x_best.shape)
        # extra kick to a few points, occasionally full reinsert
        r = rng.rand()
        if r < 0.35:
            kk = int(rng.randint(1, 4))
            idx = rng.choice(99, kk, replace=False)
            xp[idx] += 0.35 * rng.randn(kk, 13)
        elif r < 0.55:
            kk = int(rng.randint(1, 3))
            idx = rng.choice(99, kk, replace=False)
            rnd = rng.randn(kk, 13)
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            xp[idx] = rnd + 0.05 * rng.randn(kk, 13)
        xp = _normalize(xp)
        # smooth stages
        for k in ks:
            if time.time() > deadline - 15.0:
                break
            # adapt maxiter to remaining time / trial
            xp = _lbfgs_smooth(xp, k, maxiter=120, deadline=deadline)
        cur = _max_cos(xp)
        # only spend LP time if promising
        if cur < best + 0.015 and time.time() < deadline - 15.0:
            xp, _ = _hole_improve(xp, rng, n_try=3000)
            # short LP
            xp, v = _lp_polish(xp, time_budget=2.0, deadline=deadline - 12.0)
            cur = _max_cos(xp)
            if cur < best - 1e-12:
                best = cur
                x_best = xp.copy()
        trial += 1

    # final long polish of best
    remain = deadline - time.time()
    if remain > 3.0:
        # a couple of hole attempts + LP on best
        try:
            x_best, _ = _hole_improve(x_best, rng, n_try=6000)
        except Exception:
            pass
        try:
            x_best, best = _lp_polish(x_best, time_budget=min(remain - 0.5, 20.0),
                                      deadline=deadline)
        except Exception:
            pass
    x_best = _normalize(x_best)
    return x_best

import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end_limit=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _adam_quench(X0, beta, steps, lr, rng):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # decay lr linearly
    for t in range(1, steps + 1):
        G = X @ X.T
        gv = G[iu]
        gm = gv.max()
        e = np.exp(beta * (gv - gm))
        s = e.sum()
        w = e / s
        # build W@X without full matrix
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        grad = W @ X
        proj = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        cur_lr = lr * (1.0 - 0.5 * t / steps)
        X = X - cur_lr * mh / (np.sqrt(vh) + eps)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _smooth_lbfgs(X0, betas, iters_list, best_ref=1e9):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = X0.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    for beta, it in zip(betas, iters_list):
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
            r1 = rn.squeeze() - 1.0
            pen = float((r1 ** 2).mean())
            Ftot = float(F) + pen
            gradY = gradY + (2.0 / n) * r1[:, None] * X
            return Ftot, gradY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': int(it), 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 12})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # early prune: if far from ref, stop
        if best_ref < 0.5:
            Gc = Y @ Y.T
            np.fill_diagonal(Gc, -2.0)
            if float(Gc.max()) > best_ref + 0.02:
                break
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def _farthest_hole(R, rng, ncand=5000):
    d = R.shape[1]
    cand = rng.standard_normal(size=(ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ R.T).max(axis=1)
    return cand[np.argmin(scores)]

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 172.0
    try:
        Xbest = np.load("best.npy")
    except Exception:
        try:
            Xbest = np.load("record.npy")
        except Exception:
            r0 = np.random.default_rng(0)
            Xbest = r0.standard_normal(size=(71, 12))
    Xbest = np.asarray(Xbest, dtype=float)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best = _max_cos(Xbest)
    n, d = Xbest.shape
    rng = np.random.default_rng(20261101)
    iu_all = np.triu_indices(n, 1)
    EXPLORE_END = t0 + 138.0
    trial = 0
    # beta schedules
    scheds = [
        ([70.0, 250.0], [60, 70]),
        ([50.0, 150.0, 400.0], [50, 60, 60]),
        ([100.0, 350.0], [60, 80]),
        ([30.0, 90.0, 220.0], [40, 50, 60]),
    ]
    while _time.time() < EXPLORE_END and _time.time() < HARD_END - 28 and trial < 600:
        trial += 1
        X0 = Xbest.copy()
        u = rng.random()
        # contact counts for targeting
        try:
            Gc = X0 @ X0.T
            np.fill_diagonal(Gc, -2.0)
            cnt = (Gc > best - 5e-4).sum(axis=1)
            mxr = Gc.max(axis=1)
        except Exception:
            cnt = np.zeros(n); mxr = np.zeros(n)
        if u < 0.30:
            # hub-targeted gaussian shake
            sigma = [0.05, 0.12, 0.22, 0.03, 0.08][trial % 5]
            w = np.ones(n)
            # boost top overloaded
            order = np.argsort(-cnt)
            w[order[:8]] *= 2.0
            w[mxr > best - 1e-6] *= 1.4
            noise = rng.standard_normal(size=X0.shape)
            X0 = X0 + sigma * noise * w[:, None]
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.55:
            # single kill-refill (hub) + jitter
            order = np.argsort(-cnt - 0.5 * (mxr - mxr.min()) / (mxr.max() - mxr.min() + 1e-12))
            k = int(rng.integers(0, min(5, n)))
            victim = int(order[k])
            R = np.delete(X0, victim, axis=0)
            X0[victim] = _farthest_hole(R, rng, ncand=6000)
            X0 = X0 + 0.012 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.78:
            # multi kill 3-5 sequential hole fill
            k = 3 + int(rng.integers(0, 3))
            order = np.argsort(-cnt)
            # randomize among top-12
            pool = order[:12].copy()
            rng.shuffle(pool)
            victims = np.sort(pool[:k])
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            R = X0[keep_mask]
            for vv in victims:
                nv = _farthest_hole(R, rng, ncand=4000)
                X0[vv] = nv
                R = np.vstack([R, nv[None, :]])
            X0 = X0 + 0.008 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # coherent cap rotation + noise
            du = rng.standard_normal(size=(d,))
            du /= np.linalg.norm(du)
            dots = X0 @ du
            # cap size 6-14 points
            cap = 6 + int(rng.integers(0, 9))
            idx = np.argsort(-dots)[:cap]
            # random plane
            e1 = rng.standard_normal(size=(d,)); e1 /= np.linalg.norm(e1)
            tmp = rng.standard_normal(size=(d,))
            tmp = tmp - (tmp @ e1) * e1
            ne = np.linalg.norm(tmp)
            if ne < 1e-9:
                tmp = rng.standard_normal(size=(d,)); tmp -= (tmp @ e1) * e1
                ne = np.linalg.norm(tmp)
            e2 = tmp / ne
            theta = float(rng.normal(0, 0.18))
            if abs(theta) > 0.5:
                theta = 0.5 * np.sign(theta)
            ct, st = np.cos(theta), np.sin(theta)
            Xc = X0[idx]
            a = Xc @ e1
            b = Xc @ e2
            ap = ct * a - st * b
            bp = st * a + ct * b
            Xc = Xc + (ap - a)[:, None] * e1[None, :] + (bp - b)[:, None] * e2[None, :]
            # small noise
            Xc = Xc + 0.02 * rng.standard_normal(size=Xc.shape)
            X0[idx] = Xc
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # choose schedule
        sc = scheds[trial % len(scheds)]
        betas, iters = sc
        # fast adam pre-quench
        try:
            abeta = 80.0 if (trial % 2 == 0) else 50.0
            steps = 250 if u < 0.78 else 350
            lr = 0.02 if u < 0.5 else 0.015
            X0 = _adam_quench(X0, abeta, steps, lr, rng)
        except Exception:
            pass
        # quick check: if adam already terrible, skip lbfgs? still do short
        try:
            Xopt = _smooth_lbfgs(X0, betas, iters, best_ref=best)
        except Exception:
            continue
        v = _max_cos(Xopt)
        if v < best - 1e-12:
            Xopt2 = _lp_polish(Xopt, seconds=2.0, t_end_limit=HARD_END - 14)
            v2 = _max_cos(Xopt2)
            if v2 < best:
                Xbest, best = Xopt2, v2
            else:
                Xbest, best = Xopt, v
            # update cnt baseline implicitly next iter
        elif v < best + 0.0015:
            if (trial % 3 == 0) or (v < best + 0.0004):
                Xopt2 = _lp_polish(Xopt, seconds=1.2, t_end_limit=HARD_END - 14)
                v2 = _max_cos(Xopt2)
                if v2 < best - 1e-13:
                    Xbest, best = Xopt2, v2
        if _time.time() > HARD_END - 28:
            break
    remain = HARD_END - _time.time() - 2.0
    if remain > 3:
        try:
            Xbest = _lp_polish(Xbest, seconds=min(remain, 26.0))
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

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

def _smooth_opt(X0, betas, iters=120):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    Y = Y.copy()
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
            r1 = rn.squeeze() - 1.0
            pen = float((r1 ** 2).mean())
            Ftot = float(F) + pen
            gradY = gradY + (2.0 / n) * r1[:, None] * X
            return Ftot, gradY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 20})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def _refine_insertion(R, r0, beta=250.0, iters=60):
    from scipy.optimize import minimize
    m, d = R.shape
    y0 = r0.copy()
    def fun(yflat):
        y = yflat
        rn = max(np.linalg.norm(y), 1e-12)
        x = y / rn
        s = R @ x
        sm = s.max()
        e = np.exp(beta * (s - sm))
        S = e.sum()
        F = sm + np.log(S) / beta
        w = e / S
        gx = w @ R
        dot = float(gx @ x)
        gy = (gx - dot * x) / rn
        r1 = rn - 1.0
        Ftot = float(F) + r1 * r1
        gy = gy + 2.0 * r1 * x
        return Ftot, gy
    try:
        res = minimize(fun, y0, method='L-BFGS-B', jac=True,
                       options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-8})
        y = res.x
        y = y / np.linalg.norm(y)
        return y
    except Exception:
        return r0 / np.linalg.norm(r0)

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
            rng0 = np.random.default_rng(0)
            Xbest = rng0.standard_normal(size=(71, 12))
    Xbest = np.asarray(Xbest, dtype=float)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best = _max_cos(Xbest)
    n, d = Xbest.shape
    rng = np.random.default_rng(1)
    iu_all = np.triu_indices(n, 1)

    def contacts_count(X, thr):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return (G > thr).sum(axis=1), G

    EXPLORE_END = t0 + 138.0
    trial = 0
    sigmas = [0.03, 0.07, 0.14, 0.025, 0.05, 0.10, 0.20]
    while _time.time() < EXPLORE_END and _time.time() < HARD_END - 28 and trial < 300:
        trial += 1
        X0 = Xbest.copy()
        u = rng.random()
        if u < 0.35:
            sigma = sigmas[trial % len(sigmas)]
            cnt, G = contacts_count(X0, best - 1e-9)
            order = np.argsort(-cnt)
            w = np.ones(n)
            # boost top hubs
            nhub = 8
            w[order[:nhub]] *= 2.0
            noise = rng.standard_normal(size=X0.shape)
            # make tangent
            proj = (noise * X0).sum(axis=1, keepdims=True)
            noise = noise - proj * X0
            X0 = X0 + sigma * noise * w[:, None]
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.55:
            # hub cluster collective shift
            cnt, G = contacts_count(X0, best - 1e-9)
            order = np.argsort(-cnt)
            cand = order[:6]
            h = cand[rng.integers(0, len(cand))]
            neigh = np.where(G[h] > best - 0.02)[0]
            if len(neigh) < 2:
                neigh = np.where(G[h] > best - 0.05)[0]
            cluster = np.concatenate(([h], neigh))
            cluster = np.unique(cluster)
            z = rng.standard_normal(size=(d,))
            zt = z - float(z @ X0[h]) * X0[h]
            nz = np.linalg.norm(zt)
            if nz < 1e-12:
                zt = rng.standard_normal(size=(d,)) * 0.1
            else:
                zt = zt / nz
            step = 0.05 + 0.06 * rng.random()
            for idx in cluster:
                X0[idx] = X0[idx] + step * zt
            # individual jitter
            noise = rng.standard_normal(size=X0.shape) * 0.02
            proj = (noise * X0).sum(axis=1, keepdims=True)
            noise = noise - proj * X0
            X0 = X0 + noise
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.75:
            # worst-pair repulsion shake to break contacts
            steps = 150 + int(rng.integers(0, 150))
            eps = 0.008 + 0.008 * rng.random()
            for s in range(steps):
                G = X0 @ X0.T
                np.fill_diagonal(G, -2.0)
                k = int(np.argmax(G))
                ii = k // n; jj = k % n
                # handle ordering for triu argmax over full matrix: fine
                if ii == jj:
                    break
                xi = X0[ii]; xj = X0[jj]
                dij = float(xi @ xj)
                xi_t = xj - dij * xi
                xj_t = xi - dij * xj
                X0[ii] = xi - eps * xi_t
                X0[jj] = xj - eps * xj_t
                if s % 10 == 0:
                    nz = rng.standard_normal(size=(2, d)) * 0.004
                    X0[ii] += nz[0]; X0[jj] += nz[1]
                X0[ii] /= np.linalg.norm(X0[ii])
                X0[jj] /= np.linalg.norm(X0[jj])
                if _time.time() > HARD_END - 30:
                    break
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # kill-and-refill 1-3 worst with optimized insertion
            cnt, G = contacts_count(X0, best - 1e-9)
            order = np.argsort(-cnt)
            q = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
            # randomize among top-8
            pool = order[:8].copy()
            rng.shuffle(pool)
            victims = np.sort(pool[:q])
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            R = X0[keep_mask]
            new_pts = []
            for _ in range(q):
                cand = rng.standard_normal(size=(3000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                scores = (cand @ R.T).max(axis=1)
                r0 = cand[np.argmin(scores)]
                r0 = _refine_insertion(R, r0, beta=250.0, iters=50)
                new_pts.append(r0)
                R = np.vstack([R, r0[None, :]])
            for vi, rp in zip(victims, new_pts):
                X0[vi] = rp
            noise = rng.standard_normal(size=X0.shape) * 0.012
            proj = (noise * X0).sum(axis=1, keepdims=True)
            noise = noise - proj * X0
            X0 = X0 + noise
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

        if trial % 3 == 0:
            betas = [60.0, 200.0, 600.0]
        elif trial % 3 == 1:
            betas = [40.0, 120.0, 350.0]
        else:
            betas = [80.0, 250.0, 800.0]
        iters = 100 if trial % 2 == 0 else 150
        # time guard: skip heavy opt if little time
        Xopt = _smooth_opt(X0, betas, iters=iters)
        v = _max_cos(Xopt)
        if v < best - 1e-12:
            Xopt2 = _lp_polish(Xopt, seconds=2.0, t_end_limit=HARD_END - 14)
            v2 = _max_cos(Xopt2)
            if v2 < best:
                Xbest, best = Xopt2, v2
            else:
                Xbest, best = Xopt, v
        elif v < best + 0.002:
            if trial % 2 == 0 or v < best + 0.0004:
                Xopt2 = _lp_polish(Xopt, seconds=1.2, t_end_limit=HARD_END - 14)
                v2 = _max_cos(Xopt2)
                if v2 < best:
                    Xbest, best = Xopt2, v2
        if _time.time() > HARD_END - 28:
            break

    # alternating final squeeze
    remain = HARD_END - _time.time() - 1.5
    if remain > 4:
        # high-beta LSE squeeze then LP
        try:
            Xbest = _smooth_opt(Xbest, [300.0, 1000.0, 2500.0], iters=120)
        except Exception:
            pass
        remain2 = HARD_END - _time.time() - 1.0
        if remain2 > 3:
            Xbest = _lp_polish(Xbest, seconds=min(remain2 - 0.5, 25.0))
            # one more light LSE+LP if time
            if HARD_END - _time.time() > 6:
                try:
                    Xbest = _smooth_opt(Xbest, [500.0, 1500.0], iters=80)
                except Exception:
                    pass
                Xbest = _lp_polish(Xbest, seconds=max(1.0, HARD_END - _time.time() - 1.0))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest

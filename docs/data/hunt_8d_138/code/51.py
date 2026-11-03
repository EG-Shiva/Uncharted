import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lbfgs_lse(X0, beta, maxiter):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Y0 = _norm(X0.copy())
    # precompute for closure
    def fun(Yf):
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu0]
        mv = float(c.max())
        E = np.exp(beta * (c - mv))
        S = float(E.sum()) + 1e-300
        f = mv + np.log(S) / beta
        W = np.zeros((n, n))
        W[iu0] = E
        W = W + W.T
        GX = (W @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        GY = Gt / nr
        return float(f), GY.ravel()
    try:
        res = minimize(fun, Y0.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 12})
        Y = res.x.reshape(n, d)
        return _norm(Y)
    except Exception:
        return Y0

def _smooth_schedule(X, sched):
    Y = X
    for beta, it in sched:
        Y = _lbfgs_lse(Y, beta, it)
    return Y

def _lp_polish(X, deadline, radius0=6e-3, active=8e-4, cap=2800):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = radius0
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - active
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.6, 0.03)
            # tighten active as we converge
            active = max(active * 0.7, 1e-4)
        else:
            radius *= 0.5
    return X

def solve():
    import time
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)

    # Phase A: squeeze incumbent with high-beta smoothing + LP
    try:
        if time.time() < deadline - 40:
            Ab = _smooth_schedule(best, [(200.0, 120), (500.0, 150), (1000.0, 120)])
            Ab = _norm(Ab)
            va = _max_cos(Ab)
            # short LP
            Ab2 = _lp_polish(Ab, min(deadline - 30, time.time() + 8.0))
            va2 = _max_cos(Ab2)
            if va2 < best_val:
                best, best_val = Ab2, va2
            elif va < best_val:
                best, best_val = Ab, va
    except Exception:
        pass

    # Phase B: basin hopping targeted at overloaded + rattlers
    it = 0
    # predefine amplitude schedule
    amps = [0.02, 0.05, 0.10, 0.18, 0.07, 0.03, 0.12, 0.25, 0.04, 0.08,
            0.15, 0.06, 0.02, 0.20, 0.09, 0.035, 0.13, 0.05, 0.11, 0.07]
    while time.time() < deadline - 22.0:
        it += 1
        amp = amps[(it - 1) % len(amps)]
        try:
            ppm, G = _per_point_max(best)
        except Exception:
            ppm = np.zeros(n)
        # worst indices (bottleneck)
        order = np.argsort(-ppm)
        # rattlers: points whose max is well below global max
        mglob = float(ppm.max())
        ratt = np.where(ppm < mglob - 5e-4)[0]
        # choose subset to kick
        k_worst = 8 + (it % 7)  # 8..14
        k_rand = 4 + (it % 5)
        sel = set(order[:k_worst].tolist())
        if len(ratt) > 0:
            # add a few rattlers
            nr = min(len(ratt), 4 + (it % 4))
            idxr = rng.choice(ratt, size=nr, replace=False)
            for q in idxr:
                sel.add(int(q))
        # add random
        for _ in range(k_rand):
            sel.add(int(rng.integers(0, n)))
        sel = np.array(sorted(sel), dtype=int)
        # occasionally do global shake
        do_global = (it % 9 == 0)
        Xc = best.copy()
        if do_global:
            Nz = rng.standard_normal(size=(n, d))
            proj = np.sum(Nz * Xc, axis=1, keepdims=True)
            Nz = Nz - proj * Xc
            nn = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-30
            Nz = Nz / nn
            Xc = Xc + (amp * 0.5) * Nz
            Xc = _norm(Xc)
        else:
            Nz = rng.standard_normal(size=(len(sel), d))
            Xs = Xc[sel]
            proj = np.sum(Nz * Xs, axis=1, keepdims=True)
            Nz = Nz - proj * Xs
            nn = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-30
            Nz = Nz / nn
            Xs2 = Xs + amp * Nz
            Xs2 = _norm(Xs2)
            Xc[sel] = Xs2

        # quick anneal with moderate beta to spread load
        try:
            # two-stage filter: cheap polish first
            Xq = _smooth_schedule(Xc, [(70.0, 60), (160.0, 80)])
            vq = _max_cos(Xq)
        except Exception:
            continue
        # only invest LP if promising (within tolerance)
        if vq > best_val + 0.004 and (it % 3 != 0):
            # sometimes still try random fresh start instead
            if (it % 7 == 0) and (time.time() < deadline - 60):
                # fresh random diverse basin attempt
                Xr = _norm(rng.standard_normal(size=(n, d)))
                try:
                    Xr = _smooth_schedule(Xr, [(50.0, 100), (130.0, 100), (300.0, 120)])
                    vr = _max_cos(Xr)
                    if vr < best_val + 0.002:
                        Xr2 = _lp_polish(Xr, min(deadline - 22, time.time() + 3.0))
                        vr2 = _max_cos(Xr2)
                        if vr2 < best_val:
                            best, best_val = Xr2, vr2
                except Exception:
                    pass
            continue
        try:
            Xq2 = _smooth_schedule(Xq, [(350.0, 100), (700.0, 80)])
            vq2 = _max_cos(Xq2)
            if vq2 < best_val + 0.001:
                Xp = _lp_polish(Xq2, min(deadline - 22, time.time() + 3.5))
                vp = _max_cos(Xp)
                if vp < best_val - 1e-12:
                    best, best_val = Xp, vp
        except Exception:
            continue

    # Phase C: final squeeze
    try:
        if time.time() < deadline - 0.5:
            best = _smooth_schedule(best, [(600.0, 150), (1200.0, 150)])
            best = _lp_polish(best, deadline - 0.3, radius0=4e-3, active=5e-4)
    except Exception:
        pass
    best = _norm(best)
    return best

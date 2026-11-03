import time
import numpy as np

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lbfgs_smooth(x_init, betas, maxiters):
    from scipy.optimize import minimize
    n, d = x_init.shape
    Y = x_init.copy()
    # keep rows unit to avoid radial drift
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    for beta, mi in zip(betas, maxiters):
        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            # upper triangle
            iu = np.triu_indices(n, 1)
            c = G[iu]
            mv = c.max()
            # weights
            E = np.exp(beta * (c - mv))
            S = E.sum() + 1e-300
            f = mv + np.log(S) / beta
            # symmetric weight matrix
            W = np.zeros((n, n))
            W[iu] = E
            W = W + W.T
            GX = (W @ X) / S
            # grad wrt X -> grad wrt Y (tangential / norm)
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
            return float(f), GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-6, 'maxfun': int(mi*20), 'maxcor': 20})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        nr = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)) + 1e-30
        Y = Y / nr
    X = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return X

def refine_void(v0, Xrem, beta=120.0, steps=200, lr=0.05):
    v = v0.copy()
    v = v / (np.linalg.norm(v) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ Xrem) / S
        gt = g - np.dot(g, v) * v
        # decay lr
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        v = v / (np.linalg.norm(v) + 1e-30)
    return v

def best_void_replace(cur, rng):
    n, d = cur.shape
    G = cur @ cur.T
    np.fill_diagonal(G, -2.0)
    per = G.max(axis=1)
    # pick victim among worst few randomly
    order = np.argsort(-per)
    pool = order[:10]
    victim = int(pool[rng.randint(len(pool))])
    Xrem = np.delete(cur, victim, axis=0)
    Ncand = 20000
    cand = rng.randn(Ncand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    smax = (cand @ Xrem.T).max(axis=1)
    bi = int(np.argmin(smax))
    v0 = cand[bi]
    try:
        v0 = refine_void(v0, Xrem, beta=120.0, steps=150, lr=0.08)
    except Exception:
        pass
    prop = cur.copy()
    prop[victim] = v0
    return prop

def lp_polish(x, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    import time
    import numpy as np
    t0 = time.time()
    rng = np.random.RandomState(20261102)
    n, d = 138, 8
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n)
        d0 = min(xb.shape[1], d)
        tmp[:m0, :d0] = xb[:m0, :d0]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # also compare record.npy if different
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            if max_cos(xr) < max_cos(xb):
                xb = xr.copy()
    except Exception:
        pass
    best_val = max_cos(xb)
    x_best = xb.copy()
    cur = xb.copy()
    cur_val = best_val

    # quick jam of starting point with short LBFGS to get baseline jammed value
    try:
        if time.time() - t0 < 150:
            cand = lbfgs_smooth(cur.copy(), [80.0, 250.0], [120, 120])
            v = max_cos(cand)
            if v < best_val:
                best_val = v
                x_best = cand.copy()
                cur = cand.copy()
                cur_val = v
    except Exception:
        pass

    stall = 0
    it = 0
    EXPL = 142.0
    while time.time() - t0 < EXPL:
        it += 1
        mode = it % 4
        try:
            if mode == 0:
                sigma = 0.03 if (it % 8 != 0) else 0.12
                prop = cur + sigma * rng.randn(n, d)
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                polished = lbfgs_smooth(prop, [40.0, 130.0], [100, 150])
            elif mode == 1:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                cmax = G.max()
                cnt = (G > cmax - 0.01).sum(axis=1)
                order = np.argsort(-cnt)
                pool = order[:10]
                k = 4 if (it % 8 == 0) else 2
                chosen = pool[rng.choice(len(pool), size=k, replace=False)]
                prop = cur.copy()
                prop[chosen] += 0.22 * rng.randn(k, d)
                prop += 0.008 * rng.randn(n, d)
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                polished = lbfgs_smooth(prop, [40.0, 150.0], [100, 150])
            elif mode == 2:
                prop = best_void_replace(cur, rng)
                prop += 0.004 * rng.randn(n, d)
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                polished = lbfgs_smooth(prop, [40.0, 130.0], [100, 150])
            else:
                # big global jump from best
                sigma = 0.06 if (it % 16 != 0) else 0.20
                prop = x_best + sigma * rng.randn(n, d)
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                polished = lbfgs_smooth(prop, [30.0, 120.0], [80, 130])
        except Exception:
            stall += 1
            continue
        try:
            val = max_cos(polished)
        except Exception:
            stall += 1
            continue
        if val < cur_val - 1e-12:
            cur = polished
            cur_val = val
            stall = 0
            if val < best_val - 1e-12:
                best_val = val
                x_best = polished.copy()
        else:
            stall += 1
            # sideways move rarely to escape
            if rng.rand() < 0.03 and val < cur_val + 0.008:
                cur = polished
                cur_val = val
            if stall >= 12:
                try:
                    tmp = x_best + 0.10 * rng.randn(n, d)
                    tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
                    tmp = lbfgs_smooth(tmp, [40.0, 120.0], [80, 100])
                    cur_val = max_cos(tmp)
                    cur = tmp
                    if cur_val < best_val - 1e-12:
                        best_val = cur_val
                        x_best = cur.copy()
                except Exception:
                    cur = x_best.copy()
                    cur_val = best_val
                stall = 0
        if time.time() - t0 >= EXPL:
            break

    # intensive jam of best
    try:
        tleft = 174.0 - (time.time() - t0)
        if tleft > 6:
            mi1 = 200 if tleft > 30 else 100
            mi2 = 200 if tleft > 30 else 100
            x_best = lbfgs_smooth(x_best.copy(), [150.0, 600.0], [mi1, mi2])
    except Exception:
        pass
    try:
        tleft2 = 178.0 - (time.time() - t0)
        if tleft2 > 4:
            x_best = lp_polish(x_best, min(tleft2 - 1.0, 22.0))
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return x_best

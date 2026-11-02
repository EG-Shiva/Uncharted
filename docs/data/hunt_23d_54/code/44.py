import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _anneal(X, steps, k0, k1, lr0, lr1, mu=0.85, seed_noise=0.0, rng=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights exp(k*(G-m)) in [0,1]
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        rs = np.sum(W, axis=1, keepdims=True) + 1.0
        Ft /= rs
        V = mu * V + Ft
        # normalize step scale by rms
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    if seed_noise > 0 and rng is not None:
        pass
    return X

def _void_vector(X, keep_mask, rng, n_try=2000):
    kept = X[keep_mask]
    C = rng.randn(n_try, X.shape[1])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[np.argmin(m)]

def _lp_polish_short(X, seconds, t_global=None, global_budget=175.0):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_global is not None and (time.time() - t_global) > global_budget:
            break
        if radius < 1e-14:
            break
        Gf = X @ X.T
        ca = Gf[iu]
        m = ca.max()
        sel = ca > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(ca[idx])[::-1][:900]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
        I, J = iu[0][sel], iu[1][sel]
        c = ca[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    it = 0
    # hopping loop, reserve ~10s for final LP
    while time.time() - t0 < BUDGET - 10.0:
        it += 1
        remain = BUDGET - 10.0 - (time.time() - t0)
        if remain < 2.0:
            break
        steps = 700
        if remain < 15:
            steps = 400
        mode = it % 9
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        cnt = np.sum(G > (best - 0.015), axis=1)
        ppm = np.max(G, axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-ppm)

        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.010
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=60, k1=2500, lr0=0.03, lr1=0.002, mu=0.85)
        elif mode == 1:
            x += rng.randn(n, d) * 0.030
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=40, k1=3000, lr0=0.04, lr1=0.002, mu=0.88)
        elif mode == 2:
            x += rng.randn(n, d) * 0.070
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=30, k1=2500, lr0=0.05, lr1=0.003, mu=0.88)
        elif mode == 3:
            ksel = 3 + (it % 3)
            idx = order_worst[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _void_vector(x, mask, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.02
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.005
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=50, k1=3500, lr0=0.035, lr1=0.0015, mu=0.86)
        elif mode == 4:
            ksel = 4 + (it % 3)
            idx = order_lo[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _void_vector(x, mask, rng, n_try=1500)
                x[j] = nv
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=50, k1=3000, lr0=0.035, lr1=0.002, mu=0.86)
        elif mode == 5:
            # random subset reinit (topology break)
            ksel = 5
            idx = rng.choice(n, size=ksel, replace=False)
            x[idx] = rng.randn(ksel, d)
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            x += rng.randn(n, d) * 0.005
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=25, k1=2000, lr0=0.05, lr1=0.003, mu=0.88)
        elif mode == 6:
            # collective shake of jammed cluster
            ksel = 6
            idx = order_worst[-ksel:]
            x[idx] += rng.randn(ksel, d) * 0.12
            x += rng.randn(n, d) * 0.006
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=40, k1=2800, lr0=0.04, lr1=0.002, mu=0.87)
        elif mode == 7:
            # single worst void + medium noise
            j = int(order_worst[0])
            mask = np.ones(n, dtype=bool); mask[j] = False
            nv = _void_vector(x, mask, rng, n_try=2500)
            x[j] = nv
            x += rng.randn(n, d) * 0.012
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=70, k1=4000, lr0=0.030, lr1=0.0012, mu=0.85)
        else:
            # pure random restart annealed (diversity)
            x = rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # mix with best to keep some structure
            a = 0.5
            x = a * x + (1 - a) * x_best
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=20, k1=1800, lr0=0.06, lr1=0.003, mu=0.88)

        v = _max_cos(x)
        if v < best:
            best = v
            x_best = x.copy()
            # quick high-k refinement of the new basin
            try:
                xr2 = _anneal(x_best.copy(), steps=300, k0=800, k1=6000, lr0=0.008, lr1=0.0008, mu=0.85)
                v2 = _max_cos(xr2)
                if v2 < best:
                    best, x_best = v2, xr2.copy()
            except Exception:
                pass
        # timeout guard
        if time.time() - t0 > BUDGET - 10.0:
            break

    # final refinement of best with high-k anneal then short LP
    try:
        if time.time() - t0 < BUDGET - 9.0:
            x_best = _anneal(x_best.copy(), steps=500, k0=300, k1=6000, lr0=0.010, lr1=0.0006, mu=0.86)
            best = _max_cos(x_best)
    except Exception:
        pass
    try:
        rem = BUDGET - (time.time() - t0) - 0.5
        if rem > 2.0:
            x_best = _lp_polish_short(x_best, seconds=min(rem, 8.0), t_global=t0, global_budget=BUDGET)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

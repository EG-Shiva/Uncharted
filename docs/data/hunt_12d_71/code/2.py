import time
import numpy as np

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def normed(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def smooth_descent(X_init, steps, T0, T1, lr0, lr1, mu=0.85, check_every=128, t_end=1e12):
    X = X_init.copy()
    V = np.zeros_like(X)
    cur_best_val = max_cos(X)
    bestX = X.copy()
    bestV = cur_best_val
    for s in range(steps):
        if time.time() > t_end:
            break
        frac = s / max(1, steps - 1)
        # geometric schedules
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights, stable (max subtracted)
        # 71x71 exp is cheap
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # momentum
        V = mu * V + Gt
        maxn = np.linalg.norm(V, axis=1).max()
        if maxn > 1e-15 and np.isfinite(maxn):
            step = V / maxn * lr
        else:
            step = V * 0.0
        X = X - step
        X = normed(X)
        if (s % check_every == 0) or (s == steps - 1):
            v = max_cos(X)
            if v < bestV:
                bestV = v
                bestX = X.copy()
    return bestX, bestV

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0  # leave margin for 180s
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(71, 12))
    xb = normed(np.asarray(xb, dtype=np.float64))
    if xb.shape != (71, 12):
        # fix shape if needed by random
        tmp = rng.normal(size=(71, 12))
        xb = normed(tmp)
    try:
        xr = normed(np.asarray(np.load("record.npy"), dtype=np.float64))
        if xr.shape == (71, 12):
            if max_cos(xr) < max_cos(xb):
                xb = xr
    except Exception:
        pass
    best = xb.copy()
    bestV = max_cos(best)
    n, d = 71, 12

    # precompute contact counts for targeting
    def worst_indices(X, v):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # near-max band
        thr = v - 0.005
        cnt = np.sum(G > thr, axis=1)
        order = np.argsort(-cnt)
        return order, cnt

    trial = 0
    # fast initial squeeze with tiny T to ensure at optimum
    # (cheap, 500 steps)
    if time.time() - t0 < TIME_LIMIT - 10:
        Xp, Vp = smooth_descent(best, steps=600, T0=0.008, T1=0.0015, lr0=0.008, lr1=0.0008,
                                mu=0.85, check_every=100, t_end=t0 + TIME_LIMIT)
        if Vp < bestV:
            best, bestV = Xp, Vp

    while time.time() - t0 < TIME_LIMIT - 2.0:
        trial += 1
        remaining = t0 + TIME_LIMIT - time.time()
        if remaining < 3.0:
            break
        # adapt steps to remaining time
        if remaining < 20:
            steps = 800
        elif remaining < 60:
            steps = 1500
        else:
            steps = 2000
        r = rng.random()
        X0 = None
        # choose perturbation type
        if r < 0.35:
            # global jitter, log-uniform sigma
            sigma = 10.0 ** rng.uniform(-3.0, -1.1)
            X0 = best + sigma * rng.normal(size=(n, d))
            X0 = normed(X0)
            T0_, T1_ = 0.012, 0.0012
            lr0_, lr1_ = 0.015, 0.0006
        elif r < 0.60:
            # single point full reinsertion
            idx = int(rng.integers(0, n))
            # occasionally pick worst
            if rng.random() < 0.5:
                order, _ = worst_indices(best, bestV)
                idx = int(order[0])
            X0 = best.copy()
            X0[idx] = rng.normal(size=(d,))
            # small jitter others to break symmetry
            X0 += 0.002 * rng.normal(size=(n, d))
            X0 = normed(X0)
            T0_, T1_ = 0.015, 0.0010
            lr0_, lr1_ = 0.020, 0.0005
        elif r < 0.85:
            # targeted kick of overloaded hub
            order, _ = worst_indices(best, bestV)
            k = int(rng.integers(0, 5))  # top-5 worst
            idx = int(order[k])
            sigma = 10.0 ** rng.uniform(-2.2, -0.9)
            X0 = best.copy()
            X0[idx] = X0[idx] + sigma * rng.normal(size=(d,))
            # also jitter its contact neighbors a bit
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            neigh = np.where(G[idx] > bestV - 0.01)[0]
            for j in neigh:
                if rng.random() < 0.4:
                    X0[j] = X0[j] + 0.005 * rng.normal(size=(d,))
            X0 = normed(X0)
            T0_, T1_ = 0.010, 0.0008
            lr0_, lr1_ = 0.012, 0.0004
        else:
            # two-point swap / block reinit
            idx1 = int(rng.integers(0, n))
            idx2 = int(rng.integers(0, n))
            X0 = best.copy()
            X0[idx1] = rng.normal(size=(d,))
            X0[idx2] = rng.normal(size=(d,))
            X0 += 0.003 * rng.normal(size=(n, d))
            X0 = normed(X0)
            T0_, T1_ = 0.018, 0.0015
            lr0_, lr1_ = 0.022, 0.0007
        mu = 0.85 if rng.random() < 0.7 else 0.92
        Xp, Vp = smooth_descent(X0, steps=steps, T0=T0_, T1=T1_, lr0=lr0_, lr1=lr1_,
                                mu=mu, check_every=200, t_end=t0 + TIME_LIMIT - 0.5)
        # extra micro-squeeze if promising (within 0.002 of best)
        if Vp < bestV + 0.002 and time.time() - t0 < TIME_LIMIT - 3:
            Xp2, Vp2 = smooth_descent(Xp, steps=400, T0=0.0012, T1=0.0004,
                                      lr0=0.003, lr1=0.0003, mu=0.9,
                                      check_every=100, t_end=t0 + TIME_LIMIT - 0.5)
            if Vp2 < Vp:
                Xp, Vp = Xp2, Vp2
        if Vp < bestV - 1e-12:
            best, bestV = Xp, Vp
        # opportunistic: if we improved a lot, intensify around it immediately
        # (loop will naturally do it since best updated)
    return normed(best)

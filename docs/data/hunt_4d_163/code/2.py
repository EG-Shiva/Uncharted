import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 165.0
    rng = np.random.RandomState(0)

    def normalize(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # load start
    best = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape == (163, 4):
                v = normalize(v)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = normalize(rng.randn(163, 4))
        best_val = max_cos(best)
    best = normalize(best)

    def polish(X_init, steps, tau0, tau1, lr0, lr1):
        X = X_init.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        local_best_val = max_cos(X)
        local_best = X.copy()
        # precompute decay factors
        for t in range(1, steps + 1):
            if time.time() - t0 > BUDGET:
                break
            frac = t / max(1, steps)
            # exponential schedule
            tau = tau0 * (tau1 / tau0) ** frac
            lr = lr0 * (lr1 / lr0) ** frac
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cur_max = float(np.max(G))
            # smooth-max weights
            # clip for overflow
            S = (G - cur_max) / tau
            # avoid underflow: threshold
            # exp only where > -12
            W = np.exp(np.maximum(S, -12.0))
            np.fill_diagonal(W, 0.0)
            Grad = W @ X
            proj = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - proj * X
            m1 = b1 * m1 + (1.0 - b1) * Gt
            v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
            mhat = m1 / (1.0 - b1 ** t)
            vhat = v1 / (1.0 - b2 ** t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            X = X - step
            X = normalize(X)
            # track occasionally to save time
            if (t & 31) == 0 or t == steps:
                c = max_cos(X)
                if c < local_best_val:
                    local_best_val = c
                    local_best = X.copy()
        return local_best, local_best_val

    def contact_degrees(X, m, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G > (m - thr)
        return np.sum(C, axis=1), G

    def find_hole(X, exclude_idx=None, n_cand=20000):
        # random candidates on S3
        R = rng.randn(n_cand, 4)
        R = normalize(R)
        if exclude_idx is not None:
            mask = np.ones(X.shape[0], dtype=bool)
            mask[exclude_idx] = False
            Y = X[mask]
        else:
            Y = X
        D = R @ Y.T  # (cand, n)
        vals = np.max(D, axis=1)
        i = int(np.argmin(vals))
        return R[i].copy(), float(vals[i])

    def refine_hole(X, h, steps=200, lr=0.05):
        # minimize max dot to X by moving h on sphere away from neighbors
        h = h.copy()
        for _ in range(steps):
            d = X @ h  # (n,)
            m = float(np.max(d))
            # weights to closest points
            tau = 0.01
            w = np.exp((d - m) / tau)
            # gradient wrt h: sum w_i X_i ; we want to go opposite
            g = w @ X
            # tangent
            g = g - np.dot(g, h) * h
            h = h - lr * g
            nh = np.linalg.norm(h)
            if nh == 0:
                break
            h = h / nh
        return h

    # initial fine polish of record
    try:
        b1x, b1v = polish(best, steps=2500, tau0=0.008, tau1=0.002, lr0=0.01, lr1=0.002)
        if b1v < best_val:
            best, best_val = b1x, b1v
    except Exception:
        pass

    it = 0
    # main basin hopping
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        # choose mode
        mode = it % 6
        Xtry = best.copy()
        if mode in (0, 1):
            # global / subset shake
            if rng.rand() < 0.35:
                sigma = [0.01, 0.03, 0.08, 0.15, 0.005][rng.randint(5)]
                Xtry = Xtry + sigma * rng.randn(*Xtry.shape)
                Xtry = normalize(Xtry)
            else:
                k = int(rng.randint(3, 21))
                idx = rng.choice(163, size=k, replace=False)
                # larger sigma for subset
                sigma = [0.05, 0.12, 0.25, 0.03][rng.randint(4)]
                Xtry[idx] = Xtry[idx] + sigma * rng.randn(k, 4)
                Xtry = normalize(Xtry)
        elif mode in (2, 3):
            # relocate worst point to hole
            try:
                m = max_cos(Xtry)
                deg, G = contact_degrees(Xtry, m, thr=2e-7)
                # victim: max degree, tie by max row value
                rowmax = np.max(G, axis=1)
                order = np.lexsort((-rowmax, -deg))
                victim = int(order[0])
                hole, _ = find_hole(Xtry, exclude_idx=[victim], n_cand=12000)
                hole = refine_hole(Xtry[np.arange(163) != victim], hole, steps=120, lr=0.05)
                Xtry[victim] = hole
                # small shake of neighbors to unlock
                if rng.rand() < 0.7:
                    Gv = Xtry @ Xtry[victim]
                    nbrs = np.argsort(-Gv)[1:6]
                    Xtry[nbrs] += 0.02 * rng.randn(len(nbrs), 4)
                    Xtry = normalize(Xtry)
            except Exception:
                Xtry = Xtry + 0.02 * rng.randn(*Xtry.shape)
                Xtry = normalize(Xtry)
        elif mode == 4:
            # relocate rattler / low-degree to hole (hole filling)
            try:
                m = max_cos(Xtry)
                deg, _ = contact_degrees(Xtry, m, thr=1e-7)
                lows = np.where(deg <= 1)[0]
                if len(lows) == 0:
                    lows = np.argsort(deg)[:2]
                victim = int(lows[rng.randint(len(lows))])
                hole, _ = find_hole(Xtry, exclude_idx=[victim], n_cand=12000)
                hole = refine_hole(Xtry[np.arange(163) != victim], hole, steps=150, lr=0.05)
                Xtry[victim] = hole
            except Exception:
                Xtry = Xtry + 0.02 * rng.randn(*Xtry.shape)
                Xtry = normalize(Xtry)
        else:
            # strong shake of jammed cluster around worst pair
            try:
                G = Xtry @ Xtry.T
                np.fill_diagonal(G, -2.0)
                ij = int(np.argmax(G))
                a, b = divmod(ij, 163)
                # neighbors of a,b
                da = G[a]
                idx = np.argsort(-da)[:10]
                Xtry[idx] += 0.06 * rng.randn(len(idx), 4)
                Xtry = normalize(Xtry)
            except Exception:
                pass

        # choose polish schedule depending on mode
        if mode in (0, 2, 4):
            steps, tau0_, tau1_, lr0_, lr1_ = 1200, 0.012, 0.0025, 0.02, 0.003
        else:
            steps, tau0_, tau1_, lr0_, lr1_ = 900, 0.015, 0.003, 0.025, 0.004
        # reduce steps if time low
        remain = BUDGET - 12.0 - (time.time() - t0)
        if remain < 20:
            steps = min(steps, 500)
        try:
            Xpol, vpol = polish(Xtry, steps=steps, tau0=tau0_, tau1=tau1_, lr0=lr0_, lr1=lr1_)
        except Exception:
            continue
        if vpol < best_val - 1e-15:
            best, best_val = Xpol, vpol
            # intensify around new best with extra fine polish if time allows
            if time.time() - t0 < BUDGET - 25.0:
                try:
                    Xf, vf = polish(best, steps=800, tau0=0.005, tau1=0.0012, lr0=0.008, lr1=0.0015)
                    if vf < best_val:
                        best, best_val = Xf, vf
                except Exception:
                    pass

    # final fine polish
    try:
        Xf, vf = polish(best, steps=2000, tau0=0.004, tau1=0.001, lr0=0.006, lr1=0.001)
        if vf < best_val:
            best, best_val = Xf, vf
    except Exception:
        pass

    best = normalize(best)
    return best

import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = t0 + 132.0
    n, d = 82, 14
    rng = np.random.default_rng(0)

    # load start
    X0 = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _max_cos(a)
                if v < bv:
                    bv = v
                    X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _norm(rng.standard_normal((n, d)))
        bv = _max_cos(X0)

    Xbest = X0.copy()
    best = float(bv)

    def quench(X, steps, beta0, beta1, lr0, mu=0.85, seed_off=0):
        X = _norm(X.copy())
        vel = np.zeros_like(X)
        # precompute beta schedule log-linear, lr cosine
        lb0 = float(np.log(beta0)); lb1 = float(np.log(beta1))
        for it in range(steps):
            if (it & 63) == 0 and time.time() > DEADLINE - 3.0:
                break
            frac = it / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
            lr = float(lr0 * (0.5 * (1.0 + np.cos(np.pi * frac))) + 1e-4))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # weights
            S = beta * (G - m)
            np.clip(S, -40.0, 0.0, out=S)
            E = np.exp(S)
            np.fill_diagonal(E, 0.0)
            s = float(np.sum(E)) * 0.5
            if not np.isfinite(s) or s <= 0:
                continue
            E /= s  # sum over unordered =1, symmetric
            Gr = E @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            # clip large grad
            gn = float(np.linalg.norm(Gt))
            if not np.isfinite(gn) or gn == 0:
                continue
            # normalize step scale: keep per-point grad O(1)
            vel = mu * vel + Gt
            X = X - lr * vel
            # renorm
            rn = np.linalg.norm(X, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            X = X / rn
            if not np.all(np.isfinite(X)):
                X = _norm(Xbest.copy())
                vel *= 0.0
        return _norm(X)

    def far_hole(X_keep, n_holes=1, n_sample=20000):
        # find directions far from X_keep (k x d)
        R = rng.standard_normal((n_sample, d))
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        if len(X_keep) == 0:
            return R[:n_holes]
        D = R @ X_keep.T  # sample x keep
        worst = np.max(D, axis=1)  # max cosine to set (lower better)
        idx = np.argpartition(worst, n_holes - 1)[:n_holes] if n_holes < n_sample else np.arange(n_sample)
        # among them pick smallest worst
        order = np.argsort(worst[idx])
        return R[idx[order]]

    def worst_points(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = np.max(G, axis=1)
        # also consider mean of top few
        idx = np.argpartition(-mx, k - 1)[:k]
        return idx[np.argsort(-mx[idx])]

    # quick initial squeeze of incumbent
    if time.time() < DEADLINE - 5.0:
        Y = quench(Xbest, steps=1200, beta0=80.0, beta1=1500.0, lr0=0.03)
        v = _max_cos(Y)
        if v < best:
            best = v
            Xbest = Y.copy()

    trial = 0
    # diverse schedule
    while time.time() < DEADLINE - 5.0:
        trial += 1
        mode = trial % 6
        Xc = Xbest.copy()
        if mode == 0:
            # 1-hole replacement of worst point
            k = 1
            wp = worst_points(Xc, 3)
            # keep all except worst[0]
            rem = np.delete(Xc, wp[0], axis=0)
            h = far_hole(rem, n_holes=1, n_sample=15000)
            Xc[wp[0]] = h[0]
            steps, b0, b1, lr = 700, 80.0, 1200.0, 0.035
        elif mode == 1:
            k = 2
            wp = worst_points(Xc, 6)
            sel = wp[:2]
            rem_idx = [i for i in range(n) if i not in set(sel.tolist())]
            rem = Xc[rem_idx]
            h = far_hole(rem, n_holes=2, n_sample=15000)
            Xc[sel[0]] = h[0]; Xc[sel[1]] = h[1]
            steps, b0, b1, lr = 800, 60.0, 1000.0, 0.04
        elif mode == 2:
            # medium global kick
            sig = 0.07
            Xc = Xc + sig * rng.standard_normal((n, d))
            Xc = _norm(Xc)
            steps, b0, b1, lr = 700, 80.0, 1500.0, 0.03
        elif mode == 3:
            # large global kick
            sig = 0.16
            Xc = Xc + sig * rng.standard_normal((n, d))
            Xc = _norm(Xc)
            steps, b0, b1, lr = 900, 50.0, 900.0, 0.045
        elif mode == 4:
            # subset large kick on 12 random points
            sel = rng.choice(n, size=12, replace=False)
            Xc[sel] = Xc[sel] + 0.25 * rng.standard_normal((12, d))
            Xc = _norm(Xc)
            steps, b0, b1, lr = 800, 60.0, 1100.0, 0.04
        else:
            # small jitter + 1 hole
            sig = 0.02
            Xc = Xc + sig * rng.standard_normal((n, d))
            Xc = _norm(Xc)
            wp = worst_points(Xc, 4)
            rem = np.delete(Xc, wp[0], axis=0)
            h = far_hole(rem, n_holes=4, n_sample=8000)
            Xc[wp[0]] = h[0]
            steps, b0, b1, lr = 700, 90.0, 1600.0, 0.028
        # vary lr slightly deterministically
        lr = lr * (0.85 + 0.3 * ((trial * 37 % 100) / 100.0))
        Y = quench(Xc, steps=steps, beta0=b0, beta1=b1, lr0=lr)
        v = _max_cos(Y)
        if v < best - 1e-12:
            best = v
            Xbest = Y.copy()
            # intensify around new best immediately
            if time.time() < DEADLINE - 8.0:
                Y2 = quench(Xbest, steps=900, beta0=150.0, beta1=2500.0, lr0=0.015)
                v2 = _max_cos(Y2)
                if v2 < best:
                    best = v2
                    Xbest = Y2.copy()
        # early exit if excellent
        if best < 0.190:
            break

    # final greedy per-point hill climb + short polish if time left
    if time.time() < DEADLINE - 4.0:
        # short high-beta polish
        Y = quench(Xbest, steps=1000, beta0=200.0, beta1=3000.0, lr0=0.012)
        v = _max_cos(Y)
        if v < best:
            best = v
            Xbest = Y.copy()
    if time.time() < DEADLINE - 2.0:
        # coordinate hill climbing: try random tangent moves per point
        X = Xbest.copy()
        cur = best
        # order points by individual max descending
        for _ in range(300):
            if time.time() > DEADLINE - 1.0:
                break
            i = int(rng.integers(0, n))
            # propose
            g = rng.standard_normal(d)
            # make tangent
            g = g - np.dot(g, X[i]) * X[i]
            gn = np.linalg.norm(g)
            if gn == 0:
                continue
            g /= gn
            step = 0.002 * (0.5 + rng.random())
            cand = X.copy()
            cand[i] = cand[i] + step * g
            cand[i] /= np.linalg.norm(cand[i])
            v = _max_cos(cand)
            if v < cur:
                cur = v
                X = cand
                if v < best:
                    best = v
                    Xbest = X.copy()
    Xbest = _norm(np.asarray(Xbest, dtype=float))
    return Xbest

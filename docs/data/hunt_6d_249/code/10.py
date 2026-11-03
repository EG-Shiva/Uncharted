import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 249, 6

    # load starts
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < best_v:
                best_v = v
                best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)

    iu0, iu1 = np.triu_indices(n, 1)

    def make_fun(alpha):
        def fun(yflat):
            Y = yflat.reshape(n, d)
            nrm = np.sqrt(np.sum(Y * Y, axis=1))
            nrm = np.maximum(nrm, 1e-12)
            X = Y / nrm[:, None]
            # pairwise cos on upper triangle
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(alpha * (g - m))
            S = float(np.sum(e))
            f = m + np.log(S) / alpha
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            Gx = W @ X
            s = np.sum(Gx * X, axis=1)
            Gy = (Gx - X * s[:, None]) / nrm[:, None]
            # radial anchor
            diff = nrm - 1.0
            f = f + 1e-4 * float(np.mean(diff * diff))
            Gy = Gy + (X * (2e-4 * diff / n)[:, None])
            return float(f), Gy.ravel()
        return fun

    from scipy.optimize import minimize

    def polish(Xinit, alphas, iters):
        Xf = Xinit.ravel().copy()
        for alpha, mi in zip(alphas, iters):
            if time.time() > T_END - 18:
                break
            fun = make_fun(float(alpha))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "maxfun": int(mi*3),
                                        "ftol": 1e-14, "gtol": 1e-7, "maxcor": 10})
                Xf = res.x
            except Exception:
                break
            # renormalize to curb drift
            Xm = Xf.reshape(n, d)
            Xm /= np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = Xm.ravel()
        Xout = Xf.reshape(n, d)
        Xout /= np.linalg.norm(Xout, axis=1, keepdims=True)
        return Xout

    def hole_replace(X, n_replace, n_cand=4000):
        # replace worst points by largest-hole candidates (greedy)
        X = X.copy()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        row_max = np.max(G, axis=1)
        order = np.argsort(-row_max)
        for k in range(n_replace):
            idx = int(order[k]) if k < len(order) else int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xo = X[mask]
            C = rng.standard_normal(size=(n_cand, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Xo.T
            cmax = np.max(D, axis=1)
            bi = int(np.argmin(cmax))
            X[idx] = C[bi]
            if k + 1 < n_replace:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                row_max = np.max(G, axis=1)
                order = np.argsort(-row_max)
        return X

    # initial sharp polish of incumbent
    cur = polish(best_x, [150.0, 600.0, 2000.0, 5000.0], [150, 180, 220, 260])
    cur_v = _max_cos(cur)
    if cur_v < best_v:
        best_v, best_x = cur_v, cur.copy()
    else:
        cur = best_x.copy()
        cur_v = best_v

    fast_a = [80.0, 350.0, 1200.0]
    fast_i = [100, 130, 160]
    mid_a = [150.0, 600.0, 2000.0]
    mid_i = [120, 150, 180]

    it = 0
    temp0 = 0.0015
    while time.time() < T_END - 20:
        it += 1
        frac = (time.time() - t0) / 150.0
        temp = temp0 * max(0.15, 1.0 - frac)
        mode = it % 7
        base = best_x if (it % 3 == 0) else cur
        if mode == 0:
            sig = 0.02
            Xt = base + sig * rng.standard_normal(size=(n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            al, mi = fast_a, fast_i
        elif mode == 1:
            sig = 0.07
            Xt = base + sig * rng.standard_normal(size=(n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            al, mi = mid_a, mid_i
        elif mode == 2:
            # core-targeted shake: worst 30 get big kick
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rm = np.max(G, axis=1)
            worst = np.argsort(-rm)[:30]
            Xt = base.copy()
            Xt += 0.012 * rng.standard_normal(size=(n, d))
            Xt[worst] += 0.16 * rng.standard_normal(size=(len(worst), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            al, mi = mid_a, mid_i
        elif mode == 3:
            # hole reinsert 2-3 worst
            Xt = hole_replace(base, 3 if (it % 2 == 0) else 2, n_cand=4000)
            al, mi = fast_a, fast_i
        elif mode == 4:
            # fully randomize 2-4 points
            Xt = base.copy()
            k = 2 + int(rng.integers(0, 3))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal(size=(k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xt[idx] = R
            # small shake rest
            Xt += 0.01 * rng.standard_normal(size=(n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            al, mi = mid_a, mid_i
        elif mode == 5:
            sig = 0.035
            Xt = base + sig * rng.standard_normal(size=(n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            al, mi = fast_a, fast_i
        else:
            # rattler shake: shake points with small row_max hard
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rm = np.max(G, axis=1)
            # rattlers = smallest rm
            ratt = np.argsort(rm)[:80]
            Xt = base.copy()
            Xt += 0.008 * rng.standard_normal(size=(n, d))
            Xt[ratt] += 0.10 * rng.standard_normal(size=(len(ratt), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            al, mi = fast_a, fast_i

        Y = polish(Xt, al, mi)
        v = _max_cos(Y)
        if v < best_v - 1e-12:
            best_v, best_x = v, Y.copy()
            cur, cur_v = Y.copy(), v
        elif v < cur_v:
            cur, cur_v = Y.copy(), v
        elif v < cur_v + temp and rng.random() < 0.6:
            cur, cur_v = Y.copy(), v
        # occasional pull back to best if cur drifts
        if cur_v > best_v + 0.008:
            cur, cur_v = best_x.copy(), best_v

    # final sharp jam on best
    if time.time() < T_END - 3:
        best_x = polish(best_x, [300.0, 1000.0, 3000.0, 8000.0], [220, 260, 320, 350])
        best_v = _max_cos(best_x)

    # cheap exact subgradient descent to squeeze ties
    deadline = T_END - 1.0
    cur2 = best_x.copy()
    cur2v = _max_cos(cur2)
    step = 2e-3
    while time.time() < deadline and step > 1e-13:
        G = cur2 @ cur2.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        window = max(10.0 * step, 3e-4)
        if step < 5e-5:
            window = max(1.5e-4, 10.0 * step)
        mask = G > (m - window)
        F = mask.astype(float) @ cur2
        dot = np.sum(F * cur2, axis=1)
        Ft = F - dot[:, None] * cur2
        trial = cur2 - step * Ft
        trial /= np.maximum(np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
        v = _max_cos(trial)
        if v < cur2v:
            cur2, cur2v = trial, v
            step = min(step * 1.25, 0.01)
        else:
            step *= 0.5
            if step < 1e-7:
                break
    if cur2v < best_v:
        best_x, best_v = cur2, cur2v
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

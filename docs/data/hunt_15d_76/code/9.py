import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(Xn):
    G = Xn @ Xn.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    TIME_BUDGET = 150.0  # leave margin for 180s limit
    rng = np.random.default_rng(12345)
    n, d = 76, 15

    # load starting point
    try:
        x_best = np.load("best.npy")
    except Exception:
        try:
            x_best = np.load("record.npy")
        except Exception:
            x_best = rng.normal(size=(n, d))
    x_best = np.asarray(x_best, dtype=float)
    if x_best.shape != (n, d):
        # fallback: random
        x_best = rng.normal(size=(n, d))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    # keep copy of initial in case everything fails
    init_best = x_best.copy()
    init_val = best_val

    iu = np.triu_indices(n, 1)
    M = len(iu[0])

    def smooth_opt(X0, betas, iters_list):
        X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
        x0 = X0.ravel().copy()
        for beta, maxiter in zip(betas, iters_list):
            lam = 1e-2
            def fun_grad(xf):
                X = xf.reshape(n, d)
                r = np.linalg.norm(X, axis=1, keepdims=True) + 1e-16
                Xn = X / r
                G = Xn @ Xn.T
                g = G[iu]
                gm = g.max()
                e = np.exp(beta * (g - gm))
                s = e.sum()
                L = gm + np.log(s) / beta
                w = e / s
                W = np.zeros((n, n))
                W[iu] = w
                W = W + W.T
                Grad_Xn = W @ Xn
                dot = (Grad_Xn * Xn).sum(axis=1, keepdims=True)
                Grad_X = (Grad_Xn - Xn * dot) / r
                L = L + lam * float(((r - 1.0) ** 2).sum())
                Grad_X = Grad_X + (2.0 * lam * (r - 1.0) * Xn)
                return float(L), Grad_X.ravel()
            try:
                res = minimize(fun_grad, x0, method="L-BFGS-B", jac=True,
                               options={"maxiter": maxiter, "maxfun": maxiter * 5,
                                        "ftol": 1e-14, "gtol": 1e-07})
                x0 = res.x
            except Exception:
                break
            if time.time() - t0 > TIME_BUDGET - 10:
                break
        X = x0.reshape(n, d)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def hard_polish(X, iters=1200, lr0=0.02, beta_w=400.0):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        cur = _max_cos(X)
        best_l = X.copy()
        best_v = cur
        lr = lr0
        for it in range(iters):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # contacts: top pairs within delta
            delta = 0.004 if it < iters // 2 else 0.0015
            # ensure at least 250 pairs
            # fast threshold
            thr = m - delta
            mask = G > thr
            # symmetric weight with sharpening
            W = np.exp(beta_w * (G - m)) * mask
            np.fill_diagonal(W, 0.0)
            # tangential force: -W@X + X * diag stuff
            WX = W @ X
            wsum_g = (W * G).sum(axis=1, keepdims=True)
            F = -WX + X * wsum_g
            # normalize force scale
            fn = np.linalg.norm(F)
            if not np.isfinite(fn) or fn == 0:
                break
            # step
            Xn = X + (lr / (1.0 + fn / n)) * F
            Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
            v = _max_cos(Xn)
            if v < m - 1e-12:
                X = Xn
                if v < best_v:
                    best_v = v
                    best_l = Xn.copy()
                lr = min(lr * 1.05, 0.05)
            else:
                # greedy: accept if not worse by much early, else shrink
                if v < m + 1e-6 and (it % 5 == 0):
                    X = Xn
                else:
                    lr *= 0.7
                    if lr < 1e-6:
                        lr = 1e-6
            if it % 200 == 199:
                beta_w = min(beta_w * 1.5, 2000.0)
            if time.time() - t0 > TIME_BUDGET - 5:
                break
        return best_l, best_v

    # trial 0: direct polish of loaded best (no noise) with short run
    try:
        Xp = smooth_opt(x_best, [80.0, 250.0, 700.0], [120, 120, 150])
        Xp2, v2 = hard_polish(Xp, iters=800)
        v = _max_cos(Xp2)
        if v < best_val:
            best_val = v
            x_best = Xp2.copy()
    except Exception:
        pass

    trial = 0
    # main basin-hopping loop
    while time.time() - t0 < TIME_BUDGET - 12 and trial < 40:
        trial += 1
        remain = TIME_BUDGET - (time.time() - t0)
        if remain < 12:
            break
        # perturbation schedule
        X0 = x_best.copy()
        if trial % 5 == 0:
            k = 10
            sigma_big = 0.55
            sigma_all = 0.02
        elif trial % 2 == 0:
            k = 5
            sigma_big = 0.35
            sigma_all = 0.015
        else:
            k = 3
            sigma_big = 0.25
            sigma_all = 0.01
        idx = rng.choice(n, size=k, replace=False)
        X0 = X0 + sigma_all * rng.normal(size=(n, d))
        X0[idx] += sigma_big * rng.normal(size=(k, d))
        # occasionally fully reinsert 1-2 rattler-like points at random
        if trial % 3 == 0:
            j = rng.integers(0, n, size=2)
            rnd = rng.normal(size=(2, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            # farthest-point style: keep random if far, else keep
            X0[j] = rnd + 0.2 * rng.normal(size=(2, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

        # staged anneal; fewer iters when time low
        if remain < 40:
            betas = [150.0, 600.0]
            iters_list = [100, 120]
            hp_iters = 600
        else:
            betas = [80.0, 250.0, 700.0]
            iters_list = [100, 100, 120]
            hp_iters = 800
        try:
            Xo = smooth_opt(X0, betas, iters_list)
            Xo2, v2 = hard_polish(Xo, iters=hp_iters)
            v = _max_cos(Xo2)
            if v < best_val - 1e-12:
                best_val = v
                x_best = Xo2.copy()
                # intensify around new best immediately
                try:
                    Xo3 = smooth_opt(x_best, [700.0], [80])
                    Xo3b, v3b = hard_polish(Xo3, iters=500)
                    v3 = _max_cos(Xo3b)
                    if v3 < best_val:
                        best_val = v3
                        x_best = Xo3b.copy()
                except Exception:
                    pass
        except Exception:
            continue

    # fallback safety
    if not np.all(np.isfinite(x_best)) or best_val > init_val + 1e-9:
        x_best = init_best
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard against NaN/zero rows (no-op but ensures contract)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr = np.where(nr < 1e-12, 1.0, nr)
    x_best = x_best / nr
    return x_best

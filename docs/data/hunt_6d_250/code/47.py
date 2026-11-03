import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return X / n

def _smooth_polish(X0, beta, maxiter, maxcor=8):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        X = Y / nr
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -50.0, 0.0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            S = 1e-300
        loss = M + (np.log(S) - np.log(2.0)) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-20,
                                "gtol": 1e-12, "maxcor": int(maxcor),
                                "maxfun": int(maxiter * 8)})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    return _norm(Y)

def _refine_single(y0, Z, beta=3000.0, iters=70, step0=0.08):
    y = np.asarray(y0, dtype=np.float64).copy()
    yn = float(np.linalg.norm(y))
    if yn < 1e-12 or not np.isfinite(yn):
        y = Z[0] * 0.0
        y[0] = 1.0
    else:
        y = y / yn
    dots = Z @ y
    best_m = float(np.max(dots))
    best_y = y.copy()
    cur = y.copy()
    for it in range(int(iters)):
        dots = Z @ cur
        M = float(np.max(dots))
        D = beta * (dots - M)
        np.clip(D, -30.0, 0.0, out=D)
        w = np.exp(D)
        s = float(np.sum(w))
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Z
        gt = g - float(g @ cur) * cur
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt /= nn
        step = step0 / (1.0 + it * 0.07)
        nxt = cur - step * gt
        nn2 = float(np.linalg.norm(nxt))
        if nn2 < 1e-12:
            continue
        nxt /= nn2
        mn = float(np.max(Z @ nxt))
        if mn < best_m:
            best_m = mn
            best_y = nxt.copy()
        cur = nxt
    return best_y, best_m

def _try_reinsert(X, idx, rng, n_cand=2500):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Z = X[mask]
    cur_m = float(np.max(Z @ X[idx]))
    C = rng.standard_normal((int(n_cand), d))
    C = _norm(C)
    dots = C @ Z.T
    cm = np.max(dots, axis=1)
    b = int(np.argmin(cm))
    if cm[b] >= cur_m - 1e-12:
        return X, float(_max_cos(X)), False
    y_best, m_best = _refine_single(C[b], Z)
    if m_best >= cur_m - 2e-12:
        return X, float(_max_cos(X)), False
    Xn = X.copy()
    Xn[idx] = y_best
    v = _max_cos(Xn)
    if v < _max_cos(X) - 1e-13:
        return Xn, v, True
    return X, float(_max_cos(X)), False

def _rattler_sweep(X, rng, t_end, max_sweeps=3, n_cand=2500):
    n = X.shape[0]
    cur = X.copy()
    cur_v = _max_cos(cur)
    for _ in range(max_sweeps):
        if time.time() > t_end:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        pmax = np.max(G, axis=1)
        order = np.argsort(-pmax)
        K = min(n, 60)
        improved = False
        for k in range(K):
            if time.time() > t_end:
                break
            idx = int(order[k])
            # only attack points near the top
            if pmax[idx] < cur_v - 5e-6 and k > 25:
                continue
            cur, cur_v_new, ok = _try_reinsert(cur, idx, rng, n_cand=n_cand)
            if ok:
                cur_v = cur_v_new
                improved = True
        if not improved:
            break
    return cur, cur_v

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    n, d = 250, 6
    best = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm(z)
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best_v:
                        best_v = v
                        best = z.copy()
        except Exception:
            continue
    if best is None:
        rng0 = np.random.default_rng(0)
        best = _norm(rng0.standard_normal((n, d)))
        best_v = _max_cos(best)
    rng = np.random.default_rng(12345)
    Xbest = best.copy()
    Mbest = float(best_v)

    # Phase 1: quick rattler / worst-point repack on incumbent
    try:
        Xbest, Mbest_try = _rattler_sweep(Xbest, rng, t_end - 100.0, max_sweeps=2, n_cand=2000)
        if Mbest_try < Mbest:
            Mbest = float(Mbest_try)
        else:
            Xbest = best.copy()
            Mbest = float(best_v)
    except Exception:
        pass

    # Phase 2: basin hopping with large kicks + smooth LSE polish
    trial = 0
    sigmas = [0.03, 0.06, 0.10, 0.15, 0.22, 0.05, 0.12, 0.08]
    betas = [250.0, 400.0, 600.0, 900.0, 500.0, 350.0, 800.0, 1200.0]
    while time.time() < t_end - 22.0 and trial < 60:
        sig = float(sigmas[trial % len(sigmas)])
        # randomize a bit deterministically
        sig = sig * (0.7 + 0.6 * rng.random())
        beta = float(betas[(trial * 3) % len(betas)])
        maxiter = 90 if (trial % 3) else 140
        try:
            if trial % 4 == 3:
                # subset kick: perturb only 40 random points strongly
                Xp = Xbest.copy()
                sel = rng.choice(n, size=40, replace=False)
                Xp[sel] += sig * 2.0 * rng.standard_normal((40, d))
                Xp = _norm(Xp)
            else:
                Xp = Xbest + sig * rng.standard_normal((n, d))
                Xp = _norm(Xp)
            Xp2 = _smooth_polish(Xp, beta=beta, maxiter=maxiter)
            v2 = _max_cos(Xp2)
            # opportunistic second polish at higher beta if promising
            if v2 < Mbest + 2e-4 and time.time() < t_end - 30.0:
                Xp3 = _smooth_polish(Xp2, beta=min(beta * 2.0, 2000.0), maxiter=60)
                v3 = _max_cos(Xp3)
                if v3 < v2:
                    Xp2, v2 = Xp3, v3
            # local hole repair if close
            if v2 < Mbest + 5e-5 and time.time() < t_end - 25.0:
                Xp2, v2r = _rattler_sweep(Xp2, rng, time.time() + 4.0, max_sweeps=1, n_cand=1500)
                v2 = float(v2r)
            if v2 < Mbest - 1e-12:
                Xbest = Xp2
                Mbest = float(v2)
        except Exception:
            pass
        trial += 1

    # Phase 3: final intensive polish of champion
    try:
        if time.time() < t_end - 15.0:
            Xc = _smooth_polish(Xbest, beta=800.0, maxiter=150)
            vc = _max_cos(Xc)
            if vc < Mbest:
                Xbest, Mbest = Xc, vc
        if time.time() < t_end - 8.0:
            Xc = _smooth_polish(Xbest, beta=1500.0, maxiter=100)
            vc = _max_cos(Xc)
            if vc < Mbest:
                Xbest, Mbest = Xc, vc
        if time.time() < t_end - 5.0:
            Xc, vc = _rattler_sweep(Xbest, rng, t_end - 2.0, max_sweeps=2, n_cand=2500)
            if vc < Mbest:
                Xbest, Mbest = Xc, vc
    except Exception:
        pass

    out = _norm(np.asarray(Xbest, dtype=np.float64))
    # strict shrink guard
    if not np.all(np.isfinite(out)):
        out = _norm(best)
    return out

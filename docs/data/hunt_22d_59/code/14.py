import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lbfgs_polish(X0, beta, maxiter, t_end, time):
    n, d = X0.shape
    Y0 = X0.copy().reshape(-1).astype(np.float64, copy=True)
    # closure
    def fg(Yf):
        Y = Yf.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # exp
        E = G - m
        E *= beta
        # clip lower to avoid underflow (exp(-50)=0)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            S = 1.0
        f = m + np.log(S) / beta
        W = E / S  # normalized weights
        GX = W @ X  # (n,d)
        GX *= 2.0
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GXt = GX - dot * X
        GY = GXt / rn
        return f, GY.reshape(-1)
    try:
        import scipy.optimize as so
        opts = {"maxiter": int(maxiter), "ftol": 1e-22, "gtol": 1e-14,
                "maxfun": int(maxiter * 3 + 10), "maxcor": 25}
        res = so.minimize(fg, Y0, jac=True, method="L-BFGS-B", options=opts)
        Yf = res.x.reshape(n, d)
        Xf = _norm(Yf)
        return Xf, _max_cos(Xf)
    except Exception:
        return X0.copy(), _max_cos(X0)

def _coord_sweep(X0, beta, lr, sweeps, rng):
    X = X0.copy()
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    for s in range(sweeps):
        order = rng.permutation(n)
        for ii in order:
            xi = X[ii]
            dots = X @ xi
            dots[ii] = -10.0
            m = float(dots.max())
            # weights
            w = dots - m
            w *= beta
            # avoid overflow (w<=0)
            np.exp(w, out=w)
            w[ii] = 0.0
            S = float(w.sum())
            if S <= 0 or not np.isfinite(S):
                continue
            g = (w @ X) / S
            # tangent
            gd = float(g @ xi)
            g -= gd * xi
            gn = float(np.linalg.norm(g))
            if gn == 0 or not np.isfinite(gn):
                continue
            # normalized step
            step = lr / (1.0 + gn)
            xn = xi - step * g
            nn = float(np.linalg.norm(xn))
            if nn < 1e-12 or not np.isfinite(nn):
                continue
            X[ii] = xn / nn
        v = _max_cos(X)
        if v < best:
            best = v
            bestX = X.copy()
    return bestX, best

def _hard_momentum(X0, steps, lr0, lr1, marg0, marg1, mu, check_every=100):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    best = _max_cos(X)
    bestX = X.copy()
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        marg = float(np.exp(lm0 + (lm1 - lm0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m - marg)).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        # avoid div by zero
        Gr = (W @ X) / np.maximum(deg, 1.0)
        z = (deg[:, 0] == 0)
        if np.any(z):
            Gr[z] = 0.0
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _norm(X)
        if (s % check_every == check_every - 1) or s == steps - 1:
            v = _max_cos(X)
            if v < best:
                best = v
                bestX = X.copy()
    return bestX, best

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0

    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val0:
                best_val0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _norm(rng.standard_normal((59, 22)))
        best_val0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = best_val0

    def left(sec=8.0):
        return time.time() < deadline - sec

    # Stage A: L-BFGS escalating beta
    for beta, it in [(250.0, 600), (600.0, 600), (1200.0, 400)]:
        if not left(25.0):
            break
        Xf, vf = _lbfgs_polish(x_best, beta=beta, maxiter=it, t_end=deadline, time=time)
        if vf < best_val:
            best_val = vf
            x_best = Xf.copy()
        else:
            # keep smoothed even if slightly worse? only if within 1e-4 to allow continuation
            if vf < best_val + 1e-4:
                # adopt for further sharpening if smooth improved? stay on best for stability
                pass

    # Stage B: Gauss-Seidel coordinate sweeps (high accuracy)
    if left(20.0):
        Xc, vc = _coord_sweep(x_best, beta=1500.0, lr=0.06, sweeps=12, rng=rng)
        if vc < best_val:
            best_val = vc
            x_best = Xc.copy()

    if left(18.0):
        Xh, vh = _hard_momentum(x_best, steps=2500, lr0=0.004, lr1=0.0002,
                                marg0=0.004, marg1=0.0003, mu=0.88, check_every=100)
        if vh < best_val:
            best_val = vh
            x_best = Xh.copy()

    # Stage C: tiny basin hops with fast re-polish
    sigmas = [0.004, 0.007, 0.011, 0.005, 0.009, 0.014, 0.006, 0.003]
    si = 0
    while left(10.0) and si < 14:
        sigma = sigmas[si % len(sigmas)]
        si += 1
        noise = rng.standard_normal((59, 22))
        Xt = _norm(x_best + sigma * noise)
        # quick LBFGS
        Xf, vf = _lbfgs_polish(Xt, beta=600.0, maxiter=220, t_end=deadline, time=time)
        if vf < best_val:
            best_val = vf
            x_best = Xf.copy()
            # deepen this basin
            if left(12.0):
                Xc2, vc2 = _coord_sweep(x_best, beta=2000.0, lr=0.04, sweeps=6, rng=rng)
                if vc2 < best_val:
                    best_val = vc2
                    x_best = Xc2.copy()
        else:
            # try coordinate rescue with small effort
            if left(12.0):
                Xc2, vc2 = _coord_sweep(Xf, beta=1200.0, lr=0.05, sweeps=3, rng=rng)
                if vc2 < best_val:
                    best_val = vc2
                    x_best = Xc2.copy()

    # Final ultra-fine polish on best basin
    if left(9.0):
        Xf, vf = _lbfgs_polish(x_best, beta=1800.0, maxiter=300, t_end=deadline, time=time)
        if vf < best_val:
            best_val = vf
            x_best = Xf.copy()
    if left(5.0):
        Xc, vc = _coord_sweep(x_best, beta=2500.0, lr=0.03, sweeps=8, rng=rng)
        if vc < best_val:
            best_val = vc
            x_best = Xc.copy()
    if left(3.0):
        Xh, vh = _hard_momentum(x_best, steps=1200, lr0=0.0015, lr1=0.00015,
                                marg0=0.0012, marg1=0.00015, mu=0.9, check_every=200)
        if vh < best_val:
            best_val = vh
            x_best = Xh.copy()

    x_best = _norm(x_best)
    return x_best

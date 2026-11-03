import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_refine(Y0, steps, beta0, beta1, lr0, lr1):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    best_val = _max_cos(Y)
    best_Y = Y.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    b1t = 1.0; b2t = 1.0
    for t in range(1, steps + 1):
        frac = (t - 1) / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s > 0 and np.isfinite(s):
            G /= s
            Gr = G @ Y
            dot = np.sum(Gr * Y, axis=1, keepdims=True)
            Gr -= Y * dot
            b1t *= b1; b2t *= b2
            M *= b1; M += (1.0 - b1) * Gr
            V *= b2; V += (1.0 - b2) * (Gr * Gr)
            Mh = M / (1.0 - b1t)
            Vh = V / (1.0 - b2t)
            Y -= lr * Mh / (np.sqrt(Vh) + eps)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            Y /= rn
        if (t % 100 == 0) or (t == steps):
            v = _max_cos(Y)
            if v < best_val:
                best_val = v
                best_Y = Y.copy()
    return best_Y, best_val

def _contact_refine(Y0, steps, step0, tol):
    Y = Y0.copy()
    best_val = _max_cos(Y)
    best_Y = Y.copy()
    step = float(step0)
    for _ in range(steps):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        A = G > (m - tol)
        np.fill_diagonal(A, False)
        Af = A.astype(np.float64)
        if float(Af.sum()) == 0:
            break
        AY = Af @ Y
        row_w = np.sum(Af * G, axis=1, keepdims=True)
        F = -AY + Y * row_w
        Yt = Y + step * F
        rn = np.linalg.norm(Yt, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Yt /= rn
        Gt = Yt @ Yt.T
        np.fill_diagonal(Gt, -2.0)
        mt = float(Gt.max())
        if mt < m - 1e-12:
            Y = Yt
            if mt < best_val:
                best_val = mt
                best_Y = Y.copy()
            step = min(step * 1.08, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                step = 1e-13
    return best_Y, best_val

def _hole_best(Xr, rng, K=25000):
    D = Xr.shape[1]
    C = rng.standard_normal((K, D))
    C = _normalize(C)
    Dots = C @ Xr.T
    vals = Dots.max(axis=1)
    idx = int(np.argmin(vals))
    return C[idx].copy(), float(vals[idx])

def _refine_single(c0, Xr, steps=350, beta=500.0, lr=0.03, mu=0.9):
    c = c0.copy()
    n0 = float(np.linalg.norm(c))
    if n0 == 0:
        c = Xr[0].copy()
        n0 = float(np.linalg.norm(c))
    c = c / n0
    vel = np.zeros_like(c)
    for _ in range(steps):
        d = Xr @ c
        m = float(d.max())
        w = np.exp(beta * (d - m))
        S = float(w.sum())
        if not np.isfinite(S) or S <= 0:
            break
        g = (w @ Xr) / S
        gt = g - float(np.dot(g, c)) * c
        vel = mu * vel + (1.0 - mu) * gt
        c = c - lr * vel
        n = float(np.linalg.norm(c))
        if n == 0:
            break
        c /= n
    return c

def _lbfgs_polish(Y0, beta, maxiter):
    try:
        import scipy.optimize as _opt
    except Exception:
        return Y0.copy(), _max_cos(Y0)
    N, D = Y0.shape
    X0 = Y0.copy().reshape(-1)
    def fun(x):
        X = x.reshape(N, D)
        rsq = np.sum(X * X, axis=1, keepdims=True)
        rn = np.sqrt(np.maximum(rsq, 1e-24))
        Y = X / rn
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        Gs = G - m
        Gs *= beta
        # clip for exp overflow (Gs<=0)
        np.exp(Gs, out=Gs)
        np.fill_diagonal(Gs, 0.0)
        Sp = float(Gs.sum() * 0.5)
        if Sp <= 0 or not np.isfinite(Sp):
            return 1e9, np.zeros_like(x)
        L = m + np.log(Sp) / beta
        GY = (Gs @ Y) / Sp
        dot = np.sum(GY * Y, axis=1, keepdims=True)
        GX = (GY - Y * dot) / rn
        return float(L), GX.reshape(-1)
    try:
        res = _opt.minimize(fun, X0, jac=True, method='L-BFGS-B',
                            options={'maxiter': maxiter, 'ftol': 1e-22, 'gtol': 1e-14, 'maxfun': maxiter * 4})
        Xo = res.x.reshape(N, D)
        Yo = _normalize(Xo)
        return Yo, _max_cos(Yo)
    except Exception:
        return Y0.copy(), _max_cos(Y0)

def solve():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    t0 = time.time()
    T_END = t0 + 172.0
    T_HOP = t0 + 138.0

    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normalize(a)
            v = _max_cos(a)
            if v < best0:
                best0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = best0

    # Stage 0: polish record directly
    y, v = _adam_refine(x_best, steps=5000, beta0=60.0, beta1=450.0, lr0=0.007, lr1=0.001)
    if v < best_val:
        best_val = v; x_best = y.copy()
    else:
        y = x_best.copy()
    y2, v2 = _contact_refine(y, steps=1800, step0=0.004, tol=0.006)
    if v2 < best_val:
        best_val = v2; x_best = y2.copy()

    sigmas = [0.010, 0.022, 0.007, 0.035, 0.014, 0.055, 0.008, 0.028, 0.045, 0.012]
    trial = 0
    while time.time() < T_HOP and time.time() < T_END - 30.0:
        trial += 1
        mode = trial % 4
        if mode != 0:
            sigma = sigmas[(trial - 1) % len(sigmas)]
            noise = rng.standard_normal((59, 22))
            # target overloaded points with extra noise: degree from best
            try:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                m0 = float(G.max())
                deg = np.sum(G > (m0 - 0.004), axis=1)
                w = (deg - deg.min() + 1.0) / (deg.max() - deg.min() + 1.0)
                noise *= (0.6 + 1.8 * w[:, None])
            except Exception:
                pass
            Xt = _normalize(x_best + sigma * noise)
            y, v = _adam_refine(Xt, steps=2800, beta0=80.0, beta1=550.0, lr0=0.006, lr1=0.0009)
            y2, v2 = _contact_refine(y, steps=1200, step0=0.0035, tol=0.005)
            cand_y, cand_v = (y2, v2) if v2 < v else (y, v)
            if cand_v < best_val - 1e-12:
                best_val = cand_v; x_best = cand_y.copy()
        else:
            # kill-and-fill: replace 1-2 worst points by largest empty cap
            try:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                nk = 2 if (trial % 8 == 0) else 1
                Xt = x_best.copy()
                for k in range(nk):
                    idx = int(order[k])
                    mask = np.ones(59, dtype=bool); mask[idx] = False
                    Xr = Xt[mask]
                    c_best, _ = _hole_best(Xr, rng, K=22000)
                    c_best = _refine_single(c_best, Xr, steps=300, beta=600.0, lr=0.03)
                    # accept only if it improves this point's max
                    cur = float(np.max(Xr @ Xt[idx]))
                    new = float(np.max(Xr @ c_best))
                    if new < cur:
                        Xt[idx] = c_best
                    else:
                        # still insert with small prob to change topology
                        if rng.random() < 0.25:
                            Xt[idx] = c_best
                    Xt = _normalize(Xt + 0.004 * rng.standard_normal((59, 22)))
                y, v = _adam_refine(Xt, steps=2800, beta0=80.0, beta1=550.0, lr0=0.006, lr1=0.0009)
                y2, v2 = _contact_refine(y, steps=1200, step0=0.0035, tol=0.005)
                cand_y, cand_v = (y2, v2) if v2 < v else (y, v)
                if cand_v < best_val - 1e-12:
                    best_val = cand_v; x_best = cand_y.copy()
            except Exception:
                continue
        if time.time() > T_HOP:
            break

    # Final high-precision polish
    if time.time() < T_END - 25.0:
        y, v = _adam_refine(x_best, steps=6000, beta0=200.0, beta1=1200.0, lr0=0.0035, lr1=0.00035)
        if v < best_val:
            best_val = v; x_best = y.copy()
        if time.time() < T_END - 20.0:
            for beta, it in ((400.0, 120), (900.0, 120), (1800.0, 100)):
                if time.time() > T_END - 12.0:
                    break
                yl, vl = _lbfgs_polish(x_best, beta=beta, maxiter=it)
                if vl < best_val:
                    best_val = vl; x_best = yl.copy()
        if time.time() < T_END - 6.0:
            y, v = _contact_refine(x_best, steps=2000, step0=0.002, tol=0.003)
            if v < best_val:
                best_val = v; x_best = y.copy()
            y, v = _contact_refine(x_best, steps=1200, step0=0.0008, tol=0.0012)
            if v < best_val:
                best_val = v; x_best = y.copy()

    x_best = _normalize(x_best)
    return x_best

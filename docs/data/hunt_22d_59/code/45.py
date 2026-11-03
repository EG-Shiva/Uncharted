import time
import numpy as np

def _norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_opt(X0, steps, beta0, beta1, lr0, lr1, mu):
    X = X0.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    eye = np.eye(n)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_X = X.copy()
        # W = exp(beta*(G-m)), diag 0
        G -= m
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum() * 0.5)
        if S < 1e-30 or not np.isfinite(S):
            continue
        Grad = (G @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel *= mu
        vel += (1.0 - mu) * Gt
        X -= lr * vel
        # retract
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X /= rn
    v = _max_cos(X)
    if v < best_val:
        best_val = v
        best_X = X.copy()
    return best_X, best_val

def _hard_opt(X0, steps, lr0, lr1, mu, marg0, marg1):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        marg = float(np.exp(lm0 + (lm1 - lm0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_X = X.copy()
        W = (G >= (m - marg)).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        # avoid div by zero
        Grad = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Grad[zm, :] = 0.0
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel *= mu
        vel += (1.0 - mu) * Gt
        X -= lr * vel
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X /= rn
    v = _max_cos(X)
    if v < best_val:
        best_val = v
        best_X = X.copy()
    return best_X, best_val

def _refine_single(y0, others, steps=250, lr0=0.05, lr1=0.002, marg0=0.01, marg1=0.001):
    y = y0.copy()
    yn = float(np.linalg.norm(y))
    if yn < 1e-12 or not np.isfinite(yn):
        y = others[0].copy()
        yn = float(np.linalg.norm(y))
    y /= max(yn, 1e-12)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        marg = float(np.exp(lm0 + (lm1 - lm0) * frac))
        d = others @ y
        m = float(d.max())
        act = d >= (m - marg)
        if not np.any(act):
            break
        g = np.mean(others[act], axis=0)
        gt = g - y * float(g @ y)
        y -= lr * gt
        yn = float(np.linalg.norm(y))
        if yn < 1e-12 or not np.isfinite(yn):
            break
        y /= yn
    return y

def _hole_for(idx, Xt, rng, n_sample=6000):
    mask = np.ones(Xt.shape[0], dtype=bool)
    mask[idx] = False
    others = Xt[mask]
    d = others.shape[1]
    C = rng.standard_normal((n_sample, d))
    C = _norm(C)
    D = C @ others.T
    vals = D.max(axis=1)
    b = int(np.argmin(vals))
    y0 = C[b].copy()
    y = _refine_single(y0, others, steps=250, lr0=0.05, lr1=0.002, marg0=0.01, marg1=0.001)
    # evaluate
    v = float(np.max(others @ y))
    return y, v

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

    # quick baseline polish
    try:
        y, v = _smooth_opt(x_best, steps=2500, beta0=40.0, beta1=350.0, lr0=0.02, lr1=0.0015, mu=0.88)
        if v < best_val:
            best_val = v
            x_best = y.copy()
        else:
            y = x_best
        y2, v2 = _hard_opt(y, steps=800, lr0=0.004, lr1=0.0004, mu=0.85, marg0=0.005, marg1=0.0007)
        if v2 < best_val:
            best_val = v2
            x_best = y2.copy()
    except Exception:
        pass

    trial = 0
    # main diversification loop
    while time.time() < deadline - 30.0 and trial < 120:
        mode = trial % 6
        Xt = None
        try:
            if mode == 0:
                sigma = 0.30
                Xt = _norm(x_best + sigma * rng.standard_normal((59, 22)))
                y, v = _smooth_opt(Xt, steps=1800, beta0=30.0, beta1=300.0, lr0=0.02, lr1=0.0015, mu=0.88)
                y2, v2 = _hard_opt(y, steps=500, lr0=0.004, lr1=0.0004, mu=0.85, marg0=0.004, marg1=0.0006)
                y, v = (y2, v2) if v2 < v else (y, v)
            elif mode == 1:
                sigma = 0.55
                Xt = _norm(x_best + sigma * rng.standard_normal((59, 22)))
                y, v = _smooth_opt(Xt, steps=1800, beta0=25.0, beta1=250.0, lr0=0.022, lr1=0.0015, mu=0.88)
                y2, v2 = _hard_opt(y, steps=500, lr0=0.004, lr1=0.0004, mu=0.85, marg0=0.004, marg1=0.0006)
                y, v = (y2, v2) if v2 < v else (y, v)
            elif mode == 2:
                # fresh random basin with different energy path
                Xt = _norm(rng.standard_normal((59, 22)))
                y, v = _smooth_opt(Xt, steps=2200, beta0=20.0, beta1=320.0, lr0=0.025, lr1=0.0012, mu=0.9)
                y2, v2 = _hard_opt(y, steps=600, lr0=0.005, lr1=0.0003, mu=0.85, marg0=0.005, marg1=0.0006)
                y, v = (y2, v2) if v2 < v else (y, v)
            elif mode == 3:
                # kill-and-fill worst 2
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                k = 2 if (trial % 12 != 9) else 3
                Xt = x_best.copy()
                for j in range(k):
                    idx = int(order[j])
                    yn, _ = _hole_for(idx, Xt, rng, n_sample=6000)
                    Xt[idx] = yn
                Xt = _norm(Xt)
                y, v = _smooth_opt(Xt, steps=1800, beta0=35.0, beta1=350.0, lr0=0.018, lr1=0.0012, mu=0.88)
                y2, v2 = _hard_opt(y, steps=500, lr0=0.003, lr1=0.0003, mu=0.85, marg0=0.003, marg1=0.0005)
                y, v = (y2, v2) if v2 < v else (y, v)
            elif mode == 4:
                # subset block shake of 8 random points
                Xt = x_best.copy()
                sel = rng.choice(59, size=8, replace=False)
                Xt[sel] += 0.5 * rng.standard_normal((8, 22))
                Xt = _norm(Xt)
                y, v = _smooth_opt(Xt, steps=1800, beta0=30.0, beta1=320.0, lr0=0.02, lr1=0.0012, mu=0.88)
                y2, v2 = _hard_opt(y, steps=500, lr0=0.004, lr1=0.0004, mu=0.85, marg0=0.004, marg1=0.0006)
                y, v = (y2, v2) if v2 < v else (y, v)
            else:
                # small shake fine search
                sigma = 0.04
                Xt = _norm(x_best + sigma * rng.standard_normal((59, 22)))
                y, v = _smooth_opt(Xt, steps=1500, beta0=60.0, beta1=500.0, lr0=0.012, lr1=0.001, mu=0.88)
                y2, v2 = _hard_opt(y, steps=400, lr0=0.003, lr1=0.0003, mu=0.85, marg0=0.003, marg1=0.0005)
                y, v = (y2, v2) if v2 < v else (y, v)
            if v < best_val - 1e-12:
                best_val = v
                x_best = y.copy()
        except Exception:
            pass
        trial += 1

    # final long polish of global best
    try:
        if time.time() < deadline - 20.0:
            y, v = _smooth_opt(x_best, steps=5000, beta0=80.0, beta1=700.0, lr0=0.012, lr1=0.0006, mu=0.9)
            if v < best_val:
                best_val = v
                x_best = y.copy()
        if time.time() < deadline - 12.0:
            y, v = _hard_opt(x_best, steps=1800, lr0=0.003, lr1=0.0002, mu=0.85, marg0=0.004, marg1=0.0004)
            if v < best_val:
                best_val = v
                x_best = y.copy()
        if time.time() < deadline - 6.0:
            y, v = _hard_opt(x_best, steps=800, lr0=0.001, lr1=0.0001, mu=0.8, marg0=0.0012, marg1=0.0003)
            if v < best_val:
                best_val = v
                x_best = y.copy()
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

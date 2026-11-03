import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _smooth_opt(X, steps, beta0, beta1, lr0, lr1, mu):
    X = X.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights, diag already tiny
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S < 1e-30:
            continue
        Grad = (W @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        # m is current max before step; evaluate new max cheaply every 50 steps
        if (s % 50 == 49) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _hard_opt(X, steps, lr0, lr1, mu, marg0, marg1):
    X = X.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    lm0 = np.log(marg0)
    lm1 = np.log(marg1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        marg = float(np.exp(lm0 + (lm1 - lm0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m - marg)).astype(float)
        # per-row averaged repulsion
        deg = W.sum(axis=1, keepdims=True)
        Grad = (W @ X) / np.maximum(deg, 1.0)
        # zero rattlers
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Grad[zm, :] = 0.0
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 100 == 99) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def solve():
    import time
    import numpy as np
    t0 = time.time()
    np.random.seed(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _normed(np.random.randn(59, 22))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    def time_left(limit=158.0):
        return (time.time() - t0) < limit

    # Stage 0: direct polish of record
    try:
        X1 = _smooth_opt(best_X, steps=6000, beta0=30.0, beta1=280.0, lr0=0.02, lr1=0.001, mu=0.9)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best_val = v1
            best_X = X1
        X1 = _hard_opt(best_X, steps=3000, lr0=0.005, lr1=0.0002, mu=0.85, marg0=0.006, marg1=0.0006)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best_val = v1
            best_X = X1
    except Exception:
        pass

    sigmas = [0.008, 0.015, 0.025, 0.04, 0.06, 0.012, 0.03]
    n_trials = 60
    for trial in range(n_trials):
        if not time_left(150.0):
            break
        sigma = sigmas[trial % len(sigmas)]
        # perturb best
        noise = np.random.randn(59, 22)
        Xt = best_X + sigma * noise
        # every 7th trial: rattler replacement
        if trial % 7 == 6:
            try:
                G = best_X @ best_X.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                rowmax = G.max(axis=1)
                ratt = np.where(rowmax < m - 0.005)[0]
                Xt = best_X.copy()
                if len(ratt) > 0:
                    k = min(len(ratt), 2)
                    choice = np.random.choice(ratt, size=k, replace=False)
                    for idx in choice:
                        Xt[idx] = np.random.randn(22)
                    # small noise on rest
                    Xt = Xt + 0.008 * np.random.randn(59, 22)
                else:
                    # kick one random point hard
                    idx = int(np.random.randint(0, 59))
                    Xt[idx] = np.random.randn(22)
                    Xt = Xt + 0.008 * np.random.randn(59, 22)
            except Exception:
                pass
        Xt = _normed(Xt)
        # vary sharpness deterministically
        b0 = 20.0 + (trial % 3) * 10.0
        b1 = 200.0 + (trial % 4) * 40.0
        try:
            Xo = _smooth_opt(Xt, steps=3500, beta0=b0, beta1=b1, lr0=0.015, lr1=0.0008, mu=0.9)
            Xo = _hard_opt(Xo, steps=1500, lr0=0.004, lr1=0.0002, mu=0.85, marg0=0.005, marg1=0.0005)
        except Exception:
            continue
        v = _max_cos(Xo)
        if v < best_val:
            best_val = v
            best_X = Xo.copy()
        # early exit if time low
        if not time_left(150.0):
            break

    # final ultra-fine polish
    if time_left(158.0):
        try:
            Xf = _hard_opt(best_X, steps=2500, lr0=0.0012, lr1=0.00008, mu=0.8, marg0=0.002, marg1=0.0004)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                best_X = Xf
        except Exception:
            pass
    if time_left(165.0):
        try:
            Xf = _smooth_opt(best_X, steps=2000, beta0=150.0, beta1=500.0, lr0=0.003, lr1=0.0002, mu=0.9)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                best_X = Xf
            Xf = _hard_opt(best_X, steps=1500, lr0=0.0006, lr1=0.00005, mu=0.8, marg0=0.001, marg1=0.0003)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                best_X = Xf
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X

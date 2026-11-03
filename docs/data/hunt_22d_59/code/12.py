import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _smooth_opt(X0, steps, beta0, beta1, lr0, lr1, mu):
    X = X0.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / float(steps - 1) if steps > 1 else 0.0
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < best_val:
            best_val = m
            np.copyto(best, X)
        np.subtract(G, m, out=G)
        np.multiply(G, beta, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G) * 0.5)
        if S < 1e-30 or not np.isfinite(S):
            continue
        Gr = (G @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= dot * X
        vel *= mu
        vel += (1.0 - mu) * Gr
        X -= lr * vel
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(rn, 1e-12, out=rn)
        X /= rn
    v = _max_cos(X)
    if v < best_val:
        best_val = v
        np.copyto(best, X)
    return best, best_val

def _hard_opt(X0, steps, lr0, lr1, mu, marg0, marg1):
    X = X0.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best = X.copy()
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    n = X.shape[0]
    eye = np.eye(n, dtype=X.dtype)
    for s in range(steps):
        f = s / float(steps - 1) if steps > 1 else 0.0
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        marg = float(np.exp(lm0 + (lm1 - lm0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < best_val:
            best_val = m
            np.copyto(best, X)
        W = (G >= (m - marg)).astype(np.float64)
        deg = W.sum(axis=1, keepdims=True)
        Gr = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Gr[zm, :] = 0.0
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= dot * X
        vel *= mu
        vel += (1.0 - mu) * Gr
        X -= lr * vel
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(rn, 1e-12, out=rn)
        X /= rn
    v = _max_cos(X)
    if v < best_val:
        best_val = v
        np.copyto(best, X)
    return best, best_val

def _refine_single(Z, x0, steps=350, beta0=30.0, beta1=500.0, lr0=0.05, lr1=0.004):
    x = x0 / max(np.linalg.norm(x0), 1e-12)
    x = x.copy()
    best = x.copy()
    d0 = float(np.max(Z @ x))
    best_v = d0
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / float(steps - 1) if steps > 1 else 0.0
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = Z @ x
        m = float(np.max(dots))
        if m < best_v:
            best_v = m
            np.copyto(best, x)
        w = np.exp(beta * (dots - m))
        S = float(np.sum(w))
        if S < 1e-30:
            continue
        g = (w @ Z) / S
        g = g - float(np.dot(g, x)) * x
        x = x - lr * g
        x = x / max(np.linalg.norm(x), 1e-12)
    v = float(np.max(Z @ best))
    return best, v

def _best_void(Z, rng, n_cand=8000):
    d = Z.shape[1]
    C = rng.standard_normal((n_cand, d))
    C = _normed(C)
    D = C @ Z.T
    rm = D.max(axis=1)
    k = min(6, n_cand)
    idx = np.argpartition(rm, k - 1)[:k]
    best_x = None
    best_v = 1e9
    for j in idx:
        x0 = C[j]
        xr, vr = _refine_single(Z, x0)
        if vr < best_v:
            best_v = vr
            best_x = xr
    return best_x, best_v

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0

    Xbest = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normed(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                Xbest = a.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = _normed(rng.standard_normal((59, 22)))
        best_val = _max_cos(Xbest)

    # light initial squeeze (already near optimum)
    try:
        y, v = _smooth_opt(Xbest, steps=2500, beta0=80.0, beta1=600.0, lr0=0.006, lr1=0.0008, mu=0.9)
        if v < best_val:
            best_val = v
            Xbest = y.copy()
        y, v = _hard_opt(y, steps=800, lr0=0.004, lr1=0.0003, mu=0.85, marg0=0.005, marg1=0.0008)
        if v < best_val:
            best_val = v
            Xbest = y.copy()
    except Exception:
        pass

    trial = 0
    # main basin-hopping loop
    while True:
        if time.time() > deadline - 22.0:
            break
        op = trial % 6
        Xt = None
        try:
            if op == 0:
                sigma = 0.015 if (trial // 6) % 2 == 0 else 0.032
                Xt = Xbest + sigma * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
            elif op == 1:
                k = 3 + (trial % 5)  # 3..7
                idx = rng.choice(59, size=k, replace=False)
                Xt = Xbest + 0.008 * rng.standard_normal((59, 22))
                Xt[idx] = rng.standard_normal((k, 22))
                Xt = _normed(Xt)
            elif op == 2 or op == 5:
                # worst-point void reinsertion
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                # pick among 6 worst at random for diversity
                order = np.argsort(-rowmax)
                cand_pool = order[:6]
                ridx = int(cand_pool[rng.integers(0, len(cand_pool))])
                # occasionally target low-degree peripheral point instead
                if trial % 12 == 11:
                    deg = (G >= (float(G.max()) - 0.004)).sum(axis=1)
                    ridx = int(np.argmin(deg))
                Z = np.delete(Xbest, ridx, axis=0)
                rep, _ = _best_void(Z, rng, n_cand=7000)
                Xt = Xbest.copy()
                Xt[ridx] = rep
                # tiny jitter on others to unlock contacts
                Xt = Xt + 0.004 * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
            elif op == 3:
                # strong subset kick
                k = 5
                idx = rng.choice(59, size=k, replace=False)
                Xt = Xbest.copy()
                Xt[idx] = Xt[idx] + 0.45 * rng.standard_normal((k, 22))
                Xt = Xt + 0.006 * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
            else:  # op == 4 large global kick
                sigma = 0.07 if (trial % 2 == 0) else 0.12
                Xt = Xbest + sigma * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
        except Exception:
            trial += 1
            continue

        try:
            # short annealed polish
            b1 = 60.0 + float(rng.uniform(-15, 30))
            b2 = 500.0 + float(rng.uniform(-80, 150))
            y, v = _smooth_opt(Xt, steps=2000, beta0=max(25.0, b1), beta1=max(200.0, b2),
                               lr0=0.006, lr1=0.0009, mu=0.9)
            y2, v2 = _hard_opt(y, steps=700, lr0=0.0035, lr1=0.00025, mu=0.85,
                                marg0=0.005, marg1=0.0007)
            # keep best of the two
            if v2 < v:
                y, v = y2, v2
            if v < best_val - 1e-12:
                best_val = v
                Xbest = y.copy()
            elif v < best_val + 0.002 and (trial % 25 == 0):
                # occasional sideways move to diversify: keep in place? do nothing
                pass
        except Exception:
            pass
        trial += 1
        # safety: avoid infinite loop
        if trial > 500:
            break

    # final intensive polish of best
    try:
        if time.time() < deadline - 16.0:
            y, v = _smooth_opt(Xbest, steps=6000, beta0=150.0, beta1=1100.0,
                               lr0=0.004, lr1=0.00035, mu=0.9)
            if v < best_val:
                best_val = v
                Xbest = y.copy()
            else:
                y = Xbest.copy()
        if time.time() < deadline - 8.0:
            y2, v2 = _hard_opt(y, steps=2500, lr0=0.003, lr1=0.00015, mu=0.85,
                                marg0=0.004, marg1=0.0004)
            if v2 < best_val:
                best_val = v2
                Xbest = y2.copy()
        if time.time() < deadline - 4.0:
            y3, v3 = _hard_opt(Xbest, steps=900, lr0=0.001, lr1=0.0001, mu=0.8,
                                marg0=0.0015, marg1=0.0003)
            if v3 < best_val:
                best_val = v3
                Xbest = y3.copy()
    except Exception:
        pass

    Xbest = _normed(np.asarray(Xbest, dtype=np.float64))
    # final safety: replace any non-finite row
    for i in range(Xbest.shape[0]):
        if not np.all(np.isfinite(Xbest[i])) or np.linalg.norm(Xbest[i]) < 1e-9:
            Xbest[i] = _normed(rng.standard_normal((1, 22)))[0]
    Xbest = _normed(Xbest)
    return Xbest

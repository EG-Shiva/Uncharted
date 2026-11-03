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

def _smooth(X0, steps, beta0, beta1, lr0, lr1, mu, seed_track=True):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    if seed_track:
        bv = _max_cos(X)
        bX = X.copy()
    else:
        bv = 1e9
        bX = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    eye = np.eye(n)
    for s in range(steps):
        f = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S < 1e-30 or not np.isfinite(S):
            continue
        Gr = (W @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= X * dot
        # clip extreme grad
        vel = mu * vel + (1.0 - mu) * Gr
        X = X - lr * vel
        X = _norm(X)
        if (s % 100 == 99) or (s == steps - 1):
            v = _max_cos(X)
            if v < bv:
                bv = v
                bX = X.copy()
    return bX, bv

def _hard(X0, steps, lr0, lr1, mu, marg0, marg1):
    X = X0.copy()
    bv = _max_cos(X)
    bX = X.copy()
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    n = X.shape[0]
    vel = np.zeros_like(X)
    for s in range(steps):
        f = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        marg = float(np.exp(lm0 + (lm1 - lm0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m - marg)).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        Gr = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Gr[zm, :] = 0.0
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= X * dot
        vel = mu * vel + (1.0 - mu) * Gr
        X = X - lr * vel
        X = _norm(X)
        if (s % 150 == 149) or (s == steps - 1):
            v = _max_cos(X)
            if v < bv:
                bv = v
                bX = X.copy()
    return bX, bv

def _void_fill(fixed, nfill, rng, ncand=3000):
    # greedily add nfill points minimizing max dot to already placed points
    d = fixed.shape[1]
    placed = fixed.copy()
    out = np.empty((nfill, d))
    for k in range(nfill):
        C = rng.standard_normal((ncand, d))
        C = _norm(C)
        D = C @ placed.T
        mx = D.max(axis=1)
        idx = int(np.argmin(mx))
        out[k] = C[idx]
        placed = np.vstack([placed, out[k:k+1]])
    return out

def solve():
    t0 = time.time()
    DEAD = 158.0
    def left(sec=0.0):
        return (time.time() - t0) < (DEAD - sec)
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _norm(rng.standard_normal((59, 22)))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    # Stage 0: quick polish of incumbent
    try:
        X1, v1 = _smooth(best_X, steps=2000, beta0=40.0, beta1=350.0, lr0=0.02, lr1=0.002, mu=0.9)
        if v1 < best_val:
            best_val = v1; best_X = X1.copy()
        X1, v1 = _hard(best_X, steps=500, lr0=0.004, lr1=0.0004, mu=0.85, marg0=0.005, marg1=0.0008)
        if v1 < best_val:
            best_val = v1; best_X = X1.copy()
    except Exception:
        pass

    trial = 0
    # topology-changing basin hopping
    while left(18.0) and trial < 60:
        trial += 1
        try:
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # contact counts with loose threshold to rank overloaded
            C = (G >= (m - 0.004)).sum(axis=1)
            order = np.argsort(-C, kind="stable")
        except Exception:
            order = np.arange(59)
        mode = trial % 5
        Xt = best_X.copy()
        if mode == 0 or mode == 1:
            # kill most overloaded k points, void fill
            k = 3 if mode == 0 else 5
            # add randomness: mostly top, one random
            kill = list(order[:k])
            if trial % 3 == 0:
                kill[-1] = int(rng.integers(0, 59))
                kill = list(dict.fromkeys(kill))
                k = len(kill)
            keep_mask = np.ones(59, dtype=bool)
            keep_mask[np.array(kill, dtype=int)] = False
            fixed = best_X[keep_mask]
            fills = _void_fill(fixed, k, rng, ncand=2500)
            Xt[keep_mask] = fixed
            Xt[np.array(kill, dtype=int)] = fills
            # tiny shake on survivors
            Xt[keep_mask] += 0.006 * rng.standard_normal((59 - k, 22))
            Xt = _norm(Xt)
            s_steps, h_steps = 1000, 350
            b0, b1 = 60.0, 500.0
        elif mode == 2:
            # kill random block + void fill
            k = 4
            kill = rng.choice(59, size=k, replace=False)
            keep_mask = np.ones(59, dtype=bool)
            keep_mask[kill] = False
            fixed = best_X[keep_mask]
            fills = _void_fill(fixed, k, rng, ncand=2500)
            Xt[keep_mask] = fixed
            Xt[kill] = fills
            Xt[keep_mask] += 0.008 * rng.standard_normal((59 - k, 22))
            Xt = _norm(Xt)
            s_steps, h_steps = 900, 300
            b0, b1 = 50.0, 450.0
        elif mode == 3:
            # subset strong shake of overloaded neighbourhood
            k = 12
            idx = list(order[:6]) + list(rng.choice(59, size=6, replace=False))
            idx = list(dict.fromkeys(idx))[:k]
            Xt[np.array(idx, dtype=int)] += 0.16 * rng.standard_normal((len(idx), 22))
            Xt = _norm(Xt)
            s_steps, h_steps = 800, 250
            b0, b1 = 50.0, 400.0
        else:
            # global small shake
            sig = 0.02 if (trial % 2 == 0) else 0.045
            Xt = _norm(best_X + sig * rng.standard_normal((59, 22)))
            s_steps, h_steps = 800, 250
            b0, b1 = 60.0, 450.0
        if not left(16.0):
            break
        try:
            Y, vy = _smooth(Xt, steps=s_steps, beta0=b0, beta1=b1, lr0=0.015, lr1=0.0015, mu=0.9)
            Y2, vy2 = _hard(Y, steps=h_steps, lr0=0.003, lr1=0.0003, mu=0.85, marg0=0.004, marg1=0.0006)
        except Exception:
            continue
        if vy2 < best_val - 1e-12:
            best_val = vy2; best_X = Y2.copy()
        elif vy < best_val - 1e-12:
            best_val = vy; best_X = Y.copy()

    # final fine polish if time remains
    if left(10.0):
        try:
            Y, vy = _smooth(best_X, steps=1800, beta0=120.0, beta1=900.0, lr0=0.006, lr1=0.0005, mu=0.9)
            if vy < best_val:
                best_val = vy; best_X = Y.copy()
            Y2, vy2 = _hard(best_X, steps=600, lr0=0.002, lr1=0.0002, mu=0.85, marg0=0.003, marg1=0.0005)
            if vy2 < best_val:
                best_val = vy2; best_X = Y2.copy()
        except Exception:
            pass
    best_X = _norm(best_X)
    return best_X

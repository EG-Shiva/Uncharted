import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _descent(x, steps, lr0, lr1, b0, b1, mu=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        beta = b0 + (b1 - b0) * frac
        G = x @ x.T
        # max for shift
        # set diag low so it never is max
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights, shifted for stability
        # clip argument to avoid overflow (already <=0)
        W = np.exp(beta * (G - m))
        # diag exp is exp(beta*(-2-m)) ~ 0, set to 0 exactly
        # zero diagonal by index (faster than fill)
        # use flat indexing
        W[np.diag_indices(n)] = 0.0
        S = float(np.sum(W)) * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (W @ x) / S
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        if mu != 0.0:
            v = mu * v + Gt
            step = v
        else:
            step = Gt
        x = x - lr * step
        # renormalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        # guard zero
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _holes(x, k, rng, n_sample=8000):
    n, d = x.shape
    # sample random directions
    Z = rng.standard_normal((n_sample, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    # max cosine to set: worst proximity
    M = Z @ x.T
    worst = np.max(M, axis=1)  # (n_sample,)
    idx = np.argpartition(worst, k)[:k]
    # order by best first
    idx = idx[np.argsort(worst[idx])]
    return Z[idx]

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=np.float64)
    # ensure shape (76,15)
    assert x0.shape == (76, 15)
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    best = x0.copy()
    best_val = _max_cos(best)

    # quick polish of incumbent (cheap, may shave 1e-12)
    if time.time() - t0 < 150:
        try:
            y = _descent(best.copy(), 1200, 0.008, 0.0005, 120.0, 900.0)
            y = _descent(y, 400, 0.001, 0.0002, 1200.0, 2500.0)
            vy = _max_cos(y)
            if vy < best_val:
                best, best_val = y, vy
        except Exception:
            pass

    sigmas = [0.005, 0.015, 0.04, 0.09, 0.18, 0.03, 0.07, 0.12]
    trial = 0
    # main basin-hopping loop, leave margin for 180s limit
    while time.time() - t0 < 158:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        y = best + sigma * rng.standard_normal(best.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)

        # every 3rd trial: hole-guided reseeding of a few points
        if trial % 3 == 0:
            try:
                k = int(rng.integers(3, 7))  # 3..6 points
                H = _holes(best, k, rng, n_sample=8000)
                idx = rng.choice(best.shape[0], size=k, replace=False)
                y[idx] = H + 0.01 * rng.standard_normal((k, best.shape[1]))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                pass
        # every 5th trial: reseed worst overloaded points more aggressively
        elif trial % 5 == 0:
            try:
                # estimate contacts of best to pick overloaded (compute once per trial set)
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                m0 = float(np.max(G))
                cnt = np.sum(G > m0 - 1e-9, axis=1)
                # pick top-k overloaded
                k2 = 4
                order = np.argsort(-cnt)[:k2]
                H2 = _holes(y, k2, rng, n_sample=5000)
                y[order] = H2
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                pass

        # annealed smooth-max descent
        # adapt steps to remaining time
        remain = 158 - (time.time() - t0)
        if remain < 3:
            break
        steps1 = 2500 if remain > 30 else 1200
        try:
            y = _descent(y, steps1, 0.02, 0.0015, 60.0, 600.0, mu=0.5)
        except Exception:
            continue
        if time.time() - t0 > 162:
            vy = _max_cos(y)
            if vy < best_val:
                best, best_val = y / np.linalg.norm(y, axis=1, keepdims=True), vy
            break
        try:
            y = _descent(y, 600, 0.002, 0.0002, 800.0, 2500.0, mu=0.0)
        except Exception:
            continue
        vy = _max_cos(y)
        if vy < best_val - 1e-12:
            # tiny shrink guard not needed but keep finite
            best, best_val = y, vy
            # early exit if clearly new record by margin? continue to push further
        # opportunistic: if close, try extra greedy single-point hole tuning on worst point
        # (cheap, only occasionally)
        if trial % 7 == 0 and vy < best_val + 0.005:
            try:
                # one more short low-lr sharpen
                y2 = _descent(y.copy(), 800, 0.001, 0.0001, 1500.0, 3000.0)
                vy2 = _max_cos(y2)
                if vy2 < best_val - 1e-12:
                    best, best_val = y2, vy2
            except Exception:
                pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # sanitize
    best = np.where(np.isfinite(best), best, 0.0)
    # in pathological case fallback to x0
    try:
        if not np.all(np.isfinite(best)) or best.shape != (76, 15):
            return x0
        nr = np.linalg.norm(best, axis=1)
        if np.any(nr < 1e-12) or np.any(~np.isfinite(nr)):
            return x0
    except Exception:
        return x0
    return best

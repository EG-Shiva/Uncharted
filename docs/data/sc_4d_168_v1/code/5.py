import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X

def _smooth_opt(X, steps, tau0, tau1, lr0, lr1):
    # annealed softmax of dots ~ smooth max; gradient step on sphere
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        # weights exp((G-m)/tau) in [0,1]
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True) + 1e-30
        Grad = (W @ X) / rs
        d = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - d * X
        X = X - lr * Gt
        X = _norm(X)
    return X

def _direct_polish(X, best_val, t_end):
    step = 2e-3
    # subgradient on true max with line search
    while time.time() < t_end:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        window = max(12.0 * step, 5e-4)
        Mask = G > (m - window)
        cnt = np.sum(Mask, axis=1, keepdims=True) + 1e-12
        Grad = (Mask.astype(float) @ X) / cnt
        d = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - d * X
        # try step
        Y = _norm(X - step * Gt)
        v = _max_cos(Y)
        if v < best_val:
            best_val = v
            X = Y
            step = min(step * 1.3, 5e-3)
        else:
            step *= 0.5
            if step < 1e-12:
                # restart with small random tangent kick to escape flat
                step = 1e-3
                # tiny jitter on worst points only
                idx = np.argmax(np.sum(Mask, axis=1))
                r = np.random.randn(4)
                r -= np.dot(r, X[idx]) * X[idx]
                rn = np.linalg.norm(r) + 1e-30
                X2 = X.copy()
                X2[idx] = _norm((X[idx] + 0.002 * r / rn)[None, :])[0]
                v2 = _max_cos(X2)
                if v2 < best_val:
                    X = X2
                    best_val = v2
                # if stuck too long, break to smooth
                if time.time() > t_end - 2:
                    break
    return X, best_val

def solve():
    t0 = time.time()
    BUDGET = 175.0
    rng = np.random.RandomState(7)
    # load start
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = rng.randn(168, 4)
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = Xb
    Xb = _norm(Xb)
    Xr = _norm(Xr)
    vb = _max_cos(Xb)
    vr = _max_cos(Xr)
    if vr < vb:
        Xb = Xr
        vb = vr
    best_val = vb
    n = 168

    def farthest_reinsert(Y, idx_list, n_cand=2000):
        # reinsert each idx at void far from others, sequentially
        for idx in idx_list:
            cand = rng.randn(n_cand, 4)
            cand = _norm(cand)
            # dots to all except idx
            Gc = cand @ Y.T  # n_cand x n
            # ignore self column by setting to -2
            Gc[:, idx] = -2.0
            worst = np.max(Gc, axis=1)
            pick = np.argmin(worst)
            Y[idx] = cand[pick]
        return Y

    trial = 0
    # main basin hopping, reserve ~22s for final squeeze
    while time.time() - t0 < BUDGET - 22.0:
        trial += 1
        Y = Xb.copy()
        mode = trial % 6
        if mode == 0:
            sigma = 0.015
            Y = _norm(Y + sigma * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1 = 700, 0.015, 0.002, 0.08, 0.008
        elif mode == 1:
            sigma = 0.07
            Y = _norm(Y + sigma * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1 = 1000, 0.02, 0.0015, 0.12, 0.008
        elif mode == 2:
            sigma = 0.22
            Y = _norm(Y + sigma * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1 = 1400, 0.025, 0.0012, 0.15, 0.006
        elif mode == 3:
            # reinsert most jammed + small jitter
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > (m - 0.003), axis=1)
            k = 4 + (trial % 4)  # 4..7
            idx = np.argsort(-deg)[:k]
            Y = farthest_reinsert(Y, idx, n_cand=2500)
            Y = _norm(Y + 0.02 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1 = 1200, 0.02, 0.0012, 0.12, 0.006
        elif mode == 4:
            # rattler + low-contact reinsert
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > (m - 0.003), axis=1)
            idx_lo = np.argsort(deg)[:3]
            G2 = Y @ Y.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            deg2 = np.sum(G2 > (m2 - 0.003), axis=1)
            idx_hi = np.argsort(-deg2)[:3]
            idx = np.unique(np.concatenate([idx_lo, idx_hi]))
            Y = farthest_reinsert(Y, idx, n_cand=2500)
            Y = _norm(Y + 0.01 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1 = 1200, 0.018, 0.001, 0.10, 0.005
        else:
            # random subset shake
            k = 12
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = _norm(rng.randn(k, 4))
            # also jitter rest slightly
            Y = _norm(Y + 0.03 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1 = 1400, 0.025, 0.0012, 0.14, 0.006

        # time guard: shorten steps if little time
        remain = (BUDGET - 22.0) - (time.time() - t0)
        if remain < 5:
            steps = min(steps, 300)
        Y = _smooth_opt(Y, steps, tau0, tau1, lr0, lr1)
        v = _max_cos(Y)
        if v < best_val - 1e-12:
            Xb = Y
            best_val = v
        # sideways acceptance very rarely to diversify without losing best
        # (no need to keep sideways as current; we always restart from best)

        if time.time() - t0 > BUDGET - 23.0:
            break

    # final intensive anneal from best
    if time.time() - t0 < BUDGET - 15.0:
        Xb = _smooth_opt(Xb, 2500, 0.008, 0.0008, 0.04, 0.004)
        v = _max_cos(Xb)
        if v < best_val:
            best_val = v

    # final direct squeeze for remaining time
    t_end = t0 + BUDGET - 1.0
    Xb, best_val = _direct_polish(Xb, best_val, t_end)

    Xb = _norm(Xb)
    return Xb

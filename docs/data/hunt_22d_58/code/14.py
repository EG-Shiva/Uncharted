import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, lr0, lr1, tau0, tau1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # soft-max weights
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        # tangential forces: F = W@X - X * ((W*G).sum)
        WX = W @ X
        WG = (W * G).sum(axis=1, keepdims=True)
        F = WX - X * WG
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        F /= rs
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hole_replace(X, idx, rng, n_cand=2000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Y = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ Y.T
    worst = dots.max(axis=1)
    best_c = C[np.argmin(worst)]
    X2 = X.copy()
    X2[idx] = best_c
    return X2

def _greedy_push(X, steps, eps0, eps1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = eps0 * (eps1 / eps0) ** f if eps0 != eps1 else eps0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy()
        xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = np.linalg.norm(ti) + 1e-12
        nj = np.linalg.norm(tj) + 1e-12
        # normalized push
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= np.linalg.norm(X[i])
        X[j] /= np.linalg.norm(X[j])
    return X

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # load starts
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (58, 22):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            c = np.where(np.isfinite(c), c, 0.0)
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if not cands:
        c = rng.standard_normal((58, 22))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    # quick initial quench to have clean baseline (short)
    if time.time() - t0 < 150:
        best = _quench(best, 3000, 0.01, 0.001, 0.006, 0.0015)
        best_val = _max_cos(best)

    # basin hopping trials
    trial = 0
    # contact threshold for picking victims
    while time.time() - t0 < 115 and trial < 600:
        trial += 1
        mode = trial % 4
        cur = best.copy()
        if mode == 0:
            sigma = 0.008
            cur = cur + sigma * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            steps = 2500
            q = _quench(cur, steps, 0.02, 0.002, 0.008, 0.0015)
        elif mode == 1:
            sigma = 0.03
            # shake subset of 8 random points harder
            idxs = rng.choice(58, size=8, replace=False)
            cur[idxs] += sigma * rng.standard_normal((8, 22))
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 3000, 0.03, 0.002, 0.01, 0.002)
        elif mode == 2:
            # reinsert worst point into hole
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # degree with near-max pairs
            cnt = (G > m - 0.004).sum(axis=1)
            # pick among top degrees
            order = np.argsort(-cnt)
            k = int(rng.integers(0, 4))
            victim = int(order[k])
            cur = _hole_replace(best, victim, rng, n_cand=3000)
            # small shake of victim neighbourhood
            cur += 0.005 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 3000, 0.025, 0.0015, 0.009, 0.0015)
        else:
            # double reinsertion of max pair
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // 58; j = k % 58
            # replace one of them at random
            victim = i if rng.random() < 0.5 else j
            cur = _hole_replace(best, victim, rng, n_cand=3000)
            # also shake the other a bit
            cur[j if victim == i else i] += 0.01 * rng.standard_normal(22)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2500, 0.02, 0.002, 0.008, 0.0012)
        v = _max_cos(q)
        if v < best_val - 1e-12:
            best, best_val = q, v
        # opportunistic extra greedy push if very close (within 2e-5) to squeeze
        if v < best_val + 3e-05 and time.time() - t0 < 110:
            # try short greedy push on q to see if it dips below best
            q2 = _greedy_push(q.copy(), 2000, 0.002, 0.0002)
            v2 = _max_cos(q2)
            if v2 < best_val - 1e-12:
                best, best_val = q2, v2

    # final long polish on best
    remain = 170 - (time.time() - t0)
    if remain > 8:
        s1 = 12000 if remain > 40 else 6000
        best = _quench(best, s1, 0.012, 0.0003, 0.005, 0.0006)
        best_val = _max_cos(best)
    if 170 - (time.time() - t0) > 5:
        best = _greedy_push(best, 8000, 0.0015, 0.00005)
        best_val = _max_cos(best)
    if 170 - (time.time() - t0) > 5:
        best = _quench(best, 4000, 0.003, 0.0002, 0.0012, 0.0004)
        best = _greedy_push(best, 4000, 0.0006, 0.00002)

    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

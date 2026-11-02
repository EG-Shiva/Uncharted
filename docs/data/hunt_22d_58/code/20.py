import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _quench(X, steps, lr0, lr1, tau0, tau1):
    n, d = X.shape
    lrl0 = np.log(lr0); lrl1 = np.log(lr1)
    lrt0 = np.log(tau0); lrt1 = np.log(tau1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = float(np.exp(lrl0 + (lrl1 - lrl0) * f))
        tau = float(np.exp(lrt0 + (lrt1 - lrt0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        WX = W @ X
        WG = (W * G).sum(axis=1, keepdims=True)
        F = WX - X * WG
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        F /= rs
        X = X - lr * F
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
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
    best_c = C[int(np.argmin(worst))]
    X2 = X.copy()
    X2[idx] = best_c
    return X2

def _greedy_push(X, steps, eps0, eps1):
    n = X.shape[0]
    le0 = np.log(eps0); le1 = np.log(eps1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = float(np.exp(le0 + (le1 - le0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n; j = k % n
        g = float(G[i, j])
        xi = X[i].copy(); xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= np.linalg.norm(X[i]) + 1e-16
        X[j] /= np.linalg.norm(X[j]) + 1e-16
    return X

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (58, 22):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                c = np.where(np.isfinite(c), c, 0.0)
                c /= np.linalg.norm(c, axis=1, keepdims=True)
                cands.append(c)
        except Exception:
            pass
    if not cands:
        c = rng.standard_normal((58, 22))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    # initial cold polish to jam
    best = _quench(best, 4000, 0.012, 0.001, 0.005, 0.0008)
    best = _greedy_push(best, 1500, 0.001, 0.0001)
    best_val = _max_cos(best)

    trial = 0
    # basin hopping, fixed trial budget with time guard
    while time.time() - t0 < 120.0 and trial < 60:
        trial += 1
        mode = trial % 4
        cur = best.copy()
        if mode == 0:
            cur = cur + 0.008 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2000, 0.02, 0.002, 0.008, 0.0012)
        elif mode == 1:
            idxs = rng.choice(58, size=6, replace=False)
            cur[idxs] += 0.03 * rng.standard_normal((6, 22))
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2200, 0.03, 0.002, 0.01, 0.0015)
        elif mode == 2:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            victim = int(order[int(rng.integers(0, 4))])
            cur = _hole_replace(best, victim, rng, n_cand=2500)
            cur += 0.004 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2200, 0.025, 0.0015, 0.009, 0.001)
        else:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // 58; j = k % 58
            victim = i if rng.random() < 0.5 else j
            cur = _hole_replace(best, victim, rng, n_cand=2500)
            oth = j if victim == i else i
            cur[oth] += 0.01 * rng.standard_normal(22)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2000, 0.02, 0.002, 0.008, 0.001)
        v = _max_cos(q)
        if v < best_val - 1e-12:
            # squeeze before accepting
            q2 = _greedy_push(q, 800, 0.001, 0.0002)
            v2 = _max_cos(q2)
            if v2 < v:
                q, v = q2, v2
            best, best_val = q, v
        elif v < best_val + 2e-05 and time.time() - t0 < 115:
            q2 = _greedy_push(q.copy(), 800, 0.001, 0.0002)
            v2 = _max_cos(q2)
            if v2 < best_val - 1e-12:
                best, best_val = q2, v2

    # final jam: cold quench + greedy + cold quench
    if 170 - (time.time() - t0) > 8:
        best = _quench(best, 8000, 0.012, 0.0003, 0.004, 0.0005)
        best_val = _max_cos(best)
    if 170 - (time.time() - t0) > 6:
        best = _greedy_push(best, 5000, 0.0012, 0.00004)
    if 170 - (time.time() - t0) > 6:
        best = _quench(best, 3000, 0.003, 0.0002, 0.001, 0.0004)
        best = _greedy_push(best, 2000, 0.0005, 0.00002)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

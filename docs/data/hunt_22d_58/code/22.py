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
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        WX = W @ X
        WG = (W * G).sum(axis=1, keepdims=True)
        F = WX - X * WG
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        F /= rs
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hole_replace(X, idx, rng, n_cand=3000):
    n, d = X.shape
    Y = np.delete(X, idx, axis=0)
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
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= np.linalg.norm(X[i])
        X[j] /= np.linalg.norm(X[j])
    return X

def _lp_polish(X, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    rng = np.random.default_rng(1)
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
    # short clean of baseline
    best = _quench(best, 2000, 0.008, 0.001, 0.005, 0.0015)
    best_val = _max_cos(best)

    trial = 0
    # hopping budget ~85s
    while time.time() - t0 < 85 and trial < 80:
        trial += 1
        mode = trial % 4
        if mode == 0:
            cur = best + 0.008 * rng.standard_normal(best.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2200, 0.02, 0.002, 0.008, 0.0015)
        elif mode == 1:
            cur = best.copy()
            idxs = rng.choice(58, size=6, replace=False)
            cur[idxs] += 0.05 * rng.standard_normal((6, 22))
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2500, 0.03, 0.002, 0.01, 0.002)
        elif mode == 2:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            victim = int(order[int(rng.integers(0, 4))])
            cur = _hole_replace(best, victim, rng, n_cand=3000)
            cur += 0.004 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2500, 0.025, 0.0015, 0.009, 0.0015)
        else:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // 58; j = k % 58
            victim = i if rng.random() < 0.5 else j
            cur = _hole_replace(best, victim, rng, n_cand=3000)
            other = j if victim == i else i
            cur[other] += 0.01 * rng.standard_normal(22)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2200, 0.02, 0.002, 0.008, 0.0012)
        v = _max_cos(q)
        if v < best_val - 1e-12:
            best, best_val = q, v
        elif v < best_val + 2e-05 and time.time() - t0 < 80:
            q2 = _greedy_push(q.copy(), 1200, 0.002, 0.0002)
            v2 = _max_cos(q2)
            if v2 < best_val - 1e-12:
                best, best_val = q2, v2

    if time.time() - t0 < 130:
        best = _quench(best, 6000, 0.012, 0.0003, 0.005, 0.0006)
        best = _greedy_push(best, 4000, 0.0012, 0.00005)
        best = _quench(best, 3000, 0.003, 0.0002, 0.0012, 0.0004)

    if time.time() - t0 < 150:
        deadline = min(t0 + 168, time.time() + 20)
        best = _lp_polish(best, deadline)

    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

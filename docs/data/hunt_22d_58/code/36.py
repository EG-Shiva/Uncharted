import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, b0, b1, lr0, lr1, mu=0.8, clip=0.05):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = np.where(sn > clip, step * (clip / sn), step)
        X = X - step
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def _greedy_push(X, steps, e0, e1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1) if steps > 1 else 0.0
        eps = e0 * (e1 / e0) ** f if e0 != e1 else e0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy(); xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= (np.linalg.norm(X[i]) + 1e-12)
        X[j] /= (np.linalg.norm(X[j]) + 1e-12)
    return X

def _hole_replace(X, victim, rng, n_cand=3000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool); mask[victim] = False
    Y = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    dots = C @ Y.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    X2 = X.copy()
    X2[victim] = C[bi]
    return X2

def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel); nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    HOP_END = t0 + 140.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = _max_cos(best)

    it = 0
    while time.time() < HOP_END and it < 160:
        it += 1
        if time.time() >= HOP_END:
            break
        mode = it % 6
        Xc = best.copy()
        # current max and degrees for guided choice
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        rowmax = G.max(axis=1)
        # degrees with small gap for hub detection
        deg = (G > m - 0.004).sum(axis=1)
        if mode == 0:
            victim = int(np.argmax(rowmax))
            Xc = _hole_replace(Xc, victim, rng, n_cand=4000)
            Xc = _quench(Xc, 350, 80, 1500, 0.03, 0.004, mu=0.82)
            Xc = _greedy_push(Xc, 20, 0.008, 0.002)
        elif mode == 1:
            order = np.argsort(-deg)
            v1 = int(order[0]); v2 = int(order[1])
            Xc = _hole_replace(Xc, v1, rng, n_cand=3000)
            # recompute remaining for second hole using updated Xc
            Xc = _hole_replace(Xc, v2, rng, n_cand=3000)
            Xc = _quench(Xc, 400, 100, 2500, 0.025, 0.003, mu=0.85)
            Xc = _greedy_push(Xc, 25, 0.01, 0.002)
        elif mode == 2:
            k = 3
            idx = rng.choice(n, size=k, replace=False)
            noise = rng.standard_normal((k, d)) * 0.15
            Xc[idx] = Xc[idx] + noise
            Xc[idx] /= (np.linalg.norm(Xc[idx], axis=1, keepdims=True) + 1e-12)
            Xc = _quench(Xc, 300, 60, 1200, 0.03, 0.005, mu=0.8)
        elif mode == 3:
            Xc = Xc + rng.standard_normal(Xc.shape) * 0.02
            Xc /= (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
            Xc = _quench(Xc, 300, 120, 2000, 0.02, 0.003, mu=0.8)
            Xc = _greedy_push(Xc, 15, 0.006, 0.002)
        elif mode == 4:
            # worst pair: move one endpoint to hole far from the other
            k = int(np.argmax(G)); i = k // n; j = k % n
            # victim is higher-degree endpoint
            victim = int(i if deg[i] >= deg[j] else j)
            Xc = _hole_replace(Xc, victim, rng, n_cand=4000)
            Xc = _quench(Xc, 350, 90, 1800, 0.028, 0.004, mu=0.82)
        else:
            # 5-point hole restart of worst points
            order = np.argsort(-rowmax)
            for vv in order[:3]:
                Xc = _hole_replace(Xc, int(vv), rng, n_cand=2000)
            Xc = _quench(Xc, 450, 70, 2000, 0.03, 0.003, mu=0.85)
            Xc = _greedy_push(Xc, 20, 0.008, 0.002)
        v = _max_cos(Xc)
        if v < best_val - 1e-12:
            best = Xc
            best_val = v
            if time.time() > HOP_END - 10:
                break
    # final polish with remaining time (capped)
    try:
        remain = DEADLINE - time.time() - 2.0
        if remain > 3.0:
            best = _lp_polish(best, min(remain, 18.0), DEADLINE - 1.0)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

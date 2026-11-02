import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, lr0, lr1, tau0, tau1, mom=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    # log interpolation
    for s in range(steps):
        f = s / max(1, steps - 1)
        # avoid pow issues
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        # stable softmax
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        WX = W @ X
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        # weighted mean direction
        F = WX / rs - X * ((WX * X).sum(axis=1, keepdims=True) / rs)
        # tangent (already tangent, clean numerically)
        F = F - X * np.sum(F * X, axis=1, keepdims=True)
        V = mom * V + (1.0 - mom) * F
        Xn = X - lr * V
        Xn = Xn - X * np.sum((Xn - X) * X, axis=1, keepdims=True)
        nr = np.linalg.norm(Xn, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = Xn / nr
    return X

def _greedy(X, steps, eps0, eps1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = eps0 * (eps1 / eps0) ** f if eps0 != eps1 else eps0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n; j = k % n
        g = float(G[i, j])
        xi = X[i].copy(); xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ti /= (np.linalg.norm(ti) + 1e-12)
        tj /= (np.linalg.norm(tj) + 1e-12)
        X[i] = xi - eps * ti
        X[j] = xj - eps * tj
        ni = np.linalg.norm(X[i]); nj = np.linalg.norm(X[j])
        if ni > 0: X[i] /= ni
        if nj > 0: X[j] /= nj
    return X

def _hole(X, victims, rng, n_cand=2000):
    n, d = X.shape
    Y = X.copy()
    for v in victims:
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Y.T
        D[:, int(v)] = -2.0
        worst = D.max(axis=1)
        Y[int(v)] = C[int(np.argmin(worst))]
    return Y

def _lp_polish(x, seconds, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds: break
        if time.time() >= t_deadline: break
        if radius < 1e-13: break
        it += 1
        if it > 30: break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        window = max(25.0 * radius, 2e-4)
        order = np.argsort(-c, kind="mergesort")
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 350:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 350: break
        if len(sel_list) == 0: break
        sel = np.array(sel_list)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best - 1e-15:
            x = y; best = v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None; best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22): continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = _max_cos(v)
            if c < best_val:
                best_val = c; best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z; best_val = _max_cos(best)

    hop_end = min(t0 + 120.0, t0 + TIME_LIMIT - 35.0)
    it = 0
    while time.time() < hop_end and it < 45:
        it += 1
        if time.time() > t0 + TIME_LIMIT - 32: break
        mode = it % 5
        if mode == 0:
            cur = best + 0.006 * rng.standard_normal(best.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 500, 0.025, 0.003, 0.009, 0.0012, 0.85)
            q = _greedy(q, 80, 0.004, 0.0005)
        elif mode == 1:
            idxs = rng.choice(n, size=6, replace=False)
            cur = best.copy()
            cur[idxs] += 0.04 * rng.standard_normal((6, d))
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 600, 0.030, 0.0025, 0.010, 0.0013, 0.8)
            q = _greedy(q, 80, 0.004, 0.0005)
        elif mode == 2:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.005).sum(axis=1)
            order = np.argsort(-cnt, kind="stable")
            k = 4 if (it % 10) < 5 else 6
            victims = order[:k].tolist()
            if rng.random() < 0.5:
                victims[-1] = int(rng.integers(0, n))
            cur = _hole(best, victims, rng, n_cand=2000)
            cur += 0.004 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 650, 0.030, 0.002, 0.010, 0.0012, 0.8)
            q = _greedy(q, 100, 0.005, 0.0004)
        elif mode == 3:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i0 = k // n; j0 = k % n
            row = G[i0]
            top = np.argsort(-row, kind="stable")[:2].tolist()
            victims = [int(i0), int(j0)] + [int(t) for t in top[:1]]
            victims = list(dict.fromkeys(victims))[:3]
            cur = _hole(best, victims, rng, n_cand=2500)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 600, 0.028, 0.002, 0.009, 0.0011, 0.85)
            q = _greedy(q, 100, 0.005, 0.0004)
        else:
            victims = rng.choice(n, size=8, replace=False).tolist()
            cur = _hole(best, victims, rng, n_cand=1500)
            cur += 0.006 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 700, 0.032, 0.0025, 0.011, 0.0014, 0.8)
            q = _greedy(q, 80, 0.004, 0.0005)
        v = _max_cos(q)
        if v < best_val - 1e-12:
            best_val = v; best = q.copy()
        if best_val < 0.062:
            break

    remain = (t0 + TIME_LIMIT) - time.time() - 1.0
    if remain > 5:
        try:
            polished = _lp_polish(best, min(remain - 1.0, 20.0), t0 + TIME_LIMIT - 1.0)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished; best_val = vp
        except Exception:
            pass
    try:
        if time.time() < t0 + TIME_LIMIT - 1:
            cand = _greedy(best.copy(), 60, 0.001, 0.0002)
            vc = _max_cos(cand)
            if vc < best_val:
                best = cand
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best

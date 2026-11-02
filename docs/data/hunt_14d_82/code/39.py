import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench_lse(X, steps, lr0, lr1, b0, b1, rng=None, t_end=None):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    p1, p2 = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    e0, e1 = np.log(b0), np.log(b1)
    for s in range(steps):
        if t_end is not None and (s & 63) == 0 and time.time() > t_end:
            break
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        # E = exp(beta*(G-gm))
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        p1 *= b_1
        p2 *= b_2
        m1 = b_1 * m1 + (1.0 - b_1) * Gt
        m2 = b_2 * m2 + (1.0 - b_2) * (Gt * Gt)
        mh = m1 / (1.0 - p1)
        vh = m2 / (1.0 - p2)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        X = _norm(X)
    return X

def _hole_replace(X, rng, n_kill=2, n_try=4000):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    worst = G.max(axis=1)
    victims = np.argsort(-worst, kind="stable")[:n_kill]
    keep = np.ones(n, dtype=bool)
    keep[victims] = False
    fixed = X[keep]
    existing = fixed.copy()
    news = []
    for _ in range(n_kill):
        Z = rng.standard_normal((n_try, d))
        Z = _norm(Z)
        S = Z @ existing.T
        m = S.max(axis=1)
        bi = int(np.argmin(m))
        news.append(Z[bi])
        existing = np.vstack([existing, Z[bi][None, :]])
    Xnew = np.vstack([fixed] + [v[None, :] for v in news])
    return _norm(Xnew)

def _lp_polish(X, deadline, radius_init=3e-4, cap=1500, max_it=12):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_it and radius > 1e-14 and time.time() < deadline:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(25.0 * radius, 2e-7)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        z = rng.standard_normal((n, d))
        try:
            m = min(z.size, xb.size)
            z.ravel()[:m] = xb.ravel()[:m]
        except Exception:
            pass
        xb = z
    xb = _norm(xb)
    best_val = _max_cos(xb)
    X_best = xb.copy()
    # fast initial joint squeeze without LP
    if t_end - time.time() > 20:
        tmp = _quench_lse(X_best, 600, 0.02, 0.002, 80.0, 1200.0, t_end=t_end - 12)
        v = _max_cos(tmp)
        if v < best_val:
            best_val, X_best = v, tmp
    sigmas = [0.005, 0.015, 0.04, 0.10, 0.25, 0.02, 0.06, 0.15, 0.008, 0.35]
    it = 0
    # reserve ~10s for final polish
    while time.time() < t_end - 10.0:
        it += 1
        if time.time() >= t_end - 10.0:
            break
        mode = it % 28
        if mode == 0:
            X0 = _norm(rng.standard_normal((n, d)))
            steps, lr0, lr1, b0, b1 = 500, 0.03, 0.002, 50.0, 800.0
        elif mode % 7 == 0:
            nk = int(rng.integers(1, 4))
            try:
                X0 = _hole_replace(X_best, rng, n_kill=nk, n_try=4000)
            except Exception:
                X0 = _norm(X_best + 0.05 * rng.standard_normal((n, d)))
            steps, lr0, lr1, b0, b1 = 350, 0.02, 0.002, 60.0, 1000.0
        else:
            sigma = sigmas[it % len(sigmas)]
            # occasionally start from second-best perturbation via large noise
            base = X_best
            X0 = _norm(base + sigma * rng.standard_normal((n, d)))
            if sigma < 0.03:
                steps, lr0, lr1, b0, b1 = 250, 0.015, 0.0015, 100.0, 1500.0
            elif sigma < 0.12:
                steps, lr0, lr1, b0, b1 = 400, 0.02, 0.002, 60.0, 1000.0
            else:
                steps, lr0, lr1, b0, b1 = 500, 0.03, 0.002, 40.0, 700.0
        try:
            X1 = _quench_lse(X0, steps, lr0, lr1, b0, b1, t_end=t_end - 10.0)
        except Exception:
            continue
        v1 = _max_cos(X1)
        if v1 < best_val - 1e-12:
            best_val, X_best = v1, X1.copy()
            # extra refine immediately on improvement with high-beta micro quench
            if time.time() < t_end - 12.0:
                try:
                    X2 = _quench_lse(X_best, 300, 0.005, 0.0005, 400.0, 2500.0, t_end=t_end - 10.0)
                    v2 = _max_cos(X2)
                    if v2 < best_val:
                        best_val, X_best = v2, X2.copy()
                except Exception:
                    pass
        # opportunistic second-stage squeeze for near-best
        elif v1 < best_val + 2e-4 and (it % 5 == 0) and time.time() < t_end - 11.0:
            try:
                X2 = _quench_lse(X1, 200, 0.005, 0.0005, 500.0, 2500.0, t_end=t_end - 10.0)
                v2 = _max_cos(X2)
                if v2 < best_val - 1e-12:
                    best_val, X_best = v2, X2.copy()
            except Exception:
                pass
    # final intensive polish
    try:
        if time.time() < t_end - 8.0:
            Xf = _quench_lse(X_best, 1200, 0.008, 0.0003, 150.0, 3000.0, t_end=t_end - 4.5)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, X_best = vf, Xf.copy()
    except Exception:
        pass
    try:
        if time.time() < t_end - 3.5:
            Xl = _lp_polish(X_best, deadline=t_end - 0.8, radius_init=2e-4, cap=1500, max_it=12)
            vl = _max_cos(Xl)
            if vl < best_val:
                best_val, X_best = vl, Xl.copy()
    except Exception:
        pass
    X_best = _norm(X_best)
    return X_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def smooth_refine(Y0, steps, beta0, beta1, lr0, lr1, mu=0.85):
    Y = Y0.copy()
    V = np.zeros_like(Y)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True)
        row = np.maximum(row, 1e-30)
        EY = E @ Y
        Avg = EY / row
        dot = np.sum(Avg * Y, axis=1, keepdims=True)
        Gt = Avg - dot * Y
        V = mu * V + (1.0 - mu) * Gt
        bc = 1.0 - mu ** (it + 1)
        Vh = V / max(bc, 1e-12)
        Y = Y - lr * Vh
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        Y = Y / nrm
    return Y

def best_hole_sample(Xk, rng, n_samples=20000):
    n, d = Xk.shape[0], Xk.shape[1]
    C = rng.randn(n_samples, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xk.T
    scores = S.max(axis=1)
    idx = int(np.argmin(scores))
    return C[idx]

def refine_hole_vec(cc, Xk, steps=200, beta=250.0, lr=0.02):
    c = cc.copy()
    for _ in range(steps):
        g = Xk @ c
        mm = float(g.max())
        w = np.exp(beta * (g - mm))
        s = float(w.sum())
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        grad = w @ Xk
        gt = grad - float(grad @ c) * c
        c = c - lr * gt
        n = float(np.linalg.norm(c))
        if n < 1e-12 or not np.isfinite(n):
            break
        c /= n
    return c

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t_deadline is not None and now >= t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 5000:
            # keep most tight
            idx = np.argsort(-c[sel])[:5000]
            I = I[idx]; J = J[idx]
            cc = c[sel][idx]
            p = len(I)
        else:
            cc = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ar = np.arange(d)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    DEADLINE = T0 + 172.0
    rng = np.random.RandomState(0)
    np.random.seed(0)
    n, d = 95, 13
    X_best = None
    best_val = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            X = np.load(fname).astype(float)
            if X.shape != (n, d):
                continue
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            v = max_cos(X)
            if v < best_val:
                best_val = v
                X_best = X.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = rng.randn(n, d)
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = max_cos(X_best)
    X_cur = X_best.copy()
    cur_val = best_val

    SEARCH_END = T0 + 112.0
    it = 0
    while time.time() < SEARCH_END and it < 5000:
        it += 1
        if time.time() > DEADLINE - 55:
            break
        r = rng.rand()
        if r < 0.30:
            sigma = 0.003 + 0.012 * rng.rand()
            Xt = X_best + sigma * rng.randn(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, steps=650, beta0=60.0, beta1=450.0, lr0=0.02, lr1=0.002)
        elif r < 0.60:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            k = int(rng.randint(2, 7))
            order = np.argsort(-cnt + 0.2 * rng.randn(n))
            sel = order[:k]
            sigma = 0.06 + 0.14 * rng.rand()
            Xt = X_best.copy()
            Xt[sel] += sigma * rng.randn(k, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, steps=850, beta0=50.0, beta1=550.0, lr0=0.025, lr1=0.002)
        elif r < 0.82:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.003).sum(axis=1)
            order = np.argsort(-cnt + 0.05 * rng.randn(n))
            cand = order[:6]
            idx = int(cand[rng.randint(len(cand))])
            mask = np.ones(n, dtype=bool); mask[idx] = False
            Xk = X_best[mask]
            c = best_hole_sample(Xk, rng, n_samples=20000)
            c = refine_hole_vec(c, Xk, steps=200, beta=250.0, lr=0.02)
            Xt = X_best.copy()
            Xt[idx] = c
            # small shake of neighbours to open hole
            Xt = Xt + 0.004 * rng.randn(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, steps=850, beta0=50.0, beta1=550.0, lr0=0.02, lr1=0.002)
        else:
            sigma = 0.025 + 0.05 * rng.rand()
            Xt = X_cur + sigma * rng.randn(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, steps=550, beta0=40.0, beta1=320.0, lr0=0.02, lr1=0.003)
        G = Xt @ Xt.T
        np.fill_diagonal(G, -2.0)
        v = float(G.max())
        if not np.isfinite(v):
            continue
        if v < best_val - 1e-12:
            # squeeze promising basin briefly
            try:
                if time.time() < SEARCH_END - 2:
                    Xt2 = lp_polish(Xt, seconds=1.2, t_deadline=DEADLINE - 45)
                    v2 = max_cos(Xt2)
                    if np.isfinite(v2) and v2 < v:
                        Xt, v = Xt2, v2
            except Exception:
                pass
            X_best = Xt
            best_val = v
            X_cur = Xt
            cur_val = v
        elif v < cur_val + 0.0015:
            elapsed = time.time() - T0
            Tt = 0.001 * max(0.0, 1.0 - elapsed / 130.0)
            Tt = max(Tt, 2e-04)
            if v <= cur_val or rng.rand() < np.exp((cur_val - v) / Tt):
                X_cur = Xt
                cur_val = v
        # occasional recenter cur on best
        if it % 25 == 0:
            X_cur = X_best.copy()
            cur_val = best_val

    try:
        if time.time() < DEADLINE - 40:
            X_best = smooth_refine(X_best, steps=1200, beta0=100.0, beta1=900.0, lr0=0.012, lr1=0.001)
    except Exception:
        pass
    try:
        rem = DEADLINE - time.time()
        if rem > 8:
            X_best = lp_polish(X_best, seconds=min(rem - 2.0, 45.0), t_deadline=DEADLINE - 1.0)
    except Exception:
        pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    for i in range(n):
        if not np.all(np.isfinite(X_best[i])) or float(np.linalg.norm(X_best[i])) < 1e-9:
            vv = rng.randn(d); vv /= np.linalg.norm(vv)
            X_best[i] = vv
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best

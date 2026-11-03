import time
import numpy as np

def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _riesz(X, s, steps, lr0, lr1):
    n = X.shape[0]
    for it in range(steps):
        f = it / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        D2 = 2.0 - 2.0 * G
        np.fill_diagonal(D2, 1.0)
        np.clip(D2, 1e-6, 4.0, out=D2)
        W = D2 ** (-s * 0.5 - 1.0)
        np.fill_diagonal(W, 0.0)
        Gt = W @ X
        dot = np.einsum('ij,ij->i', Gt, X)[:, None]
        Tan = Gt - dot * X
        tn = np.sqrt(np.sum(Tan * Tan) / n) + 1e-30
        X = X - (lr / tn) * Tan
        X = _norm(X)
    return X

def _balanced_anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max(axis=1, keepdims=True)
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum(axis=1, keepdims=True) + 1e-300
        P = E / S
        A = 0.5 * (P + P.T)
        Grad = A @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        # normalize to keep steps stable
        tn = np.sqrt((Tan * Tan).sum()) + 1e-30
        scale = lr / max(1.0, tn * 0.15)
        x = x - scale * Tan
        x = _norm(x)
    return x

def _global_anneal(x, steps, beta0, beta1, lr0, lr1):
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        tn = np.sqrt((Tan * Tan).sum()) + 1e-30
        scale = lr / max(1.0, tn * 0.15)
        x = x - scale * Tan
        x = _norm(x)
    return x

def _refine_void(v0, Xrem, beta=120.0, steps=100, lr=0.12):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        v = v / (np.linalg.norm(v) + 1e-30)
    return v

def _void_for_rem(Xrem, rng, ncand=12000, beta=120.0):
    d = Xrem.shape[1]
    best_v = None
    best_s = 1e9
    chunk = 4000
    for s0 in range(0, ncand, chunk):
        m = min(chunk, ncand - s0)
        cand = rng.standard_normal((m, d))
        cand = _norm(cand)
        smax = (cand @ Xrem.T).max(axis=1)
        bi = int(np.argmin(smax))
        if float(smax[bi]) < best_s:
            best_s = float(smax[bi])
            best_v = cand[bi].copy()
    v = _refine_void(best_v, Xrem, beta=beta, steps=100, lr=0.12)
    return v

def _mass_reinsert(X, rng, K=12):
    n, d = X.shape
    per = _per_max(X)
    order = np.argsort(-per)
    victims = order[:K].copy()
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    Xkeep = X[keep_mask]
    # greedily reinsert
    new_pts = []
    cur = Xkeep
    for k in range(K):
        v = _void_for_rem(cur, rng, ncand=12000, beta=110.0)
        new_pts.append(v)
        cur = np.vstack([cur, v[None, :]])
    Xnew = X.copy()
    for idx, v in zip(victims, new_pts):
        Xnew[idx] = v
    return Xnew

def _core_shake(X, rng, kcore=22, s_core=0.65, s_rest=0.12):
    per = _per_max(X)
    order = np.argsort(-per)
    core = order[:kcore]
    Xn = X + s_rest * rng.standard_normal(X.shape)
    Xn[core] = X[core] + s_core * rng.standard_normal((kcore, X.shape[1]))
    return _norm(Xn)

def _lse_jam(X0, betas, iters):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Y = _norm(X0.copy())
    for beta, mi in zip(betas, iters):
        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            c = G[iu0]
            mv = c.max()
            E = np.exp(beta * (c - mv))
            S = E.sum() + 1e-300
            f = float(mv + np.log(S) / beta)
            W = np.zeros((n, n))
            W[iu0] = E
            W = W + W.T
            GX = (W @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
            return f, GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-5, 'maxcor': 10})
            Y = res.x.reshape(n, d)
            Y = _norm(Y)
        except Exception:
            pass
    return _norm(Y)

def _lp_polish(X, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2800:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2800]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _e8_roots():
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    for mask in range(256):
        v = np.empty(8)
        neg = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -0.5
                neg += 1
            else:
                v[k] = 0.5
        if neg % 2 == 0:
            pts.append(v)
    A = np.array(pts, dtype=float)
    return _norm(A)

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    bestX = None
    bestV = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < bestV:
                    bestV = c
                    bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = _norm(rng.standard_normal((n, d)))
        bestV = _max_cos(bestX)
    E8 = _e8_roots()
    it = 0
    while time.time() < deadline - 26.0 and it < 60:
        it += 1
        mode = it % 6
        try:
            if mode == 0:
                X0 = _core_shake(bestX, rng, kcore=22, s_core=0.7, s_rest=0.15)
                X0 = _riesz(X0, s=5.0, steps=180, lr0=0.05, lr1=0.008)
                X0 = _mass_reinsert(X0, rng, K=10)
                X0 = _balanced_anneal(X0, steps=220, beta0=25.0, beta1=220.0, lr0=0.10, lr1=0.03)
                X0 = _global_anneal(X0, steps=220, beta0=30.0, beta1=260.0, lr0=0.08, lr1=0.02)
                X1 = _lse_jam(X0, betas=[90.0, 220.0, 450.0], iters=[80, 90, 90])
            elif mode == 1:
                idx = rng.choice(240, size=n, replace=False)
                X0 = E8[idx] + 0.08 * rng.standard_normal((n, d))
                X0 = _norm(X0)
                X0 = _riesz(X0, s=6.0, steps=220, lr0=0.06, lr1=0.01)
                X0 = _balanced_anneal(X0, steps=250, beta0=20.0, beta1=200.0, lr0=0.12, lr1=0.03)
                X0 = _global_anneal(X0, steps=200, beta0=40.0, beta1=300.0, lr0=0.08, lr1=0.02)
                X1 = _lse_jam(X0, betas=[80.0, 200.0, 420.0], iters=[80, 90, 90])
            elif mode == 2:
                X0 = _norm(rng.standard_normal((n, d)))
                X0 = _riesz(X0, s=4.0, steps=250, lr0=0.07, lr1=0.01)
                X0 = _balanced_anneal(X0, steps=250, beta0=25.0, beta1=240.0, lr0=0.12, lr1=0.03)
                X0 = _mass_reinsert(X0, rng, K=6)
                X0 = _global_anneal(X0, steps=200, beta0=50.0, beta1=320.0, lr0=0.08, lr1=0.02)
                X1 = _lse_jam(X0, betas=[70.0, 190.0, 430.0], iters=[80, 90, 90])
            elif mode == 3:
                X0 = bestX + 0.30 * rng.standard_normal((n, d))
                X0 = _norm(X0)
                X0 = _mass_reinsert(X0, rng, K=14)
                X0 = _balanced_anneal(X0, steps=220, beta0=30.0, beta1=240.0, lr0=0.10, lr1=0.025)
                X0 = _global_anneal(X0, steps=220, beta0=40.0, beta1=300.0, lr0=0.08, lr1=0.02)
                X1 = _lse_jam(X0, betas=[90.0, 230.0, 480.0], iters=[80, 90, 90])
            elif mode == 4:
                X0 = _core_shake(bestX, rng, kcore=14, s_core=0.45, s_rest=0.08)
                X0 = _mass_reinsert(X0, rng, K=5)
                X0 = _riesz(X0, s=7.0, steps=150, lr0=0.04, lr1=0.008)
                X0 = _balanced_anneal(X0, steps=200, beta0=35.0, beta1=260.0, lr0=0.09, lr1=0.025)
                X0 = _global_anneal(X0, steps=200, beta0=60.0, beta1=340.0, lr0=0.07, lr1=0.02)
                X1 = _lse_jam(X0, betas=[100.0, 260.0, 500.0], iters=[70, 80, 80])
            else:
                X0 = bestX + 0.55 * rng.standard_normal((n, d))
                X0 = _norm(X0)
                X0 = _riesz(X0, s=3.0, steps=200, lr0=0.06, lr1=0.01)
                X0 = _balanced_anneal(X0, steps=220, beta0=20.0, beta1=210.0, lr0=0.11, lr1=0.03)
                X0 = _global_anneal(X0, steps=220, beta0=35.0, beta1=280.0, lr0=0.08, lr1=0.02)
                X1 = _lse_jam(X0, betas=[75.0, 180.0, 400.0], iters=[80, 90, 90])
        except Exception:
            continue
        try:
            v = _max_cos(X1)
        except Exception:
            continue
        if v < bestV - 1e-12:
            bestV = v
            bestX = X1.copy()
        if time.time() >= deadline - 26.0:
            break
    try:
        bestX = _lse_jam(bestX, betas=[550.0, 900.0], iters=[120, 120])
        bestX = _lp_polish(bestX, deadline - 1.0)
    except Exception:
        pass
    return _norm(bestX)

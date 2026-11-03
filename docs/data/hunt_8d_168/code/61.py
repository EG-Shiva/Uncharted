import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _balanced_quench(X, steps, b0, b1, lr0, lr1, mu=0.85, lam=0.12, bl_frac=0.25, t_lim=None):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    one_mu = 1.0 - mu
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if t_lim is not None and (s & 127) == 0 and time.time() > t_lim:
            break
        frac = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Sg = E.sum() / max(n, 1) + 1e-30
        Grad_g = (E @ X) / Sg
        bl = max(beta * bl_frac, 8.0)
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl * (G - mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-30
        Grad_l = (W @ X) / Rs
        Grad = Grad_g + lam * Grad_l
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 3.0 / (nn + 1e-12))
        Gt = Gt * scale
        V = mu * V + one_mu * Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best

def _hard_polish(X, steps=250, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=1e-6, t_lim=None):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(steps):
        if t_lim is not None and (s & 63) == 0 and time.time() > t_lim:
            break
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        active = (cnt[:, 0] > 0.5)
        if not np.any(active):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, deadline, r0=1.0e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            idx = np.argsort(c[sel])[-2500:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_hole(Y, p, iters=140, beta=90.0, lr0=0.1, lr1=0.003):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        gm = g.max()
        w = np.exp(beta * (g - gm))
        s = w.sum() + 1e-18
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0 or not np.finite(nr):
            break
        p /= nr
    return p

def _find_void(Y, rng, n_probe=8000):
    d = Y.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 4000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        bs[a:b] = (P[a:b] @ Y.T).max(axis=1)
    order = np.argsort(bs)
    cands = []
    for idx in order[:80]:
        p = P[idx]
        ok = True
        for q in cands:
            if abs(float(p @ q)) > 0.9:
                ok = False
                break
        if ok:
            cands.append(p)
        if len(cands) >= 5:
            break
    if len(cands) == 0:
        cands = [P[order[0]]]
    best_p = cands[0]
    best_v = 1e9
    for p in cands:
        q = _refine_hole(Y, p, iters=140, beta=80.0 + rng.rand() * 40.0)
        vv = float((Y @ q).max())
        if vv < best_v:
            best_v = vv
            best_p = q
    return best_p

def solve():
    t0 = time.time()
    BUDGET = 175.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            continue
    if best is None:
        r0 = np.random.RandomState(7)
        best = r0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    rng = np.random.RandomState(12345)
    # quick squeeze of starting best
    try:
        tmp = _lp_polish(best.copy(), t0 + 8.0, r0=8e-4)
        v = _max_cos(tmp)
        if v < best_val:
            best_val = v
            best = tmp.copy()
    except Exception:
        pass
    it = 0
    sigmas = [0.005, 0.010, 0.020, 0.035, 0.060]
    while time.time() < t0 + BUDGET - 20.0:
        it += 1
        if time.time() + 4.0 > t0 + BUDGET - 18.0:
            break
        try:
            u = rng.rand()
            if u < 0.62:
                X = best.copy()
                r = rng.rand()
                if r < 0.45:
                    sig = sigmas[rng.randint(len(sigmas))]
                    X = X + rng.randn(168, 8) * float(sig)
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                elif r < 0.85:
                    Gc = X @ X.T
                    np.fill_diagonal(Gc, -2.0)
                    rm = Gc.max(axis=1)
                    k = 1 + int(rng.randint(3))
                    idx = np.argsort(rm)[-k:]
                    for j in idx:
                        oth = np.delete(X, j, axis=0)
                        X[j] = _find_void(oth, rng, n_probe=6000 + int(rng.randint(4000)))
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                else:
                    sig = 0.02 + rng.rand() * 0.05
                    X = X + rng.randn(168, 8) * float(sig)
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                    Gc = X @ X.T
                    np.fill_diagonal(Gc, -2.0)
                    rm = Gc.max(axis=1)
                    j = int(np.argmax(rm))
                    oth = np.delete(X, j, axis=0)
                    X[j] = _find_void(oth, rng, n_probe=8000)
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
            elif u < 0.80:
                X = rng.randn(168, 8)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                Y = rng.randn(168, 8)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                alpha = 0.15 + rng.rand() * 0.35
                X = (1 - alpha) * best + alpha * Y
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            lam = 0.08 + rng.rand() * 0.12
            blf = 0.20 + rng.rand() * 0.15
            b0 = 22.0 + rng.rand() * 15.0
            # stage 1
            X = _balanced_quench(X, steps=750, b0=b0, b1=380.0, lr0=0.006, lr1=0.0006,
                                mu=0.85, lam=lam, bl_frac=blf, t_lim=t0 + BUDGET - 18.0)
            # stage 2 higher beta
            X = _balanced_quench(X, steps=550, b0=150.0, b1=650.0, lr0=0.003, lr1=0.00035,
                                mu=0.88, lam=lam, bl_frac=blf, t_lim=t0 + BUDGET - 18.0)
            X = _hard_polish(X, steps=180, lr0=0.0025, lr1=0.00012, tol0=2e-4, tol1=1e-6,
                             t_lim=t0 + BUDGET - 18.0)
            v = _max_cos(X)
            if v < best_val and np.all(np.isfinite(X)):
                # squeeze promising candidate
                if v < best_val + 2e-4 and time.time() < t0 + BUDGET - 22.0:
                    X2 = _lp_polish(X.copy(), min(time.time() + 4.0, t0 + BUDGET - 16.0), r0=1.0e-3)
                    v2 = _max_cos(X2)
                    if v2 < v and np.all(np.isfinite(X2)):
                        X, v = X2, v2
                if v < best_val:
                    best_val = v
                    best = X.copy()
        except Exception:
            continue
    # final squeeze
    try:
        if time.time() < t0 + BUDGET - 1.0:
            best = _lp_polish(best.copy(), t0 + BUDGET - 0.5, r0=7e-4)
            best_val = _max_cos(best)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

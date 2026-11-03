import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _balanced_quench(X, steps, b0, b1, lr0, lr1, lam=0.15, mu=0.85, bl_frac=0.25):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    om = 1.0 - mu
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
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
        scale = np.minimum(1.0, 3.0 / nn)
        Gt = Gt * scale
        V = mu * V + om * Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best, best_m

def _hard_polish(X, steps=250, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=1e-6):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        act = (cnt[:, 0] > 0.5)
        if not np.any(act):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        st = np.zeros_like(Gt)
        st[act] = Gt[act] / nn[act]
        X = X - lr * st
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_hole(Y, p, iters=120, beta=90.0, lr0=0.08, lr1=0.003):
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
        if nr == 0 or not np.isfinite(nr):
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
    for idx in order[:60]:
        p = P[idx]
        ok = True
        for q in cands:
            if abs(float(p @ q)) > 0.92:
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
        q = _refine_hole(Y, p, iters=100, beta=80.0 + rng.rand() * 50.0)
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
            pass
    if best is None:
        r0 = np.random.RandomState(7)
        best = r0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    rng = np.random.RandomState(12345)
    # stage 1: quick polish of incumbent to have tight baseline
    try:
        Xb, vb = _balanced_quench(best, steps=800, b0=40.0, b1=500.0, lr0=0.004, lr1=0.0004, lam=0.12)
        if vb < best_val:
            best_val = vb
            best = Xb.copy()
    except Exception:
        pass
    it = 0
    # main basin-hopping: mix perturbations + void surgery + anneal
    while time.time() < t0 + BUDGET - 28.0:
        it += 1
        if time.time() + 4.0 > t0 + BUDGET - 26.0:
            break
        try:
            u = rng.rand()
            if u < 0.55:
                X = best.copy()
                r = rng.rand()
                if r < 0.4:
                    sig = float(rng.choice(np.array([0.004, 0.010, 0.022, 0.045, 0.09])))
                    X = X + rng.randn(168, 8) * sig
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                elif r < 0.8:
                    Gc = X @ X.T
                    np.fill_diagonal(Gc, -2.0)
                    rm = Gc.max(axis=1)
                    k = 1 + int(rng.randint(4))
                    idx = np.argsort(rm)[-k:]
                    for j in idx:
                        oth = np.delete(X, j, axis=0)
                        X[j] = _find_void(oth, rng, n_probe=6000 + int(rng.randint(6000)))
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                else:
                    # block re-arrange: pick worst region and resample cluster
                    Gc = X @ X.T
                    np.fill_diagonal(Gc, -2.0)
                    rm = Gc.max(axis=1)
                    k = 4 + int(rng.randint(8))
                    idx = np.argsort(rm)[-k:]
                    cen = X[idx].mean(axis=0)
                    cen /= np.linalg.norm(cen) + 1e-18
                    for j in idx:
                        v = rng.randn(8) * 0.6 + cen * 2.0
                        v /= np.linalg.norm(v)
                        # push away from others via quick hole refine
                        oth = np.delete(X, j, axis=0)
                        # mix random + refined
                        if rng.rand() < 0.5:
                            X[j] = _refine_hole(oth, v, iters=60, beta=60.0 + rng.rand() * 60.0)
                        else:
                            X[j] = v
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                # anneal params vary to change basin width
                b1 = float(rng.choice(np.array([350.0, 500.0, 700.0, 900.0])))
                lam = float(rng.choice(np.array([0.05, 0.12, 0.20, 0.30])))
                steps = int(rng.choice(np.array([700, 1000, 1400])))
                lr0 = float(rng.choice(np.array([0.003, 0.005, 0.008])))
                Xq, vq = _balanced_quench(X, steps=steps, b0=25.0, b1=b1, lr0=lr0, lr1=0.0004, lam=lam)
                Xq = _hard_polish(Xq, steps=120, lr0=0.002, lr1=0.0001)
                vq2 = _max_cos(Xq)
                if vq2 < best_val and np.all(np.isfinite(Xq)):
                    best_val = vq2
                    best = Xq.copy()
            else:
                # fresh diverse restart with lower weight to explore new topology
                X = rng.randn(168, 8)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                # seed partially with best via crossover
                if rng.rand() < 0.6:
                    k = 40 + int(rng.randint(80))
                    sel = rng.choice(168, size=k, replace=False)
                    X[sel] = best[sel] + rng.randn(k, 8) * 0.02
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                Xq, vq = _balanced_quench(X, steps=1000, b0=20.0, b1=600.0, lr0=0.006, lr1=0.0005, lam=0.15)
                Xq = _hard_polish(Xq, steps=120, lr0=0.002, lr1=0.0001)
                vq2 = _max_cos(Xq)
                if vq2 < best_val and np.all(np.isfinite(Xq)):
                    best_val = vq2
                    best = Xq.copy()
        except Exception:
            continue
    # final squeeze: hard + LP on best
    try:
        best = _hard_polish(best, steps=250, lr0=0.003, lr1=0.0001)
        nv = _max_cos(best)
        if nv < best_val:
            best_val = nv
    except Exception:
        pass
    remain = (t0 + BUDGET - 2.0) - time.time()
    if remain > 4.0:
        try:
            secs = min(remain - 1.0, 22.0)
            best = _lp_polish(best, seconds=secs, r0=1.2e-3)
            best_val = _max_cos(best)
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

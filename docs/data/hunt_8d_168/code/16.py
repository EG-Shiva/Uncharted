import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _quench(X, steps, beta0, beta1, lr0, lr1, rng=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    b1p = 1.0
    b2p = 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_hole(Y, p, iters=200, beta=80.0, lr0=0.15, lr1=0.005):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        m = g.max()
        w = np.exp(beta * (g - m))
        s = w.sum() + 1e-12
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def solve():
    t0 = time.time()
    BUDGET = 175.0
    HOP_END = 133.0
    n, d = 168, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    rng = np.random.RandomState(12345)
    # hole finder closure using rng
    def find_holes(Y, k, n_probe=16000):
        P = rng.randn(n_probe, d)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        CH = 4000
        bs = np.empty(n_probe)
        for a0 in range(0, n_probe, CH):
            b0 = min(a0 + CH, n_probe)
            D = P[a0:b0] @ Y.T
            bs[a0:b0] = D.max(axis=1)
        order = np.argsort(bs)
        sel = []
        for idx in order:
            if len(sel) >= k:
                break
            p = P[idx]
            ok = True
            for q in sel:
                if abs(float(p @ q)) > 0.92:
                    ok = False
                    break
            if ok:
                sel.append(p)
        return sel
    it = 0
    while time.time() - t0 < HOP_END:
        it += 1
        if time.time() - t0 > BUDGET - 45:
            break
        X = best.copy()
        u = rng.rand()
        if u < 0.32:
            sigmas = (0.008, 0.015, 0.030, 0.060)
            sigma = sigmas[rng.randint(len(sigmas))]
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 30.0 + rng.rand() * 30.0
            beta1 = 200.0 + rng.rand() * 400.0
            steps = 900 + int(rng.randint(700))
            lr0 = 0.003 + rng.rand() * 0.005
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        elif u < 0.62:
            # worst-point to hole re-insertion
            ppm = _per_point_max(X)
            k = 1 + int(rng.randint(3))  # 1..3
            if rng.rand() < 0.7:
                idx = np.argsort(ppm)[-k:]
            else:
                idx = rng.choice(n, k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            Ybase = X[keep]
            holes = find_holes(Ybase, k, n_probe=16000)
            for t_, j in enumerate(idx):
                if t_ < len(holes):
                    h = _refine_hole(Ybase, holes[t_], iters=200, beta=80.0)
                else:
                    h = rng.randn(d); h /= np.linalg.norm(h)
                    h = _refine_hole(Ybase, h, iters=200, beta=80.0)
                X[j] = h
                Ybase = np.vstack([Ybase, h[None, :]]) if t_ < 1 else Ybase
            # small global jitter
            X = X + rng.randn(n, d) * 0.004
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 40.0
            beta1 = 300.0 + rng.rand() * 300.0
            steps = 1000 + int(rng.randint(700))
            lr0 = 0.004 + rng.rand() * 0.004
            lr1 = lr0 * 0.12
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        elif u < 0.84:
            # coherent component shift + rattler recycle
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m_ = G.max()
            # contact graph with loose threshold to find jammed clusters
            thr = m_ - 5e-4
            adj = (G > thr)
            # BFS components
            seen = np.zeros(n, dtype=bool)
            comps = []
            for s in range(n):
                if seen[s]:
                    continue
                stack = [s]; seen[s] = True; comp = [s]
                while stack:
                    cur = stack.pop()
                    nbrs = np.where(adj[cur])[0]
                    for nb in nbrs:
                        if not seen[nb]:
                            seen[nb] = True
                            stack.append(nb)
                            comp.append(nb)
                if len(comp) > 1:
                    comps.append(comp)
            if len(comps) > 0 and rng.rand() < 0.8:
                comp = comps[rng.randint(len(comps))]
                if len(comp) > 25:
                    comp = list(rng.choice(comp, 25, replace=False))
                sigma_c = 0.02 + rng.rand() * 0.04
                shift = rng.randn(d) * sigma_c
                X[comp] = X[comp] + shift[None, :]
                X[comp] += rng.randn(len(comp), d) * 0.008
            else:
                # random cap shake
                c = rng.randn(d); c /= np.linalg.norm(c)
                dots = X @ c
                k2 = 8 + int(rng.randint(8))
                idx2 = np.argsort(dots)[-k2:]
                X[idx2] += rng.randn(k2, d) * 0.05
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 35.0 + rng.rand() * 20.0
            beta1 = 350.0 + rng.rand() * 250.0
            steps = 1100 + int(rng.randint(600))
            lr0 = 0.005
            lr1 = 0.0005
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        else:
            # rattler swarm randomization into holes
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m_ = G.max()
            ppm = G.max(axis=1)
            ratt = np.where(ppm < m_ - 1e-4)[0]
            if len(ratt) < 4:
                ratt = np.argsort(ppm)[:10]
            k3 = min(len(ratt), 4 + int(rng.randint(6)))
            idx3 = rng.choice(ratt, k3, replace=False) if len(ratt) > k3 else ratt
            keep = np.ones(n, dtype=bool); keep[idx3] = False
            Ybase = X[keep]
            holes = find_holes(Ybase, k3, n_probe=20000)
            for t_, j in enumerate(idx3):
                if t_ < len(holes):
                    h = _refine_hole(Ybase, holes[t_], iters=250, beta=100.0)
                else:
                    h = rng.randn(d); h /= np.linalg.norm(h)
                X[j] = h
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 45.0
            beta1 = 400.0 + rng.rand() * 200.0
            steps = 1100 + int(rng.randint(600))
            lr0 = 0.004 + rng.rand() * 0.003
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        if not np.all(np.isfinite(Xq)):
            continue
        vq = _max_cos(Xq)
        if vq + 1e-12 < best_val:
            # short polish to squeeze
            rem = (t0 + BUDGET) - time.time()
            if rem > 12:
                try:
                    Xp = _lp_polish(Xq, min(1.2, rem - 8.0))
                    vp = _max_cos(Xp)
                    if vp < vq and np.all(np.isfinite(Xp)):
                        Xq, vq = Xp, vp
                except Exception:
                    pass
            if vq + 1e-12 < best_val:
                best = Xq
                best_val = vq
        elif vq < best_val + 0.0015 and (it % 7 == 0):
            rem = (t0 + BUDGET) - time.time()
            if rem > 15:
                try:
                    Xp = _lp_polish(Xq, 0.8)
                    vp = _max_cos(Xp)
                    if vp + 1e-12 < best_val and np.all(np.isfinite(Xp)):
                        best = Xp
                        best_val = vp
                except Exception:
                    pass
    remaining = (t0 + BUDGET) - time.time()
    if remaining > 7.0:
        secs = min(remaining - 2.0, 38.0)
        try:
            polished = _lp_polish(best, secs)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished
                best_val = vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(bad.sum(), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

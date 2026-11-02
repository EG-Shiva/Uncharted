import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _maxcos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _adam_quench(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        # renormalize
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X /= nrm
    return X

def _hard_polish(X0, steps, lr0):
    X = X0.copy()
    n = X.shape[0]
    for t in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        # active set near max
        thr = M - 1e-4
        mask = G > thr
        # weight only active
        W = np.zeros_like(G)
        W[mask] = 1.0
        # symmetrize (G symmetric so mask symmetric)
        grad = W @ X
        # scale
        cnt = np.maximum(np.sum(W, axis=1, keepdims=True), 1.0)
        grad /= cnt
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        nn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt /= nn
        lr = lr0 * (1.0 - 0.8 * t / max(steps - 1, 1))
        X = X - lr * gt
        X = _norm(X)
    return X

def _hole(Y, rng, n_cand=4000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    search_end = t0 + 162.0
    rng = np.random.default_rng(20261101)
    n, d = 55, 21
    best = None
    best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _maxcos(a)
                if s < best_s:
                    best_s = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_s = _maxcos(best)
    base_best = best.copy()

    trial = 0
    # precompute fixed binary pool for algebraic starts
    while time.time() < search_end and trial < 800:
        trial += 1
        kind = trial % 7
        X0 = None
        if kind == 0:
            # pure random
            X0 = _norm(rng.standard_normal((n, d)))
            steps1, a0, a1, lr0, lr1 = 700, 20.0, 400.0, 0.03, 0.004
        elif kind == 1:
            # antipodal symmetric init (27 pairs + 1)
            Z = _norm(rng.standard_normal((27, d)))
            X0 = np.vstack([Z, -Z, _norm(rng.standard_normal((1, d)))])
            steps1, a0, a1, lr0, lr1 = 800, 25.0, 500.0, 0.03, 0.003
        elif kind == 2:
            # binary algebraic init +/-1 then jitter
            B = rng.integers(0, 2, size=(n, d)).astype(np.float64) * 2.0 - 1.0
            B += 0.15 * rng.standard_normal((n, d))
            X0 = _norm(B)
            steps1, a0, a1, lr0, lr1 = 900, 15.0, 450.0, 0.035, 0.004
        elif kind == 3:
            # equi-frame init: random orthogonal blocks
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            X0 = _norm(rng.standard_normal((n, d)) @ Q.T)
            steps1, a0, a1, lr0, lr1 = 700, 30.0, 600.0, 0.025, 0.003
        elif kind == 4:
            # large hole-reinsertion kick of global best
            X0 = best.copy()
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            mx = np.max(G)
            cnt = np.sum(G > mx - 2e-3, axis=1)
            k = 3 + int(rng.integers(0, 4))  # 3-6 victims
            victims = np.argsort(-cnt)[:k]
            if rng.random() < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = X0[keep]
            for vi in victims:
                h = _hole(Y, rng, n_cand=4000)
                X0[vi] = h
                Y = np.vstack([Y, h[None, :]])
            X0 = _norm(X0 + 0.01 * rng.standard_normal(X0.shape))
            steps1, a0, a1, lr0, lr1 = 800, 30.0, 550.0, 0.02, 0.0025
        elif kind == 5:
            # medium shake of global best, different basin attempt
            X0 = _norm(best + (0.08 + 0.12 * rng.random()) * rng.standard_normal((n, d)))
            steps1, a0, a1, lr0, lr1 = 650, 20.0, 500.0, 0.02, 0.003
        else:
            # sparse subspace init: groups on random low-dim subspheres
            X0 = np.zeros((n, d))
            for i in range(n):
                sub = rng.choice(d, size=10, replace=False)
                v = np.zeros(d)
                v[sub] = rng.standard_normal(10)
                X0[i] = v / (np.linalg.norm(v) + 1e-12)
            X0 = _norm(X0 + 0.05 * rng.standard_normal((n, d)))
            steps1, a0, a1, lr0, lr1 = 900, 15.0, 400.0, 0.03, 0.004

        # quench
        try:
            X1 = _adam_quench(X0, steps1, a0, a1, lr0, lr1)
        except Exception:
            continue
        s1 = _maxcos(X1)
        # second stage fine quench if promising
        if s1 < best_s + 0.015:
            try:
                X2 = _adam_quench(X1, 400, 400.0, 1200.0, 0.008, 0.0015)
            except Exception:
                X2 = X1
            s2 = _maxcos(X2)
            # occasional hole refinement
            if s2 < best_s + 0.005 and time.time() < search_end - 1:
                # try 1-hole improvement
                G = X2 @ X2.T
                np.fill_diagonal(G, -2.0)
                mx = np.max(G)
                # worst point
                cnt = np.sum(G > mx - 1e-3, axis=1)
                wi = int(np.argmax(cnt))
                keep = np.ones(n, dtype=bool); keep[wi] = False
                Y = X2[keep]
                h = _hole(Y, rng, n_cand=4000)
                Xh = X2.copy(); Xh[wi] = h
                try:
                    Xh = _adam_quench(Xh, 300, 300.0, 900.0, 0.006, 0.001)
                except Exception:
                    pass
                sh = _maxcos(Xh)
                if sh < s2:
                    X2, s2 = Xh, sh
            if s2 < best_s:
                # hard polish to squeeze
                try:
                    X3 = _hard_polish(X2, 200, 0.002)
                    s3 = _maxcos(X3)
                    if s3 < s2:
                        X2, s2 = X3, s3
                except Exception:
                    pass
                best, best_s = X2.copy(), float(s2)
                base_best = best.copy()
        if time.time() >= search_end:
            break

    # final tiny polish of global best with remaining time
    try:
        if time.time() < deadline - 2:
            Xf = _hard_polish(best, 150, 0.001)
            sf = _maxcos(Xf)
            if sf < best_s:
                best, best_s = Xf, sf
    except Exception:
        pass
    best = _norm(np.asarray(best, dtype=np.float64))
    return best

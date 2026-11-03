import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _quench(X, steps, beta0, beta1, lr0, lr1):
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

def _refine_hole(Y, p, iters=180, beta=90.0, lr0=0.25, lr1=0.005):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        gm = g.max()
        w = np.exp(beta * (g - gm))
        s = w.sum() + 1e-16
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        n = np.linalg.norm(p)
        if n == 0 or not np.isfinite(n):
            break
        p /= n
    return p

def _find_void(Y, rng, n_probe=9000, refine_iters=160, beta=90.0):
    d = Y.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 3000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    idx = int(np.argmin(bs))
    p0 = P[idx]
    p = _refine_hole(Y, p0, iters=refine_iters, beta=beta)
    # check refined vs raw, keep better
    def _worst(q):
        return float((Y @ q).max())
    if _worst(p) > _worst(p0) + 1e-12:
        p = p0
        # try second beta
        p2 = _refine_hole(Y, p0, iters=refine_iters, beta=beta * 2.0)
        if _worst(p2) < _worst(p):
            p = p2
    return p

def _reinsert_voids(X, idx, rng, n_probe=9000):
    Y = X.copy()
    # remove indexed points progressively, refill greedily
    # build keep set
    keep_mask = np.ones(Y.shape[0], dtype=bool)
    keep_mask[np.asarray(idx)] = False
    Ykeep = Y[keep_mask]
    new_pts = []
    for _ in idx:
        Ycur = np.vstack([Ykeep] + new_pts) if new_pts else Ykeep
        beta = float(rng.choice([60.0, 90.0, 130.0]))
        p = _find_void(Ycur, rng, n_probe=n_probe,
                       refine_iters=140 + int(rng.randint(80)),
                       beta=beta)
        new_pts.append(p[None, :])
    Y2 = Y.copy()
    for j, p in zip(idx, new_pts):
        Y2[j] = p[0]
    return Y2

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

def solve():
    t0 = time.time()
    BUDGET = 174.0
    PHASE1 = 132.0
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
        r0 = np.random.RandomState(0)
        best = r0.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    rng = np.random.RandomState(20261103)
    deadline1 = t0 + PHASE1
    it = 0
    # elite pool for diversity
    elites = [(best_val, best.copy())]
    while time.time() < deadline1:
        it += 1
        if time.time() + 2.5 > t0 + BUDGET:
            break
        # pick base: mostly best, sometimes second elite
        if len(elites) > 1 and rng.rand() < 0.25:
            base = elites[rng.randint(len(elites))][1]
        else:
            base = best
        X = base.copy()
        rm = _rowmax(X)
        cmax = rm.max()
        # contact degree with tight tol
        # rattlers: loose points
        order_worst = np.argsort(rm)  # ascending, worst at end
        order_best = order_worst  # same
        u = rng.rand()
        if u < 0.34:
            # rattler / loose reinsertion: replace loosest + random
            k = 2 + int(rng.randint(7))  # 2..8
            # loosest points are smallest rowmax
            nloose = int(np.sum(rm < cmax - 1e-4))
            cand_loose = order_worst[:max(nloose, 10)]
            idx1 = list(rng.choice(cand_loose, min(k // 2 + 1, len(cand_loose)), replace=False)) if len(cand_loose) > 0 else []
            idx2 = list(rng.choice(n, k - len(idx1), replace=False))
            idx = np.unique(np.array(idx1 + idx2, dtype=int))
            # ensure size k
            while len(idx) < k:
                idx = np.unique(np.append(idx, rng.randint(n)))
            idx = idx[:k]
            X = _reinsert_voids(X, idx, rng, n_probe=8000 + int(rng.randint(6000)))
            beta0 = 35.0 + rng.rand() * 30.0
            beta1 = 250.0 + rng.rand() * 450.0
            steps = 1100 + int(rng.randint(900))
            lr0 = 0.003 + rng.rand() * 0.005
            Xq = _quench(X, steps, beta0, beta1, lr0, lr0 * 0.1)
        elif u < 0.62:
            # hub deletion: delete worst jammed points
            k = 1 + int(rng.randint(4))  # 1..4
            # bias to worst rowmax
            if rng.rand() < 0.7:
                idx = order_worst[-k:]
            else:
                idx = rng.choice(n, k, replace=False)
            X = _reinsert_voids(X, idx, rng, n_probe=9000 + int(rng.randint(6000)))
            beta0 = 40.0 + rng.rand() * 20.0
            beta1 = 300.0 + rng.rand() * 400.0
            steps = 1300 + int(rng.randint(900))
            lr0 = 0.004 + rng.rand() * 0.004
            Xq = _quench(X, steps, beta0, beta1, lr0, lr0 * 0.12)
        elif u < 0.85:
            # global Gaussian kick
            sigmas = (0.006, 0.012, 0.025, 0.05)
            sigma = sigmas[rng.randint(len(sigmas))]
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 25.0 + rng.rand() * 35.0
            beta1 = 200.0 + rng.rand() * 500.0
            steps = 1000 + int(rng.randint(1000))
            lr0 = 0.003 + rng.rand() * 0.006
            Xq = _quench(X, steps, beta0, beta1, lr0, lr0 * 0.1)
        else:
            # local cap shake: pick random cap, jitter points inside
            c = rng.randn(d)
            c /= np.linalg.norm(c)
            dots = X @ c
            k = 6 + int(rng.randint(14))
            idx = np.argsort(dots)[-k:]
            sigma = 0.03 + rng.rand() * 0.08
            noise = np.zeros((n, d))
            noise[idx] = rng.randn(k, d) * sigma
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # also reinsert one worst inside cap via void w.r.t outside
            if rng.rand() < 0.5:
                j = idx[rng.randint(k)]
                others = np.delete(X, j, axis=0)
                p = _find_void(others, rng, n_probe=6000, refine_iters=120, beta=80.0)
                # mix: move partway to void to keep topology change moderate
                alpha = 0.5 + rng.rand() * 0.5
                y = X[j] * (1 - alpha) + p * alpha
                y /= np.linalg.norm(y)
                X[j] = y
            beta0 = 40.0 + rng.rand() * 20.0
            beta1 = 350.0 + rng.rand() * 350.0
            steps = 1300 + int(rng.randint(700))
            lr0 = 0.004 + rng.rand() * 0.004
            Xq = _quench(X, steps, beta0, beta1, lr0, lr0 * 0.1)
        if not np.all(np.isfinite(Xq)):
            continue
        vq = _max_cos(Xq)
        if vq + 1e-12 < best_val:
            best = Xq
            best_val = vq
            elites.append((vq, Xq.copy()))
            elites = sorted(elites, key=lambda z: z[0])[:3]
        elif len(elites) < 3 and vq < best_val + 5e-5 and rng.rand() < 0.1:
            elites.append((vq, Xq.copy()))
            elites = sorted(elites, key=lambda z: z[0])[:3]
    remaining = (t0 + BUDGET) - time.time()
    if remaining > 7.0:
        secs = min(remaining - 2.0, 40.0)
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
        rr = np.random.RandomState(1).randn(int(bad.sum()), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

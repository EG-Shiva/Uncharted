import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _adam_quench(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = t / steps
        alpha = a0 * (a1 / a0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Mx = np.max(G)
        E = np.exp(alpha * (G - Mx))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def _hadamard64():
    H = np.array([[1.0]], dtype=np.float64)
    for _ in range(6):
        H = np.block([[H, H], [H, -H]])
    return H  # 64x64

_H64 = None

def _structured_init(kind, rng, n=55, d=21):
    if kind == 0:
        return _norm(rng.standard_normal((n, d)))
    elif kind == 1:
        # antipodal symmetric: 27 + 27 opposites + 1
        a = _norm(rng.standard_normal((27, d)))
        b = -a
        c = _norm(rng.standard_normal((1, d)))
        X = np.vstack([a, b, c])
        # X has 55 rows
        return _norm(X + 0.02 * rng.standard_normal(X.shape))
    elif kind == 2:
        # binary / Hadamard slice: pick 55 rows, 21 cols
        global _H64
        if _H64 is None:
            try:
                _H64 = _hadamard64()
            except Exception:
                return _norm(rng.standard_normal((n, d)))
        rows = rng.choice(64, size=n, replace=False)
        cols = rng.choice(64, size=d, replace=False)
        X = _H64[rows][:, cols].astype(np.float64)
        # random column sign flips to break structure
        X = X * rng.choice([-1.0, 1.0], size=(1, d))
        return _norm(X + 0.15 * rng.standard_normal((n, d)))
    elif kind == 3:
        # triangular / pair-sum: 11 bases -> 55 pair sums
        Y = _norm(rng.standard_normal((11, d)))
        X = np.zeros((n, d))
        k = 0
        for i in range(11):
            for j in range(i + 1, 11):
                X[k] = Y[i] + Y[j]
                k += 1
        return _norm(X + 0.1 * rng.standard_normal(X.shape))
    elif kind == 4:
        # two orthoplex-biased blocks
        A = _norm(rng.standard_normal((28, d)))
        B = _norm(rng.standard_normal((27, d)))
        # orthogonalize blocks slightly apart
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        A = A @ Q.T
        B = B @ Q.T + 0.3 * rng.standard_normal((27, d))
        return _norm(np.vstack([A, B]))
    else:
        # simplex-anchored: regular simplex 22 vertices + 33 random
        # simplex in R21: take 22 orthonormal-ish
        Z = rng.standard_normal((22, d))
        # center
        Z = Z - Z.mean(axis=0, keepdims=True)
        Z = _norm(Z)
        R = _norm(rng.standard_normal((33, d)))
        return _norm(np.vstack([Z, R, _norm(rng.standard_normal((0, d)))])[:n])

def _lp_polish(x, deadline, max_it=60):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and it < max_it and radius > 1e-14:
        it += 1
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1300:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1300]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
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
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _norm(y)
        vv = _mc(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_SEARCH = t0 + 128.0
    T_LP = t0 + 168.0
    rng = np.random.default_rng(20261101)
    n, d = 55, 21
    # load fallback
    base = None
    base_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _mc(a)
                if s < base_score:
                    base_score = s
                    base = a.copy()
        except Exception:
            pass
    if base is None:
        base = _norm(rng.standard_normal((n, d)))
        base_score = _mc(base)
    best_overall = base.copy()
    best_score = base_score
    # keep top quench candidates
    cands = []  # (score, X)
    trial = 0
    # hyper sets to diversify
    hypers = [
        (1200, 40.0, 600.0, 0.020, 0.002),
        (1000, 25.0, 400.0, 0.025, 0.003),
        (1400, 60.0, 900.0, 0.015, 0.0015),
        (800, 20.0, 350.0, 0.030, 0.004),
        (1600, 50.0, 800.0, 0.018, 0.001),
    ]
    while time.time() < T_SEARCH:
        kind = trial % 6
        hp = hypers[trial % len(hypers)]
        steps, a0, a1, lr0, lr1 = hp
        # shorten steps if little time
        if T_SEARCH - time.time() < 10:
            steps = min(steps, 600)
        X0 = _structured_init(kind, rng, n, d)
        # extra large shake for kind 5: perturb base strongly to leave basin
        if kind == 5:
            sigma = 0.25 + 0.15 * rng.random()
            X0 = _norm(base + sigma * rng.standard_normal((n, d)))
        try:
            Xq = _adam_quench(X0, steps, a0, a1, lr0, lr1)
        except Exception:
            trial += 1
            continue
        s = _mc(Xq)
        # keep top 4
        cands.append((s, Xq.copy()))
        cands.sort(key=lambda z: z[0])
        cands = cands[:4]
        if s < best_score:
            best_score = s
            best_overall = Xq.copy()
        trial += 1
        if trial > 300:
            break
    # LP polish: polish candidates in order, then base
    to_polish = []
    for s, X in cands:
        # only polish promising ones (within 0.01 of best) to save time
        if s < best_score + 0.008:
            to_polish.append(X)
    # always include current best overall
    # deduplicate: ensure best_overall included
    # limit to 3 polishes
    to_polish = to_polish[:3]
    for X in to_polish:
        if time.time() > T_LP - 8:
            break
        # give each ~ remaining/ (left+1) but at most 25s
        remain = T_LP - time.time()
        left = len(to_polish)
        budget_end = time.time() + min(25.0, remain / max(left, 1))
        # quick re-quench with hot alpha before LP for extra squeeze
        try:
            Xh = _adam_quench(X, 300, 200.0, 1200.0, 0.006, 0.001)
        except Exception:
            Xh = X
        if _mc(Xh) < _mc(X):
            X = Xh
        Xp = _lp_polish(X, budget_end)
        sp = _mc(Xp)
        if sp < best_score:
            best_score = sp
            best_overall = Xp.copy()
    # ensure we never regress vs loaded base
    if best_score > base_score:
        return base
    return _norm(best_overall)

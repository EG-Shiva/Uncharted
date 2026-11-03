import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _max_abs(Y):
    G = Y @ Y.T
    A = np.abs(G)
    np.fill_diagonal(A, -2.0)
    return float(np.max(A))

def _pair_init(X):
    n = X.shape[0]
    m = n // 2
    G = X @ X.T
    unused = np.ones(n, dtype=bool)
    Y = np.zeros((m, X.shape[1]))
    k = 0
    for i in range(n):
        if not unused[i]:
            continue
        # find unused j != i with minimal inner product
        best_j = -1
        best_v = 2.0
        for j in range(n):
            if unused[j] and j != i:
                v = G[i, j]
                if v < best_v:
                    best_v = v
                    best_j = j
        if best_j == -1:
            # odd leftover (should not happen for 184)
            Y[k] = X[i]
            k += 1
            unused[i] = False
        else:
            d = X[i] - X[best_j]
            nrm = np.linalg.norm(d)
            if nrm < 1e-12:
                d = X[i].copy()
                nrm = np.linalg.norm(d)
            Y[k] = d / nrm
            k += 1
            unused[i] = False
            unused[best_j] = False
    return _norm(Y)

def _smooth_sym(Y, steps, a0, a1, lr0, lr1):
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)  # sign irrelevant, weight zeroed later
        A = np.abs(G)
        np.fill_diagonal(A, -np.inf)
        m = float(np.max(A))
        W = np.exp(alpha * (A - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        Sg = np.sign(G)
        np.fill_diagonal(Sg, 0.0)
        grad = (W * Sg) @ Y
        dot = np.sum(grad * Y, axis=1, keepdims=True)
        gt = grad - dot * Y
        Y = Y - lr * gt / S
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    return Y

def _greedy_sym(Y, steps, m0, m1, lr):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = Y @ Y.T
        A = np.abs(G)
        np.fill_diagonal(A, -2.0)
        m = float(np.max(A))
        M = (A > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        Sg = np.sign(G) * M
        deg = M.sum(axis=1, keepdims=True)
        grad = Sg @ Y
        dot = np.sum(grad * Y, axis=1, keepdims=True)
        gt = grad - dot * Y
        Y = Y - lr * gt / np.maximum(deg, 1.0)
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    return Y

def _smooth_full(x, steps, a0, a1, lr0, lr1):
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = float(np.max(G))
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / S
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

def _greedy_full(x, steps, m0, m1, lr):
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / np.maximum(deg, 1.0)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.2e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t0 < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2200:
            idx = np.argpartition(-cs, 2200)[:2200]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2200
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-7:
                break
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(0)
    n, d = 184, 8
    m = 92
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _norm(rng.standard_normal(size=(n, d)))
        gval = _max_cos(gbest)
    gbest = np.ascontiguousarray(gbest)

    # build Y initializations
    Y_inits = []
    for _, Xsrc in cands:
        try:
            Y_inits.append(_pair_init(Xsrc))
        except Exception:
            pass
    # random inits
    for _ in range(4):
        Y_inits.append(_norm(rng.standard_normal(size=(m, d))))

    best_sym_val = np.inf
    best_sym = None

    # symmetric annealing schedules (diversify)
    scheds = [
        (900, 25.0, 380.0, 0.05, 0.003),
        (900, 35.0, 450.0, 0.04, 0.0025),
        (1000, 20.0, 350.0, 0.06, 0.003),
        (1000, 30.0, 500.0, 0.045, 0.002),
        (800, 25.0, 400.0, 0.05, 0.003),
        (800, 40.0, 420.0, 0.04, 0.0025),
    ]

    trial = 0
    for yi, Y0 in enumerate(Y_inits):
        if time.time() > DEADLINE - 55.0:
            break
        trial += 1
        Y = Y0.copy()
        # small jitter for greedy inits to break exact degeneracy
        if yi < len(cands):
            Y = _norm(Y + rng.standard_normal(size=Y.shape) * 0.002)
        sc = scheds[yi % len(scheds)]
        steps, a0, a1, lr0, lr1 = sc
        Y = _smooth_sym(Y, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
        Y = _greedy_sym(Y, steps=120, m0=0.008, m1=0.0008, lr=0.007)
        sv = _max_abs(Y)
        if sv < best_sym_val:
            best_sym_val = sv
            best_sym = Y.copy()
        # duplicate antipodally, release constraint, refine full
        X = np.vstack([Y, -Y])
        # break symmetry
        X = _norm(X + rng.standard_normal(size=X.shape) * 0.004)
        X = _smooth_full(X, steps=280, a0=45.0, a1=340.0, lr0=0.028, lr1=0.0028)
        X = _greedy_full(X, steps=110, m0=0.007, m1=0.0007, lr=0.007)
        v = _max_cos(X)
        if v < gval:
            gval = float(v)
            gbest = X.copy()

    # extra random symmetric trials if time permits (different basin)
    extra = 0
    while time.time() < DEADLINE - 60.0 and time.time() - t0 < 110.0 and extra < 6:
        extra += 1
        trial += 1
        if best_sym is not None and (extra % 2 == 0):
            Y = _norm(best_sym + rng.standard_normal(size=best_sym.shape) * 0.05)
        else:
            Y = _norm(rng.standard_normal(size=(m, d)))
        a1 = 350.0 + 40.0 * (extra % 3)
        Y = _smooth_sym(Y, steps=700, a0=30.0, a1=a1, lr0=0.05, lr1=0.0025)
        Y = _greedy_sym(Y, steps=100, m0=0.007, m1=0.0008, lr=0.007)
        X = np.vstack([Y, -Y])
        X = _norm(X + rng.standard_normal(size=X.shape) * 0.005)
        X = _smooth_full(X, steps=240, a0=50.0, a1=320.0, lr0=0.026, lr1=0.0025)
        X = _greedy_full(X, steps=100, m0=0.006, m1=0.0007, lr=0.007)
        v = _max_cos(X)
        if v < gval:
            gval = float(v)
            gbest = X.copy()

    remain = DEADLINE - time.time() - 1.0
    if remain > 10:
        gbest = _lp_polish(gbest, min(remain, 32.0), DEADLINE - 0.5)
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        try:
            gbest = _norm(np.asarray(np.load("best.npy"), dtype=np.float64))
        except Exception:
            pass
    return gbest

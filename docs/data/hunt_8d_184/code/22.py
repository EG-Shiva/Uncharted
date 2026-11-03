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

def _max_abs_off(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))

def _pair_to_Y(X):
    n, d = X.shape
    m = n // 2
    G = X @ X.T
    unpaired = set(range(n))
    Y = []
    while unpaired:
        i = min(unpaired)
        best_j = None
        best_v = 1e9
        for j in unpaired:
            if j == i:
                continue
            v = G[i, j]
            if v < best_v:
                best_v = v
                best_j = j
        if best_j is None:
            # odd leftover
            Y.append(X[i].copy())
            unpaired.remove(i)
            break
        v = X[i] - X[best_j]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = X[i].copy()
            nv = np.linalg.norm(v)
        Y.append(v / nv)
        unpaired.remove(i)
        unpaired.remove(best_j)
    Y = np.array(Y, dtype=np.float64)
    if len(Y) < m:
        # pad randomly (should not happen for 184)
        rng = np.random.default_rng(0)
        extra = _norm(rng.standard_normal(size=(m - len(Y), d)))
        Y = np.vstack([Y, extra])
    return Y[:m]

def _anneal_abs(Y, steps, a0, a1, lr0, lr1):
    Y = Y.copy()
    n, d = Y.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = Y @ Y.T
        A = np.abs(G)
        np.fill_diagonal(A, -np.inf)
        mx = float(np.max(A))
        W = np.exp(alpha * (A - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        Sg = np.sign(G)
        # (W * sign) @ Y
        WS = W * Sg
        grad = WS @ Y
        dot = np.sum(grad * Y, axis=1, keepdims=True)
        gt = grad - dot * Y
        Y = Y - lr * gt / S
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
    return Y

def _smooth_full(x, steps, a0, a1, lr0, lr1):
    n, d = x.shape
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
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
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
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
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
        if p > 2000:
            idx = np.argpartition(-cs, 2000)[:2000]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2000
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
        if time.time() > deadline - 0.5:
            break
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    n, d = 184, 8
    m = 92
    rng = np.random.default_rng(12345)

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

    # ---- symmetric subproblem: build Y inits ----
    Y_inits = []
    # greedy pairing from loaded codes
    for _, Xc in cands[:2]:
        try:
            Y_inits.append(_pair_to_Y(Xc))
        except Exception:
            pass
    # random inits
    for s in (11, 22, 33, 44):
        r = np.random.default_rng(s)
        Y_inits.append(_norm(r.standard_normal(size=(m, d))))

    best_Y = None
    best_Y_val = 1e9
    # anneal each Y init with increasing alpha schedule
    for Yi in Y_inits:
        if time.time() > DEADLINE - 60:
            break
        Y = Yi.copy()
        # small random kick to break exact symmetry of pairing
        Y = _norm(Y + rng.standard_normal(size=Y.shape) * 0.004)
        Y = _anneal_abs(Y, steps=700, a0=25.0, a1=280.0, lr0=0.06, lr1=0.008)
        if time.time() > DEADLINE - 55:
            # skip fine stage if low on time
            v = _max_abs_off(Y)
            if v < best_Y_val:
                best_Y_val = v; best_Y = Y.copy()
            continue
        Y = _anneal_abs(Y, steps=500, a0=280.0, a1=700.0, lr0=0.012, lr1=0.003)
        v = _max_abs_off(Y)
        if v < best_Y_val:
            best_Y_val = v; best_Y = Y.copy()

    if best_Y is None:
        best_Y = _norm(rng.standard_normal(size=(m, d)))
        best_Y_val = _max_abs_off(best_Y)

    # duplicate antipodally
    Xsym = np.vstack([best_Y, -best_Y])
    Xsym = _norm(Xsym)
    vsym = _max_cos(Xsym)

    # release constraint: refine symmetric point with full-code annealing
    # try a couple of jitter levels, keep best
    rel_best = None
    rel_val = 1e9
    # include exact symmetric as start
    starts = []
    starts.append((Xsym.copy(), 300, 45.0, 350.0, 0.030, 0.003))
    # jittered versions
    for sig in (0.004, 0.010):
        Xs = _norm(Xsym + rng.standard_normal(size=Xsym.shape) * sig)
        starts.append((Xs, 300, 45.0, 350.0, 0.030, 0.003))
    for Xs0, st, a0, a1, l0, l1 in starts:
        if time.time() > DEADLINE - 45:
            break
        Xs = Xs0.copy()
        Xs = _smooth_full(Xs, steps=st, a0=a0, a1=a1, lr0=l0, lr1=l1)
        Xs = _greedy_full(Xs, steps=120, m0=0.007, m1=0.0007, lr=0.007)
        v = _max_cos(Xs)
        if v < rel_val:
            rel_val = v; rel_best = Xs.copy()

    if rel_best is not None and rel_val < gval:
        gval = rel_val; gbest = rel_best.copy()
    # if released symmetric is worse than record, still keep it as wander base
    wander = rel_best.copy() if rel_best is not None else Xsym.copy()
    wander_val = rel_val if rel_best is not None else vsym

    # short improvement loop from new basin (bounded time)
    it = 0
    while time.time() < DEADLINE - 38.0 and it < 12:
        it += 1
        if time.time() - t0 > 130:
            break
        # alternate base between gbest and wander
        base = wander if (it % 3 != 0) else gbest
        Xt = base.copy()
        # small tangential/global noise (breaks symmetry further)
        sig = 0.006 + 0.004 * ((it % 3) / 2.0)
        Xt = _norm(Xt + rng.standard_normal(size=Xt.shape) * sig)
        Xt = _smooth_full(Xt, steps=260, a0=50.0, a1=320.0, lr0=0.028, lr1=0.0028)
        Xt = _greedy_full(Xt, steps=100, m0=0.006, m1=0.0007, lr=0.007)
        v = _max_cos(Xt)
        if v < wander_val:
            wander_val = v; wander = Xt.copy()
        if v < gval:
            gval = v; gbest = Xt.copy()

    remain = DEADLINE - time.time() - 1.0
    if remain > 10:
        gbest = _lp_polish(gbest, min(remain, 30.0), DEADLINE - 0.5)

    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(np.asarray(np.load("best.npy"), dtype=np.float64))
    return gbest

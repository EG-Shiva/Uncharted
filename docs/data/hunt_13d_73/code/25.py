import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _cheap_adam(X0, steps=350, s0=25.0, s1=160.0, lr0=0.02, lr1=0.004):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    n, d = X.shape
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = float(np.exp(ls0 + (ls1 - ls0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        E = np.exp(s * (G - cmax))
        np.fill_diagonal(E, 0.0)
        sumE = float(E.sum() * 0.5)
        if not np.isfinite(sumE) or sumE == 0:
            continue
        W = E / sumE
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X = _normalize(X)
    return X

def _smooth_lbfgs(X0, alphas=(80.0, 250.0), maxiter_each=100, t_end=1e30):
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Y = X0.ravel().copy()
    for alpha in alphas:
        if time.time() > t_end - 1.0:
            break
        def fun(Yflat):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            dots = G[iu0]
            M = float(dots.max())
            e = np.exp(alpha * (dots - M))
            S = float(e.sum())
            loss = M + np.log(S) / alpha
            p = e / S
            W = np.zeros((n, n))
            W[iu0] = p
            W = W + W.T
            grad_X = W @ X
            dg = (grad_X * X).sum(axis=1, keepdims=True)
            grad_Xt = grad_X - dg * X
            grad_Y = (grad_Xt / rn).ravel()
            return float(loss), grad_Y.astype(float)
        try:
            res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter_each, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 10})
            Y = res.x
        except Exception:
            break
        if time.time() > t_end - 1.0:
            break
    Ym = Y.reshape(n, d)
    return _normalize(Ym)

def _lp_polish(X, seconds, t_end=1e30):
    X = _normalize(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _build_pool13():
    P = []
    for a in range(13):
        for b in range(a + 1, 13):
            for sa in (1.0, -1.0):
                for sb in (1.0, -1.0):
                    v = np.zeros(13)
                    v[a] = sa / np.sqrt(2.0)
                    v[b] = sb / np.sqrt(2.0)
                    P.append(v)
    return np.array(P)

def _greedy_from_pool(pool, start_idx, jitter_rng):
    npool = pool.shape[0]
    Gpool = pool @ pool.T
    chosen = [int(start_idx % npool)]
    used = np.zeros(npool, dtype=bool)
    used[chosen[0]] = True
    jit = jitter_rng.random(npool) * 1e-7
    for _ in range(1, 73):
        md = Gpool[:, chosen].max(axis=1) + jit
        md[used] = np.inf
        b = int(np.argmin(md))
        chosen.append(b)
        used[b] = True
    return pool[np.array(chosen)].copy()

def _cyclic_seed(rng):
    bases = rng.normal(size=(5, 13))
    bases = _normalize(bases)
    pts = []
    for v in bases:
        for s in range(13):
            pts.append(np.roll(v, s))
    pts = np.array(pts)
    pts = _normalize(pts)
    exist = pts
    for _ in range(8):
        C = rng.normal(size=(2000, 13))
        C = _normalize(C)
        dots = C @ exist.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        exist = np.vstack([exist, C[bi:bi+1]])
    return _normalize(exist)

def solve():
    t0 = time.time()
    HARD_END = t0 + 165.0
    n, d = 73, 13
    # incumbent
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = None
    if Xb is None or Xb.shape != (n, d):
        rng0 = np.random.default_rng(0)
        Xb = rng0.normal(size=(n, d))
    Xb = _normalize(np.asarray(Xb, dtype=float))
    # guard nan
    if not np.all(np.isfinite(Xb)):
        rng0 = np.random.default_rng(0)
        Xb = _normalize(rng0.normal(size=(n, d)))
    best_val = _max_cos(Xb)
    Xbest = Xb.copy()

    rng = np.random.default_rng(20261102)
    seeds = []  # (family, X0)

    # (a) 30 purely random
    for _ in range(30):
        seeds.append(("rand", _normalize(rng.normal(size=(n, d)))))

    # (b) greedy maximin from algebraic pool
    try:
        pool = _build_pool13()
        starts = [0, 47, 101, 159, 213, 277]
        for si in starts:
            jr = np.random.default_rng(1000 + si)
            seeds.append(("alg", _greedy_from_pool(pool, si, jr)))
    except Exception:
        for k in range(6):
            seeds.append(("alg", _normalize(rng.normal(size=(n, d)))))

    # (c) cyclic-orbit seeds (5 generic vectors -> 65 + 8 holes)
    for k in range(5):
        try:
            cr = np.random.default_rng(5000 + k)
            seeds.append(("cyc", _cyclic_seed(cr)))
        except Exception:
            seeds.append(("cyc", _normalize(rng.normal(size=(n, d)))))

    relaxed = []  # (val, family, X)
    # cheap relaxation, at most ~2s each but actually ~0.05s; enforce global clock
    CHEAP_END = t0 + 105.0
    for fam, X0 in seeds:
        if time.time() > CHEAP_END or time.time() > HARD_END - 45.0:
            break
        try:
            Xr = _cheap_adam(X0, steps=320, s0=25.0, s1=160.0, lr0=0.02, lr1=0.004)
        except Exception:
            Xr = X0
        Xr = _normalize(Xr)
        try:
            v = _max_cos(Xr)
        except Exception:
            continue
        relaxed.append((v, fam, Xr))
        if v < best_val and np.all(np.isfinite(Xr)):
            best_val = v
            Xbest = Xr.copy()

    # rank + diversity: best per family
    if len(relaxed) == 0:
        return Xbest
    relaxed.sort(key=lambda t: t[0])
    best_per_fam = {}
    for v, fam, Xr in relaxed:
        if fam not in best_per_fam:
            best_per_fam[fam] = (v, fam, Xr)
    # finalists: best per family sorted (at most 3)
    finalists = sorted(best_per_fam.values(), key=lambda t: t[0])[:3]
    # if fewer than 3 (should be 3), pad with overall best
    if len(finalists) < 3:
        have = set(id(t) for t in finalists)
        for t in relaxed:
            if len(finalists) >= 3:
                break
            finalists.append(t)

    # order finalists by value
    finalists = sorted(finalists, key=lambda t: t[0])
    for v, fam, Xr in finalists:
        if time.time() > HARD_END - 6.0:
            break
        # L-BFGS refinement only if enough time (>= ~18s for LP after)
        rem = HARD_END - time.time()
        if rem < 20.0:
            break
        try:
            t_end_lb = HARD_END - 16.0
            X2 = _smooth_lbfgs(Xr, alphas=(80.0, 250.0), maxiter_each=90, t_end=t_end_lb)
            X2 = _normalize(X2)
            v2 = _max_cos(X2)
        except Exception:
            X2, v2 = Xr, v
        if np.all(np.isfinite(X2)) and v2 < best_val:
            best_val = v2
            Xbest = X2.copy()
            Xr, v = X2, v2
        # LP polish at most 15s each, only if enough time remains
        rem2 = HARD_END - time.time()
        if rem2 < 8.0:
            continue
        lp_sec = min(15.0, rem2 - 2.0)
        if lp_sec < 2.0:
            continue
        try:
            X3 = _lp_polish(Xr, seconds=lp_sec, t_end=HARD_END - 1.0)
            X3 = _normalize(X3)
            v3 = _max_cos(X3)
        except Exception:
            continue
        if np.all(np.isfinite(X3)) and v3 < best_val:
            best_val = v3
            Xbest = X3.copy()

    return _normalize(Xbest)

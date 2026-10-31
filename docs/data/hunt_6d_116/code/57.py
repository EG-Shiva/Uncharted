import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _softmax_adam(X, steps, T0, T1, lr0, lr1, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    b1p, b2p = 1.0, 1.0
    lT0 = np.log(T0) if T0 > 0 else 0.0
    lT1 = np.log(T1) if T1 > 0 else 0.0
    for it in range(1, steps + 1):
        if (it & 31) == 0 and time.time() > deadline:
            break
        f = it / float(steps)
        T = float(np.exp(lT0 + (lT1 - lT0) * f)) if T0 > 0 and T1 > 0 else T0
        if T < 1e-12:
            T = 1e-12
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        # avoid overflow
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Gr = W @ X
        proj = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - proj * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = v / (1.0 - b2p)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _refine_void(y, others, steps=40, step0=0.08):
    y = y / (np.linalg.norm(y) + 1e-15)
    step = step0
    for _ in range(steps):
        dd = others @ y
        j = int(np.argmax(dd))
        g = others[j] - float(others[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        ny = float(np.linalg.norm(y))
        if ny < 1e-12:
            break
        y /= ny
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _deepest_hole(fixed, rng, ntry=20000):
    d = fixed.shape[1]
    # chunk to save memory
    best_y = None
    best_w = 1e100
    chunk = 5000
    for s in range(0, ntry, chunk):
        cc = min(chunk, ntry - s)
        C = rng.standard_normal((cc, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-15)
        dots = C @ fixed.T
        w = dots.max(axis=1)
        bi = int(np.argmin(w))
        if float(w[bi]) < best_w:
            best_w = float(w[bi])
            best_y = C[bi].copy()
    y = _refine_void(best_y, fixed, steps=40)
    return y

def _d6_seed(n, rng, jitter=0.15):
    d = 6
    pts = []
    seen = set()
    # gather lattice shell directions
    tries = 0
    while len(pts) < n * 3 and tries < 20000:
        tries += 1
        v = rng.integers(-2, 3, size=d)
        if int(v.sum()) % 2 != 0:
            continue
        if np.all(v == 0):
            continue
        key = tuple(v.tolist())
        if key in seen:
            continue
        seen.add(key)
        pts.append(v.astype(float))
    pts = np.array(pts, dtype=float) if len(pts) > 0 else rng.standard_normal((n * 3, d))
    # normalize + jitter, dedup roughly
    P = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-15)
    P = P + jitter * rng.standard_normal(P.shape)
    P = _norm(P)
    if len(P) >= n:
        idx = rng.choice(len(P), size=n, replace=False)
        return _norm(P[idx])
    C = rng.standard_normal((n, d))
    return _norm(C)

def _gram_spec(X):
    try:
        G = X @ X.T
        w = np.linalg.eigvalsh(G)
        w = np.sort(w)
        return w
    except Exception:
        return None

def _pool_add(pool, X, val):
    s = _gram_spec(X)
    for (v2, X2, s2) in pool:
        if s is not None and s2 is not None and s.shape == s2.shape:
            try:
                if float(np.linalg.norm(s - s2)) < 1e-6:
                    return False
            except Exception:
                pass
    pool.append((val, X.copy(), s))
    return True

def _prune_victim(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # weight tightness with narrow kernel
    W = np.exp((G - m) / 0.005)
    np.fill_diagonal(W, 0.0)
    load = W.sum(axis=1)
    return int(np.argmax(load))

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        mm = float(c.max())
        thr = mm - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1200:
            o = np.argsort(c[idx])[-1200:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 150.0
    EXPL = t0 + 130.0
    FINAL_END = t0 + 168.0
    rng = np.random.default_rng(2026116)

    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nrm = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(nrm < 1e-12):
            return None
        return a / nrm

    incumbent = None
    inc_val = 1e100
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                v = _maxcos(c)
                if v < inc_val:
                    inc_val = v
                    incumbent = c.copy()
        except Exception:
            pass
    if incumbent is None:
        incumbent = _norm(rng.standard_normal((116, 6)))
        inc_val = _maxcos(incumbent)

    pool = []

    def quench_wrap(X, steps, T0, T1, lr0, lr1):
        return _softmax_adam(X, steps, T0, T1, lr0, lr1, HARD)

    def grow_chain(Ns, seedX, label):
        X = seedX
        for k, N in enumerate(Ns):
            if time.time() > EXPL:
                break
            cur_n = X.shape[0]
            if cur_n < N:
                # insert one by one
                for _ in range(N - cur_n):
                    if time.time() > EXPL:
                        break
                    try:
                        y = _deepest_hole(X, rng, ntry=20000)
                    except Exception:
                        y = rng.standard_normal(6)
                        y /= (np.linalg.norm(y) + 1e-15)
                    X = np.vstack([X, y[None, :]])
                    # brief re-quench after each insertion
                    try:
                        X = _softmax_adam(X, 80, 0.006, 0.0008, 0.05, 0.008, HARD)
                    except Exception:
                        pass
                    if time.time() > EXPL:
                        break
            # stage quench
            try:
                steps = 450 if k == 0 else 300
                X = _softmax_adam(X, steps, 0.008, 0.0006, 0.06, 0.006, HARD)
            except Exception:
                pass
            if time.time() > EXPL:
                break
        return X

    def prune_chain(seedX, start_n, target_n=116):
        X = seedX
        try:
            X = _softmax_adam(X, 400, 0.008, 0.0006, 0.06, 0.006, HARD)
        except Exception:
            pass
        while X.shape[0] > target_n and time.time() < EXPL:
            try:
                vic = _prune_victim(X)
            except Exception:
                vic = int(rng.integers(0, X.shape[0]))
            mask = np.ones(X.shape[0], dtype=bool)
            mask[vic] = False
            X = X[mask]
            try:
                X = _softmax_adam(X, 60, 0.005, 0.0008, 0.04, 0.008, HARD)
            except Exception:
                pass
        return X

    # Build de-novo starts (never from record)
    try:
        # Chain A: random 96 -> 104 -> 112 -> 116
        if time.time() < EXPL:
            s96 = _norm(rng.standard_normal((96, 6)))
            XA = grow_chain([96, 104, 112, 116], s96, "A")
            if XA.shape[0] == 116:
                try:
                    XA = _softmax_adam(XA, 250, 0.004, 0.0004, 0.04, 0.005, HARD)
                except Exception:
                    pass
                vA = _maxcos(XA)
                _pool_add(pool, XA, vA)
    except Exception:
        pass
    try:
        if time.time() < EXPL:
            s96b = _d6_seed(96, rng, jitter=0.12)
            XB = grow_chain([96, 104, 112, 116], s96b, "B")
            if XB.shape[0] == 116:
                try:
                    XB = _softmax_adam(XB, 250, 0.005, 0.0005, 0.045, 0.005, HARD)
                except Exception:
                    pass
                vB = _maxcos(XB)
                _pool_add(pool, XB, vB)
    except Exception:
        pass
    try:
        # Chain C: D6-seeded 104 -> 112 -> 116 (different basin)
        if time.time() < EXPL:
            s104 = _d6_seed(104, rng, jitter=0.25)
            try:
                s104 = _softmax_adam(s104, 350, 0.009, 0.0007, 0.06, 0.006, HARD)
            except Exception:
                pass
            XC = grow_chain([104, 112, 116], s104, "C")
            if XC.shape[0] == 116:
                try:
                    XC = _softmax_adam(XC, 200, 0.004, 0.0004, 0.04, 0.005, HARD)
                except Exception:
                    pass
                vC = _maxcos(XC)
                _pool_add(pool, XC, vC)
    except Exception:
        pass
    try:
        # Reverse path: 124 -> prune to 116
        if time.time() < EXPL:
            s124 = _norm(rng.standard_normal((124, 6)))
            XD = prune_chain(s124, 124, 116)
            if XD.shape[0] == 116:
                try:
                    XD = _softmax_adam(XD, 220, 0.004, 0.0004, 0.04, 0.005, HARD)
                except Exception:
                    pass
                vD = _maxcos(XD)
                _pool_add(pool, XD, vD)
    except Exception:
        pass

    # pick best distinct end-point
    best_pool_X = None
    best_pool_v = 1e100
    for (v, X, s) in pool:
        if v < best_pool_v:
            best_pool_v = v
            best_pool_X = X.copy()

    # LP-polish only the best distinct end-point for final ~20s
    if best_pool_X is not None and time.time() < FINAL_END - 1.0:
        try:
            remain = min(FINAL_END - time.time(), 20.0)
            if remain > 1.0:
                bx, bv = _lp_polish(best_pool_X, time.time() + remain)
                # update pool entry
                best_pool_X, best_pool_v = bx.copy(), float(bv)
        except Exception:
            pass

    # return best of pool / incumbent
    if best_pool_X is not None and best_pool_v < inc_val:
        return _norm(best_pool_X)
    return _norm(incumbent)

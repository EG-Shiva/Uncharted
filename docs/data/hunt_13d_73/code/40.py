import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(X0, s0, s1, steps, lr0, lr1, rng=None, noise=0.0):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros((n, d))
    v = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    iu0, iu1 = np.triu_indices(n, 1)
    K = len(iu0)
    l0, l1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = s0 + (s1 - s0) * frac
        lr = np.exp(l0 + (l1 - l0) * frac)
        G = X @ X.T
        c = G[iu0, iu1]
        mx = float(c.max())
        e = np.exp(s * (c - mx))
        se = float(e.sum())
        if not np.isfinite(se) or se <= 0:
            w = np.ones(K) / K
        else:
            w = e / se
        # GradX = W @ X without forming full matrix
        # GradX[i] = sum_j w_ij * X[j]
        WX = np.zeros((n, d))
        # accumulate: for each pair (i,j) add w*X[j] to i and w*X[i] to j
        # vectorized via bincount-like: use np.add.at
        np.add.at(WX, iu0, w[:, None] * X[iu1])
        np.add.at(WX, iu1, w[:, None] * X[iu0])
        sprod = np.sum(WX * X, axis=1)
        Gt = WX - sprod[:, None] * X
        # Adam
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        # clip step norm to avoid explosion
        X = X - lr * step
        if noise > 0 and rng is not None:
            X = X + rng.standard_normal(size=(n, d)) * (noise * lr)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-7)
        if np.count_nonzero(sel) > 2000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    rng = np.random.default_rng(0)
    n, d = 73, 13
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn)
            vv = np.asarray(vv, dtype=np.float64)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                if np.all(np.isfinite(vv)):
                    cc = _max_cos(vv)
                    if cc < best_file:
                        best_file = cc
                        Xrec = vv.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)

    # Stage A: annealed LSE Adam from incumbent
    try:
        Xa = _adam_lse(Xbest, 80.0, 300.0, 1200, 0.02, 0.002)
        if time.time() < T_END - 20:
            Xa = _adam_lse(Xa, 300.0, 800.0, 1200, 0.01, 0.0015)
        va = _max_cos(Xa)
        if va < best and np.all(np.isfinite(Xa)):
            best = va
            Xbest = Xa.copy()
    except Exception:
        pass

    # Stage B: tiny shake to explore neighboring balancing basin
    if time.time() < T_END - 30:
        try:
            X0 = Xrec + rng.standard_normal(size=(n, d)) * 0.004
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xb = _adam_lse(X0, 100.0, 400.0, 1000, 0.02, 0.002)
            if time.time() < T_END - 20:
                Xb = _adam_lse(Xb, 400.0, 1000.0, 800, 0.008, 0.001)
            vb = _max_cos(Xb)
            if vb < best and np.all(np.isfinite(Xb)):
                best = vb
                Xbest = Xb.copy()
        except Exception:
            pass

    # Stage C: worst-pair targeted micro-descent + rebalance
    if time.time() < T_END - 25:
        try:
            Xc = Xbest.copy()
            for _ in range(2):
                G = Xc @ Xc.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                ai, bi = flat // n, flat % n
                # push worst pair apart tangentially
                for __ in range(60):
                    xa, xb = Xc[ai], Xc[bi]
                    ga = xb - float(xb @ xa) * xa
                    gb = xa - float(xa @ xb) * xb
                    xa = xa - 0.01 * ga
                    xb = xb - 0.01 * gb
                    xa /= np.linalg.norm(xa)
                    xb /= np.linalg.norm(xb)
                    Xc[ai], Xc[bi] = xa, xb
                Xc = _adam_lse(Xc, 300.0, 700.0, 500, 0.008, 0.001)
                vc = _max_cos(Xc)
                if vc < best and np.all(np.isfinite(Xc)):
                    best = vc
                    Xbest = Xc.copy()
                else:
                    break
                if time.time() > T_END - 25:
                    break
        except Exception:
            pass

    # Final short LP squeeze
    if time.time() < T_END - 3.0:
        try:
            remain = T_END - time.time() - 1.0
            if remain > 2.0:
                Xp, vp = _lp_polish(Xbest, seconds=min(remain, 18.0), t_end=T_END - 0.5)
                if vp < best and np.all(np.isfinite(Xp)):
                    best = vp
                    Xbest = Xp.copy()
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xbest)):
        Xbest = Xrec / np.linalg.norm(Xrec, axis=1, keepdims=True)
    return Xbest

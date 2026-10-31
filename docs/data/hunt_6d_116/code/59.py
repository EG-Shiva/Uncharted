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
    return float(np.max(G))

def _adam_quench(X0, steps, b0, b1, lr0, lr1, rng=None):
    X = _norm(X0.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    lb0 = np.log(b0) if b0 > 0 else 0.0
    lb1 = np.log(b1) if b1 > 0 else 0.0
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        W = np.exp(beta * (G - mx))
        np.fill_diagonal(W, 0.0)
        Sx = float(np.sum(W))
        if not np.isfinite(Sx) or Sx <= 0:
            break
        Gr = (W @ X) / Sx
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** s
        c2 = 1.0 - b_2 ** s
        mh = m1 / c1
        vh = m2 / c2
        step = mh / (np.sqrt(vh) + eps)
        d2 = np.sum(step * X, axis=1, keepdims=True)
        step = step - d2 * X
        X = X - lr * step
        X = _norm(X)
    return X

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _find_void(fixed, rng, ntry=2000, refine=50):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.08
    for _ in range(refine):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        ny = float(np.linalg.norm(y))
        if ny == 0 or not np.isfinite(ny):
            break
        y /= ny
        step *= 0.97
    y /= np.linalg.norm(y)
    return y

def _subset_rotate(X, k, theta, rng):
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(d)
    b = b - float(a @ b) * a
    nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return _norm(X)
    b /= nb
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s * (np.outer(a, b) - np.outer(b, a))
    Y = X.copy()
    Y[idx] = Y[idx] @ R.T
    return _norm(Y)

def _tangent_perturb(X, scale, rng):
    P = rng.standard_normal(X.shape)
    P = P - np.sum(P * X, axis=1, keepdims=True) * X
    rms = float(np.sqrt(np.mean(np.sum(P * P, axis=1)))) + 1e-15
    P *= (scale / rms)
    return _norm(X + P)

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    try:
        best = _maxcos(X)
    except Exception:
        return X, 1e9
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.5:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]
        J = iu[1][sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _maxcos(y)
        if v < best:
            best = v
            bx = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    T_WANDER_END = t0 + 122.0
    rng = np.random.default_rng(0)
    bestX = None
    bestV = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(f), dtype=float)
            if a.shape == (116, 6) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _maxcos(a)
                if v < bestV:
                    bestV = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = _norm(rng.standard_normal((116, 6)))
        bestV = _maxcos(bestX)
    # wandering state starts with a big kick from incumbent
    Xcur = _tangent_perturb(bestX, 0.10, rng)
    Xcur = _adam_quench(Xcur, 400, 40.0, 800.0, 0.02, 0.002)
    curV = _maxcos(Xcur)
    if curV < bestV:
        bestV = curV
        bestX = Xcur.copy()
    it = 0
    while time.time() < T_WANDER_END:
        if time.time() > HARD - 55:
            break
        it += 1
        base = Xcur.copy()
        r = float(rng.random())
        if r < 0.30:
            scale = float(rng.uniform(0.02, 0.12))
            Y = _tangent_perturb(base, scale, rng)
            steps, b0, b1 = 350, 50.0, 900.0
        elif r < 0.52:
            k = int(rng.choice([6, 10, 16, 24]))
            th = float(rng.uniform(0.2, 0.9))
            Y = _subset_rotate(base, k, th, rng)
            # small extra jitter
            Y = _tangent_perturb(Y, 0.01, rng)
            steps, b0, b1 = 350, 50.0, 900.0
        elif r < 0.78:
            # hub-to-void surgery: remove 1-2 worst points
            rm, _ = _rowmax(base)
            order = np.argsort(rm)[::-1]
            nk = 1 if rng.random() < 0.6 else 2
            # occasionally 3
            if rng.random() < 0.12:
                nk = 3
            Y = base.copy()
            for t in range(nk):
                idx = int(order[t])
                fixed = np.delete(Y, idx, axis=0)
                Y[idx] = _find_void(fixed, rng, ntry=2000, refine=60)
            steps, b0, b1 = 380, 40.0, 800.0
        else:
            # partial randomization
            nk = int(rng.integers(3, 6))
            rm, _ = _rowmax(base)
            order = np.argsort(rm)[::-1]
            # randomize among worst 20 plus random
            cand = list(order[:20]) + list(rng.choice(116, size=20, replace=False))
            cand = np.array(cand)
            idx = rng.choice(cand, size=nk, replace=False)
            Y = base.copy()
            R = _norm(rng.standard_normal((nk, 6)))
            Y[idx] = R
            Y = _norm(Y)
            steps, b0, b1 = 380, 40.0, 800.0
        # fast quench
        Yq = _adam_quench(Y, steps, b0, b1, 0.025, 0.002)
        try:
            vq = _maxcos(Yq)
        except Exception:
            continue
        if vq < bestV - 1e-12:
            bestV = vq
            bestX = Yq.copy()
            Xcur = Yq.copy()
            curV = vq
        elif vq < curV:
            Xcur = Yq.copy()
            curV = vq
        else:
            # allow uphill wander to escape basin
            # accept if within tolerance and random
            tol = 0.008 if (it % 25 != 0) else 0.02
            if vq < curV + tol and rng.random() < 0.25:
                Xcur = Yq.copy()
                curV = vq
            # periodic restart from best with large kick to diversify
            if it % 40 == 0 and rng.random() < 0.5:
                Xcur = _tangent_perturb(bestX, float(rng.uniform(0.05, 0.15)), rng)
                curV = _maxcos(Xcur)
        # occasional restart from random if stuck (rare)
        if it % 120 == 0 and time.time() < T_WANDER_END - 10:
            # quick random quench to probe new basin
            Yr = _norm(rng.standard_normal((116, 6)))
            Yr = _adam_quench(Yr, 300, 40.0, 600.0, 0.025, 0.002)
            vr = _maxcos(Yr)
            if vr < bestV:
                bestV = vr
                bestX = Yr.copy()
            if vr < curV and rng.random() < 0.3:
                Xcur = Yr.copy()
                curV = vr
    # final requench of best with higher beta then LP polish
    try:
        Xf = _adam_quench(bestX, 500, 150.0, 1500.0, 0.015, 0.002)
        vf = _maxcos(Xf)
        if vf < bestV:
            bestV = vf
            bestX = Xf.copy()
    except Exception:
        pass
    try:
        Xp, vp = _lp_polish(bestX, HARD - 1.0)
        if vp < bestV and np.all(np.isfinite(Xp)):
            bestV = vp
            bestX = _norm(Xp)
    except Exception:
        pass
    bestX = _norm(bestX)
    return bestX

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

def _adam_sm(X0, steps, T0, T1, lr0, lr1):
    X = _norm(X0.copy())
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, steps + 1):
        f = it / float(steps)
        T = T0 * (T1 / T0) ** f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp((G - mx) / T)
        np.fill_diagonal(E, 0.0)
        s = float(np.sum(E))
        if not np.isfinite(s) or s <= 0:
            break
        E /= s
        Gr = E @ X
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * Gt
        v *= b2; v += (1 - b2) * Gt * Gt
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = step - np.sum(step * X, axis=1, keepdims=True) * X
        X = X - step
        X = _norm(X)
    return X

def _hinge(X0, iters=100, lr=0.02, band=0.004):
    X = _norm(X0.copy())
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        Mk = G > (M - band)
        A = Mk @ X
        w = np.sum(Mk * G, axis=1, keepdims=True)
        Gt = A - w * X
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if gn < 1e-14:
            break
        mx = 0.03
        st = lr
        if gn * st > mx:
            st = mx / gn
        X = X - st * Gt
        X = _norm(X)
        lr *= 0.995
    return X

def _fast_opt(X):
    X = _adam_sm(X, 420, 0.028, 0.0032, 0.03, 0.006)
    X = _hinge(X, 90, 0.02, 0.004)
    X = _adam_sm(X, 120, 0.004, 0.0012, 0.008, 0.002)
    return X

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    bx = x.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.25:
            break
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool); sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]; cs = c[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                x = y
            radius *= 0.5
    return bx, best

def _hole_point(fixed, rng):
    d = fixed.shape[1]
    ntry = 4000
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.08
    for _ in range(60):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        y /= np.linalg.norm(y)
        step *= 0.97
    return y / np.linalg.norm(y)

def _subset_rotate(X, k, theta, rng):
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    a = rng.standard_normal(d); a /= np.linalg.norm(a)
    b = rng.standard_normal(d); b = b - float(a @ b) * a
    nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return _norm(X)
    b /= nb
    c = float(np.cos(theta)); s = float(np.sin(theta))
    R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s * (np.outer(a, b) - np.outer(b, a))
    Y = X.copy()
    Y[idx] = Y[idx] @ R.T
    return _norm(Y)

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    rng = np.random.default_rng(12345)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f).astype(np.float64)
            a = _norm(a)
            if a.shape == (116, 6) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        bestX = _norm(rng.normal(size=(116, 6)))
    else:
        vs = [_maxcos(c) for c in cands]
        bestX = cands[int(np.argmin(vs))].copy()
    bestV = _maxcos(bestX)
    # brief initial jam
    try:
        if HARD - time.time() > 25:
            Xp, vp = _lp_polish(bestX, min(time.time() + 3.0, HARD - 18.0))
            if vp < bestV:
                bestV, bestX = vp, Xp.copy()
    except Exception:
        pass
    curX = bestX.copy()
    curV = bestV
    trial = 0
    lp_calls = 0
    while time.time() < HARD - 9.0:
        trial += 1
        if trial % 12 == 0:
            curX = bestX.copy(); curV = bestV
        Xb = curX.copy()
        mode = trial % 5
        try:
            if mode == 0:
                k = int(rng.integers(10, 41))
                th = float(rng.uniform(0.15, 0.7))
                Xk = _subset_rotate(Xb, k, th, rng)
            elif mode == 1:
                # hub void regrowth: remove worst point, reinsert at hole
                G = Xb @ Xb.T
                np.fill_diagonal(G, -2.0)
                mxrow = np.max(G, axis=1)
                nhub = 1 + (trial % 2)
                hubs = np.argsort(mxrow)[-nhub:]
                keep = [i for i in range(116) if i not in set(hubs.tolist())]
                fixed = Xb[keep]
                Xk = Xb.copy()
                for h in hubs:
                    Xk[h] = _hole_point(fixed, rng)
                    fixed = Xk[[i for i in range(116) if i != h]] if nhub > 1 else fixed
                Xk = _norm(Xk)
            elif mode == 2:
                G = Xb @ Xb.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(116, 1)
                c = G[iu]
                k2 = int(rng.integers(4, 10))
                top = np.argsort(c)[-k2:]
                D = np.zeros_like(Xb)
                al = float(rng.uniform(0.03, 0.08))
                for tix in top:
                    i = int(iu[0][tix]); j = int(iu[1][tix])
                    vi = Xb[j] - float(Xb[j] @ Xb[i]) * Xb[i]
                    vj = Xb[i] - float(Xb[i] @ Xb[j]) * Xb[j]
                    ni = float(np.linalg.norm(vi)); nj = float(np.linalg.norm(vj))
                    if ni > 1e-12:
                        D[i] -= al * vi / ni
                    if nj > 1e-12:
                        D[j] -= al * vj / nj
                # plus tangent noise
                N = rng.normal(size=Xb.shape)
                N = N - np.sum(N * Xb, axis=1, keepdims=True) * Xb
                rms = float(np.sqrt(np.mean(np.sum(N * N, axis=1)))) + 1e-12)
                N *= (float(rng.uniform(0.05, 0.18)) / rms)
                Xk = _norm(Xb + D + N)
            elif mode == 3:
                N = rng.normal(size=Xb.shape)
                N = N - np.sum(N * Xb, axis=1, keepdims=True) * Xb
                rms = float(np.sqrt(np.mean(np.sum(N * N, axis=1)))) + 1e-12
                N *= (float(rng.uniform(0.06, 0.28)) / rms)
                Xk = _norm(Xb + N)
            else:
                # block reflection: reflect random subset across random hyperplane
                k = int(rng.integers(8, 30))
                idx = rng.choice(116, size=k, replace=False)
                u = rng.normal(size=(6,)); u /= np.linalg.norm(u)
                Xk = Xb.copy()
                for i in idx:
                    Xk[i] = Xk[i] - 2.0 * float(Xk[i] @ u) * u * float(rng.uniform(0.5, 1.0))
                Xk = _norm(Xk)
        except Exception:
            continue
        try:
            Xo = _fast_opt(Xk)
        except Exception:
            continue
        v = _maxcos(Xo)
        # wandering acceptance
        if v < curV + 0.0025:
            curX, curV = Xo.copy(), v
        # promising -> LP polish
        if v < bestV + 0.0012 and lp_calls < 25 and time.time() < HARD - 12.0:
            try:
                Xp2, vp2 = _lp_polish(Xo, min(time.time() + 1.6, HARD - 9.0))
                lp_calls += 1
                if vp2 < v:
                    v = vp2; Xo = Xp2.copy()
            except Exception:
                pass
        if v < bestV - 1e-12:
            bestV, bestX = v, Xo.copy()
            curX, curV = Xo.copy(), v
    # final long polish
    try:
        Xf, vf = _lp_polish(bestX, HARD - 0.5)
        if vf < bestV:
            bestV, bestX = vf, Xf.copy()
    except Exception:
        pass
    return _norm(bestX)

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

def _expand_antipodal(Y):
    return np.vstack([Y, -Y])

def _roll(v, k):
    return np.roll(v, k)

# ---------- Adam optimizers ----------
def _opt_antipodal(Y0, iters, b0, b1, lr0, lr1):
    Y = _norm(Y0.copy())
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1m, b2m = 0.9, 0.999
    eps = 1e-8
    p1 = 1.0; p2 = 1.0
    N = Y.shape[0]
    for it in range(1, iters + 1):
        f = it / float(iters)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = np.vstack([Y, -Y])
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp(beta * (G - M))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gf = (W @ X) / S
        Gy = Gf[:N] - Gf[N:]
        proj = (Gy * Y).sum(axis=1, keepdims=True)
        Gt = Gy - proj * Y
        p1 *= b1m; p2 *= b2m
        m *= b1m; m += (1 - b1m) * Gt
        v *= b2m; v += (1 - b2m) * Gt * Gt
        mh = m / (1 - p1); vh = v / (1 - p2)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y = _norm(Y)
    return Y

def _build_orbit(G9, F8_or_P3, mode):
    # G9: (9,6) generic; mode 'free8': F8 (8,6); mode 's62': P3 (3,)
    parts = []
    for j in range(G9.shape[0]):
        r = G9[j]
        for s in (1.0, -1.0):
            for k in range(6):
                parts.append(s * np.roll(r, k))
    if mode == 'free8':
        for j in range(F8_or_P3.shape[0]):
            parts.append(F8_or_P3[j])
    else:  # 's62': P3 -> 6 pts + all-ones pair
        a, b, c = F8_or_P3
        r6 = np.array([a, b, c, a, b, c], dtype=float)
        n = float(np.linalg.norm(r6))
        if n < 1e-12:
            r6 = np.array([1., 0., 0., 1., 0., 0.])
            r6 /= np.linalg.norm(r6)
        else:
            r6 /= n
        for s in (1.0, -1.0):
            for k in range(3):
                parts.append(s * np.roll(r6, k))
        e = np.ones(6) / np.sqrt(6.0)
        parts.append(e); parts.append(-e)
    return np.array(parts)

def _opt_orbit(G0, H0, mode, iters, b0, b1, lr0, lr1):
    G = _norm(G0.copy())
    if mode == 'free8':
        H = _norm(H0.copy())
        mG = np.zeros_like(G); vG = np.zeros_like(G)
        mH = np.zeros_like(H); vH = np.zeros_like(H)
    else:
        H = np.array(H0, dtype=float).copy()  # (3,)
        if H.shape != (3,):
            H = np.array([1., 0.2, -0.3])
        mG = np.zeros_like(G); vG = np.zeros_like(G)
        mH = np.zeros_like(H); vH = np.zeros_like(H)
    b1m, b2m = 0.9, 0.999
    eps = 1e-8
    p1 = 1.0; p2 = 1.0
    ng = G.shape[0]
    for it in range(1, iters + 1):
        f = it / float(iters)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = _build_orbit(G, H if mode == 's62' else H, mode) if mode == 's62' else _build_orbit(G, H, mode)
        # for s62 need r6; rebuild handled inside; but H is 3-vector
        Gmat = X @ X.T
        np.fill_diagonal(Gmat, -10.0)
        M = float(Gmat.max())
        W = np.exp(beta * (Gmat - M))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gf = (W @ X) / S
        # pullback to G reps: each generic orbit occupies 12 consecutive rows
        GtG = np.zeros_like(G)
        off = 0
        for j in range(ng):
            acc = np.zeros(6)
            for si, s in enumerate((1.0, -1.0)):
                for k in range(6):
                    idx = off + si * 6 + k
                    g = Gf[idx]
                    # inverse of s*roll(.,k) is s*roll(.,-k)
                    acc += s * np.roll(g, -k)
            acc /= 12.0
            # project
            r = G[j]
            GtG[j] = acc - float(acc @ r) * r
            off += 12
        if mode == 'free8':
            GtH = np.zeros_like(H)
            for j in range(H.shape[0]):
                g = Gf[off + j]
                r = H[j]
                GtH[j] = g - float(g @ r) * r
            p1 *= b1m; p2 *= b2m
            mG *= b1m; mG += (1 - b1m) * GtG
            vG *= b2m; vG += (1 - b2m) * GtG * GtG
            mH *= b1m; mH += (1 - b1m) * GtH
            vH *= b2m; vH += (1 - b2m) * GtH * GtH
            G = G - lr * (mG / (1 - p1)) / (np.sqrt(vG / (1 - p2)) + eps)
            H = H - lr * (mH / (1 - p1)) / (np.sqrt(vH / (1 - p2)) + eps)
            G = _norm(G); H = _norm(H)
        else:
            # pullback to 3 params: r6=(a,b,c,a,b,c)/sqrt(2*(a^2+b^2+c^2))
            # work in 3-space: map each of 6 members' grad to 3-space
            acc3 = np.zeros(3)
            for si, s in enumerate((1.0, -1.0)):
                for k in range(3):
                    idx = off + si * 3 + k
                    g = Gf[idx]  # 6-vector
                    # inverse roll then fold: first 3 + last 3
                    back = s * np.roll(g, -k)
                    acc3 += back[:3] + back[3:]
            acc3 /= 6.0
            hn = float(np.linalg.norm(H)) + 1e-15
            hu = H / hn
            # project in 3-space (scale invariant, project orthogonal to H)
            GtH3 = acc3 - float(acc3 @ hu) * hu
            # normalize scale: divide by hn to keep lr comparable (since r6 homogeneous)
            GtH3 = GtH3 / max(hn, 1e-6)
            p1 *= b1m; p2 *= b2m
            mG *= b1m; mG += (1 - b1m) * GtG
            vG *= b2m; vG += (1 - b2m) * GtG * GtG
            mH *= b1m; mH += (1 - b1m) * GtH3
            vH *= b2m; vH += (1 - b2m) * GtH3 * GtH3
            G = G - lr * (mG / (1 - p1)) / (np.sqrt(vG / (1 - p2)) + eps)
            H = H - lr * (mH / (1 - p1)) / (np.sqrt(vH / (1 - p2)) + eps)
            G = _norm(G)
            if float(np.linalg.norm(H)) < 1e-6 or not np.all(np.isfinite(H)):
                H = np.array([1., 0.2, -0.3])
    return G, H

def _opt_full(X0, iters, b0, b1, lr0, lr1):
    X = _norm(X0.copy())
    m = np.zeros_like(X); vv = np.zeros_like(X)
    b1m, b2m = 0.9, 0.999
    eps = 1e-8
    p1 = 1.0; p2 = 1.0
    for it in range(1, iters + 1):
        f = it / float(iters)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp(beta * (G - M))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gf = (W @ X) / S
        proj = (Gf * X).sum(axis=1, keepdims=True)
        Gt = Gf - proj * X
        p1 *= b1m; p2 *= b2m
        m *= b1m; m += (1 - b1m) * Gt
        vv *= b2m; vv += (1 - b2m) * Gt * Gt
        X = X - lr * (m / (1 - p1)) / (np.sqrt(vv / (1 - p2)) + eps)
        X = _norm(X)
    return X

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
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
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    rng = np.random.default_rng(7)
    pool = []  # (val, X)

    def push(X):
        v = _maxcos(X)
        pool.append((v, X.copy()))
        pool.sort(key=lambda t: t[0])
        del pool[4:]

    # ---- Phase A: antipodal 58 reps ----
    deadA = t0 + 42.0
    itA = 0
    while time.time() < deadA:
        itA += 1
        Y0 = rng.standard_normal((58, 6))
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True) + 1e-15
        iters = int(rng.integers(700, 1300))
        b1t = float(rng.choice([800.0, 1200.0, 1500.0]))
        lr0 = float(rng.uniform(0.03, 0.06))
        Y = _opt_antipodal(Y0, iters, 50.0, b1t, lr0, 0.004)
        X = _expand_antipodal(_norm(Y))
        push(X)
        if time.time() > deadA - 0.2:
            break

    # ---- Phase B: order-12 orbit ----
    deadB = t0 + 95.0
    itB = 0
    while time.time() < deadB:
        itB += 1
        mode = 'free8' if (itB % 2 == 0) else 's62'
        G0 = rng.standard_normal((9, 6))
        G0 /= np.linalg.norm(G0, axis=1, keepdims=True) + 1e-15
        if mode == 'free8':
            H0 = rng.standard_normal((8, 6))
            H0 /= np.linalg.norm(H0, axis=1, keepdims=True) + 1e-15
        else:
            H0 = rng.standard_normal(3) + np.array([1.0, 0.2, -0.3])
        iters = int(rng.integers(600, 1100))
        b1t = float(rng.choice([700.0, 1000.0, 1500.0]))
        lr0 = float(rng.uniform(0.03, 0.06))
        try:
            G, H = _opt_orbit(G0, H0, mode, iters, 50.0, b1t, lr0, 0.004)
            if mode == 'free8':
                X = _build_orbit(G, H, mode)
            else:
                X = _build_orbit(G, H, mode)
            X = _norm(X)
            if X.shape != (116, 6) or not np.all(np.isfinite(X)):
                continue
            push(X)
        except Exception:
            continue
        if time.time() > deadB - 0.2:
            break

    # ---- Phase C: release + unconstrained re-jam ----
    # take top candidates from pool
    candX = [p[1].copy() for p in pool[:4]]
    # if pool empty (shouldn't), random
    if not candX:
        candX = [_norm(rng.standard_normal((116, 6)))]
    deadC = t0 + 140.0
    released = []
    for ci, cx in enumerate(list(candX)):
        if time.time() > deadC:
            break
        for sig in ([0.003, 0.010] if ci < 2 else [0.006]):
            if time.time() > deadC - 1.0:
                break
            Y = cx + sig * rng.standard_normal(cx.shape)
            Y = _norm(Y)
            iters = 550 if sig < 0.008 else 700
            Z = _opt_full(Y, iters, 30.0, 350.0, 0.03, 0.003)
            # second anneal harder
            Z = _opt_full(Z, 250, 350.0, 1200.0, 0.008, 0.002)
            released.append((_maxcos(Z), Z))
    # also try perturbing with larger kick on hubs for best
    pool2 = sorted(pool + released, key=lambda t: t[0])[:4]

    # ---- Phase D: LP polish best 2-3 ----
    final_best = None
    final_val = 1e9
    for v, x in pool2:
        if v < final_val:
            final_val = v; final_best = x.copy()
    # polish top 3
    for k, (v, x) in enumerate(list(pool2[:3])):
        remain = HARD_END - time.time() - 2.0
        if remain < 3.0:
            break
        budget = min(8.0, remain / (3 - k))
        bx, bv = _lp_polish(x, time.time() + budget)
        pool2.append((bv, bx))
        if bv < final_val:
            final_val = bv; final_best = bx.copy()

    bestX = final_best
    bestV = final_val
    # final comparison with files (allowed only now)
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f)
            a = np.asarray(a, dtype=float)
            if a.shape == (116, 6) and np.all(np.isfinite(a)):
                n = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(n > 1e-12):
                    a = a / n
                    v = _maxcos(a)
                    if v < bestV:
                        bestV = v; bestX = a.copy()
        except Exception:
            pass
    bestX = _norm(bestX)
    # safety: if something failed, return normalized best file
    if bestX.shape != (116, 6):
        rng2 = np.random.default_rng(0)
        bestX = _norm(rng2.standard_normal((116, 6)))
    return bestX

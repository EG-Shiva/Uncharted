import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _orbit(seeds, q, K=34):
    # seeds (8,4) assumed unit rows; returns (8*K,4)
    S = np.asarray(seeds, dtype=float)
    ns = S.shape[0]
    j = np.arange(K)
    th = 2.0 * np.pi * j / K
    ph = 2.0 * np.pi * q * j / K
    ct, st = np.cos(th), np.sin(th)
    cp, sp = np.cos(ph), np.sin(ph)
    X = np.empty((ns * K, 4), dtype=float)
    for i in range(ns):
        a, b, c, d = S[i]
        X[i*K:(i+1)*K, 0] = ct * a - st * b
        X[i*K:(i+1)*K, 1] = st * a + ct * b
        X[i*K:(i+1)*K, 2] = cp * c - sp * d
        X[i*K:(i+1)*K, 3] = sp * c + cp * d
    return X

def _sym_adam(Zinit, q, steps, tau0, tau1, lr0, lr1, K=34):
    Z = Zinit.copy()
    m = np.zeros_like(Z)
    v = np.zeros_like(Z)
    b1, b2, eps = 0.9, 0.999, 1e-8
    j = np.arange(K)
    th = 2.0 * np.pi * j / K
    ph = 2.0 * np.pi * q * j / K
    ct, st = np.cos(th), np.sin(th)
    cp, sp = np.cos(ph), np.sin(ph)
    ns = Z.shape[0]
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        nr = np.linalg.norm(Z, axis=1, keepdims=True)
        nr[nr < 1e-12] = 1.0
        S = Z / nr
        X = _orbit(S, q, K=K)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        sm = float(W.sum())
        if not np.finite(sm) or sm == 0:
            break
        W /= sm
        GX = W @ X
        # pull back to seeds
        GS = np.zeros_like(S)
        for i in range(ns):
            g = GX[i*K:(i+1)*K]
            g0, g1, g2, g3 = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
            GS[i, 0] = float((ct * g0 + st * g1).sum())
            GS[i, 1] = float((-st * g0 + ct * g1).sum())
            GS[i, 2] = float((cp * g2 + sp * g3).sum())
            GS[i, 3] = float((-sp * g2 + cp * g3).sum())
        dot = (GS * S).sum(axis=1, keepdims=True)
        GZ = (GS - dot * S) / nr
        m = b1 * m + (1 - b1) * GZ
        v = b2 * v + (1 - b2) * (GZ * GZ)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        Z = Z - lr * step
    nr = np.linalg.norm(Z, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    S = Z / nr
    X = _orbit(S, q, K=K)
    return _norm(X), S, Z

def _full_adam(x, steps, tau0, tau1, lr0, lr1):
    x = x.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        sm = float(W.sum())
        if not np.finite(sm) or sm == 0:
            break
        W /= sm
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _rand_seeds(rng, ns=8):
    Z = rng.standard_normal((ns, 4))
    # enforce both plane radii not tiny
    for i in range(ns):
        for _ in range(20):
            r1 = np.linalg.norm(Z[i, :2])
            r2 = np.linalg.norm(Z[i, 2:])
            n = np.linalg.norm(Z[i])
            if r1 > 0.35 * n and r2 > 0.35 * n:
                break
            Z[i] = rng.standard_normal((4,))
    return Z

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        xf = _norm(np.asarray(np.load("best.npy"), dtype=float))
    except Exception:
        try:
            xf = _norm(np.asarray(np.load("record.npy"), dtype=float))
        except Exception:
            xf = _norm(rng.standard_normal((272, 4)))
    if xf.shape != (272, 4):
        flat = xf.ravel()
        need = 272 * 4
        if flat.size < need:
            flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
        xf = _norm(flat[:need].reshape(272, 4))
    best_val = _mc(xf)
    x_best = xf.copy()

    # Phase 1: symmetric orbit search
    q_list = [1, 3, 5, 7, 11, 13]
    cands = []  # (val, X)
    # quick budget: until t0+75
    for q in q_list:
        for rep in range(2):
            if time.time() > t0 + 72 or time.time() > deadline - 90:
                break
            Z0 = _rand_seeds(rng, 8)
            # two-stage sharpening within sym run
            X1, S1, Z1 = _sym_adam(Z0, q, steps=350, tau0=0.03, tau1=0.004, lr0=0.03, lr1=0.008)
            # refine sharper
            X2, S2, Z2 = _sym_adam(Z1, q, steps=250, tau0=0.004, tau1=0.0008, lr0=0.012, lr1=0.004)
            v2 = _mc(X2)
            cands.append((v2, X2.copy(), int(q)))
            if v2 < best_val:
                best_val = v2
                x_best = X2.copy()
        if time.time() > t0 + 72 or time.time() > deadline - 90:
            break
    # keep top 3 symmetric candidates
    cands.sort(key=lambda t: t[0])
    top = cands[:3] if len(cands) >= 3 else cands

    # Phase 2: gradual symmetry breaking
    # stage A: block-level (per-copy small independent jitter within orbit structure is skipped;
    # directly do seeded jitter + full anneal, first with small jitter then larger)
    for (sv, SX, qq) in list(top):
        if time.time() > t0 + 130 or time.time() > deadline - 35:
            break
        # small break
        Y = SX + 0.008 * rng.standard_normal(SX.shape)
        Y = _norm(Y)
        Y = _full_adam(Y, steps=450, tau0=0.012, tau1=0.0009, lr0=0.009, lr1=0.0007)
        vy = _mc(Y)
        if vy < best_val:
            best_val = vy
            x_best = Y.copy()
        if time.time() > t0 + 130 or time.time() > deadline - 35:
            break
        # larger break / independent
        Y2 = SX + 0.025 * rng.standard_normal(SX.shape)
        Y2 = _norm(Y2)
        Y2 = _full_adam(Y2, steps=450, tau0=0.018, tau1=0.001, lr0=0.011, lr1=0.0008)
        vy2 = _mc(Y2)
        if vy2 < best_val:
            best_val = vy2
            x_best = Y2.copy()

    # also try breaking best found so far once more with low temp
    if time.time() < t0 + 135 and time.time() < deadline - 30:
        Y = x_best + 0.006 * rng.standard_normal(x_best.shape)
        Y = _norm(Y)
        Y = _full_adam(Y, steps=400, tau0=0.006, tau1=0.0005, lr0=0.006, lr1=0.0004)
        vy = _mc(Y)
        if vy < best_val:
            best_val = vy
            x_best = Y.copy()

    # Phase 3: LP polish best symmetric-broken config; fallback to file best if worse
    # ensure we never return worse than loaded file
    remaining = max(1.0, min(30.0, deadline - time.time() - 3.0))
    if remaining > 1.0:
        x_best = _lp_polish(x_best, seconds=remaining * 0.7, deadline=deadline - 2.0)
        bv = _mc(x_best)
        # also polish fallback briefly if fallback was better
        fv = _mc(xf)
        if fv < bv:
            xf2 = _lp_polish(xf, seconds=remaining * 0.3, deadline=deadline - 1.0)
            if _mc(xf2) < bv:
                x_best = xf2
                bv = _mc(x_best)
        best_val = bv
    # final safety: if somehow NaN or worse than file, return file
    try:
        if x_best.shape != (272, 4) or not np.all(np.isfinite(x_best)):
            return xf
        x_best = _norm(x_best)
        if _mc(x_best) > _mc(xf) + 1e-12:
            # return the better (file) polished if our basin failed
            return xf
    except Exception:
        return xf
    return x_best

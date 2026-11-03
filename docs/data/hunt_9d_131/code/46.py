import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _sig_sp(U):
    Sig = np.empty_like(U)
    Sp = np.empty_like(U)
    pos = U > 20.0
    neg = U < -20.0
    mid = ~(pos | neg)
    Sig[pos] = 1.0
    Sp[pos] = U[pos]
    e = np.exp(U[neg])
    Sig[neg] = e
    Sp[neg] = e
    if np.any(mid):
        Um = U[mid]
        Sig[mid] = 1.0 / (1.0 + np.exp(-Um))
        Sp[mid] = np.logaddexp(0.0, Um)
    return Sig, Sp

def _adam_tau(X, tau, steps, T0, T1, lr, t_deadline=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute powers for bias correction incrementally
    b1t, b2t = 1.0, 1.0
    for k in range(steps):
        if t_deadline is not None and (k & 255) == 0 and time.time() > t_deadline:
            break
        frac = k / max(1, steps - 1)
        # exponential anneal
        T = T0 * ((T1 / T0) ** frac)
        G = X @ X.T
        U = (G - tau) / T
        Sig, Sp = _sig_sp(U)
        W = (2.0 * Sp * Sig) / T
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1t *= b1
        b2t *= b2
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1t + 1e-300)
        vh = v / (1.0 - b2t + 1e-300)
        direction = mh / (np.sqrt(vh) + eps)
        ddot = np.sum(direction * X, axis=1, keepdims=True)
        direction = direction - ddot * X
        # clip extreme direction to keep trust region
        # Adam direction is O(1) per coord; lr controls step
        X = X - lr * direction
        X = _norm(X)
    return X

def _refine_single(z, XA, steps=250, T0=0.006, T1=0.001, lr=0.08):
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * ((T1 / T0) ** frac) if T0 > 0 and T1 > 0 else T0
        dots = XA @ z
        m0 = float(np.max(dots))
        e = np.exp((dots - m0) / T)
        w = e / (np.sum(e) + 1e-300)
        g = w @ XA
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
    return z

def _attempt(Xb, tau, rng, t_deadline):
    n, d = Xb.shape
    G = Xb @ Xb.T
    np.fill_diagonal(G, -2.0)
    rowmax = np.max(G, axis=1)
    # 9 smallest row-max are rattlers / deepest holes
    rem = np.argsort(rowmax)[:9]
    mask = np.ones(n, dtype=bool)
    mask[rem] = False
    Xc = Xb[mask].copy()
    # small kick to break contacts
    Xc = Xc + 0.008 * rng.standard_normal(Xc.shape)
    Xc = _norm(Xc)
    # core feasibility under tau
    Xc = _adam_tau(Xc, tau, steps=2200, T0=3e-3, T1=2e-4, lr=0.004, t_deadline=t_deadline)
    if time.time() > t_deadline:
        return Xb, _max_cos(Xb)
    # reinsert 9 points one by one
    XA = Xc.copy()
    for _ in range(9):
        Z = rng.standard_normal((20000, d))
        Z = _norm(Z)
        D = Z @ XA.T
        worst = np.max(D, axis=1)
        bi = int(np.argmin(worst))
        z = Z[bi].copy()
        z = _refine_single(z, XA, steps=250)
        XA = np.vstack([XA, z.reshape(1, d)])
    # joint re-polish all 131 under tau penalty
    XA = _adam_tau(XA, tau, steps=1800, T0=3e-3, T1=2e-4, lr=0.003, t_deadline=t_deadline)
    v = _max_cos(XA)
    return XA, v

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap huge constraint sets
        if p > 6000:
            idx = np.argsort(-c[sel])[:6000]
            I, J = I[idx], J[idx]
            p = 6000
            csel = c[sel][idx]
        else:
            csel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < bestV:
                    bestV = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        bestV = _max_cos(best)
    tau = 0.4220
    if bestV <= tau:
        tau = bestV - 0.0005
    feas_deadline = t0 + 155.0
    # feasibility loop
    while time.time() < feas_deadline:
        # leave at least 22s for LP
        if 178.0 - (time.time() - t0) < 24.0:
            break
        # adaptive per-attempt deadline to avoid overrun
        Xa, va = _attempt(best, tau, rng, feas_deadline)
        if va < bestV - 1e-12:
            # improvement: accept even if above tau (creates new active set)
            best, bestV = Xa, va
            if va < tau:
                # target met -> next grid target
                import math as _m
                nxt = _m.floor((va - 1e-12) / 0.0005) * 0.0005
                # ensure strictly below
                if nxt >= va:
                    nxt -= 0.0005
                tau = nxt
                # don't jump more than 0.0005 at once from va
                if tau < va - 0.0005:
                    tau = va - 0.0005
            else:
                # partial: keep same tau, retry with new best
                pass
        else:
            delta = bestV - tau
            if delta < 1e-12:
                tau = bestV - 0.0005
            elif delta < 3e-05:
                # very close target but failing: keep retrying with fresh randomness
                # perturb tau tiny bit to diversify
                pass
            else:
                delta *= 0.5
                tau = bestV - delta
        if tau < 0.35:
            break
    remain = 178.0 - (time.time() - t0)
    lp_sec = min(20.0, max(4.0, remain - 2.0))
    if remain > 6.0:
        try:
            best = _lp_polish(best, lp_sec)
        except Exception:
            pass
    best = _norm(best)
    return best

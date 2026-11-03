import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees(X, mv, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    deg = np.sum(G >= (mv - tol), axis=1)
    return deg, G

def _adam_tau(X, tau, steps, T0, T1, lr0, lr1, rng=None, noise=0.0):
    n, d = X.shape
    if noise > 0 and rng is not None:
        X = _norm(X + noise * rng.standard_normal((n, d)))
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        S = (G - tau) / max(T, 1e-12)
        np.clip(S, -30.0, 20.0, out=S)
        # stable softplus + sigmoid
        sp = np.logaddexp(0.0, S)
        sig = 1.0 / (1.0 + np.exp(-S))
        W = (2.0 * sp * sig) / max(T, 1e-12)
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr = Gr - dot * X
        # adam
        m = b1 * m + (1.0 - b1) * Gr
        v = b2 * v + (1.0 - b2) * (Gr * Gr)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        d = mh / (np.sqrt(vh) + eps)
        # tangential part
        dd = np.sum(d * X, axis=1, keepdims=True)
        d = d - dd * X
        # clip huge steps
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        # adam steps are O(lr) per coord; clip to avoid blowup
        X = X - lr * d
        X = _norm(X)
    return X

def _refine_hole(z, Xk, steps=200, T=0.005, lr=0.02):
    for _ in range(steps):
        dots = Xk @ z
        mm = float(dots.max())
        W = np.exp((dots - mm) / T)
        W = W / (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z / nz
    return z

def _reinsert(Xcore, k_need, rng, ns=20000):
    d = Xcore.shape[1]
    cur = Xcore.copy()
    for _ in range(k_need):
        Z = rng.standard_normal((ns, d))
        Z = _norm(Z)
        D = Z @ cur.T
        worst = D.max(axis=1)
        idx = int(np.argmin(worst))
        z = Z[idx].copy()
        z = _refine_hole(z, cur, steps=200, T=0.005, lr=0.02)
        cur = np.vstack([cur, z[None, :]])
    # return only the new points
    return cur[-k_need:, :]

def _lp(X, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _mc(X)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gf = X @ X.T
        ca = Gf[iu0]
        m = float(ca.max())
        th = m - max(20 * radius, 1e-9)
        sel = ca > th
        if int(sel.sum()) == 0:
            break
        if int(sel.sum()) > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-ca[idx])[:4500]
            mk = np.zeros_like(sel)
            mk[idx[order]] = True
            sel = mk
        I = iu0[0][sel]; J = iu0[1][sel]; c = ca[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        vv = _mc(Y)
        if vv < best:
            X, best = Y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.4:
            break
    return X

def solve():
    t0 = time.time()
    n, d = 131, 9
    rng = np.random.default_rng(0)
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                s = _mc(a)
                if s < best_v:
                    best_v = s; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_v = _mc(best_x)
    x_best = best_x.copy(); s_best = best_v

    FEAS_END = t0 + 158.0
    FINAL_END = t0 + 178.0

    step = 0.0005
    tau = s_best - step
    it = 0
    # tau schedule: 0.4220, 0.4215, ... but relative to current best
    while time.time() < FEAS_END:
        it += 1
        if step < 5e-7:
            step = 5e-7
        tau = s_best - step
        # identify 9 rattlers / lowest degree
        deg, _ = _degrees(x_best, s_best)
        order = np.argsort(deg)  # ascending
        rem = order[:9]
        keep = order[9:]
        # keep original indices mapping not needed; Xcore order irrelevant
        Xcore0 = x_best[keep].copy()
        # core feasibility under tau penalty, small kick to leave basin
        try:
            Xcore = _adam_tau(Xcore0, tau, steps=1400, T0=3e-3, T1=2e-4,
                              lr0=0.003, lr1=0.0004, rng=rng, noise=0.002)
        except Exception:
            step *= 0.5
            continue
        if time.time() > FEAS_END - 2.0:
            break
        # reinsert rattlers: sample >=20000, minimize max core cosine
        try:
            newpts = _reinsert(Xcore, 9, rng, ns=20000)
        except Exception:
            step *= 0.5
            continue
        Xfull = np.vstack([Xcore, newpts])
        # jointly re-polish all 131 under tau penalty
        try:
            Xfull = _adam_tau(Xfull, tau, steps=700, T0=1e-3, T1=2e-4,
                              lr0=0.002, lr1=0.0003, rng=None, noise=0.0)
        except Exception:
            step *= 0.5
            continue
        v = _mc(Xfull)
        if v < s_best - 1e-12:
            # improvement: keep it
            x_best = _norm(Xfull.copy()); s_best = v
            if v < tau:
                # target met: keep or grow step slightly
                step = min(step * 1.15, 0.0005)
            else:
                # improved but target missed: shrink step
                step *= 0.6
        else:
            # no improvement: shrink step, retry from last feasible
            step *= 0.5
            # occasional diversification if step got tiny: kick best slightly and re-evaluate
            if step < 2e-6:
                # small random tangential shake to try new basin, no acceptance unless better
                try:
                    deg2, _ = _degrees(x_best, s_best)
                    # kick a low-degree + random point
                    Y = x_best.copy()
                    k = 3
                    idx = rng.choice(n, k, replace=False)
                    Y[idx] += 0.02 * rng.standard_normal((k, d))
                    Y = _norm(Y)
                    Y = _adam_tau(Y, s_best - 1e-4, steps=500, T0=1e-3, T1=2e-4,
                                  lr0=0.002, lr1=0.0004, rng=None, noise=0.0)
                    vv = _mc(Y)
                    if vv < s_best:
                        x_best = Y; s_best = vv
                        step = 0.0002
                    else:
                        step = 0.0002
                except Exception:
                    step = 0.0002
        # safety: keep tau sequence roughly 0.4220,0.4215... never go absurd
        if s_best < 0.415:
            break
        if FEAS_END - time.time() < 3.0:
            break

    # reserve final 20s for LP polishing
    try:
        x_best = _lp(x_best, FINAL_END)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best

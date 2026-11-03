import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _adam_tau(X, tau, steps, T0, T1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = X @ X.T
        S = (G - tau) / T
        Sc = np.clip(S, -30.0, 30.0)
        Sig = 1.0 / (1.0 + np.exp(-Sc))
        U = np.where(S > 20.0, S, np.log1p(np.exp(Sc)))
        W = (2.0 * U * Sig) / T
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        # tangent projection
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        dx = lr * mh / (np.sqrt(vh) + eps)
        X = X - dx
        X = _norm(X)
    return X

def _find_hole(Xk, rng, n_samples=20000):
    d = Xk.shape[1]
    best_z = None
    best_w = 1e9
    # batch to bound memory but 20000 is fine in one go
    batch = 10000
    done = 0
    while done < n_samples:
        cur = min(batch, n_samples - done)
        Z = rng.randn(cur, d)
        Z = _norm(Z)
        D = Z @ Xk.T
        worst = D.max(axis=1)
        idx = int(np.argmin(worst))
        w = float(worst[idx])
        if w < best_w:
            best_w = w
            best_z = Z[idx].copy()
        done += cur
    return best_z, best_w

def _refine_hole(z, Xk, steps=200, T=0.005, lr=0.03):
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        e = np.exp((dots - m) / T)
        w = e / (e.sum() + 1e-300)
        g = w @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
        T = max(T * 0.995, 0.001)
    return z

def _lp_polish(x, seconds, t_start):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        if int(sel.sum()) > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    FEAS = BUDGET - 22.0  # reserve final 20s for LP
    rng = np.random.RandomState(1)
    n_target, d_target = 131, 9
    best_x = None
    best_s = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n_target, d_target):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_s:
                    best_s = s
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.randn(n_target, d_target))
        best_s = _max_cos(best_x)
    x_best = best_x.copy()
    score_best = best_s

    step = 0.00045
    tau = score_best - step  # ~0.4220
    attempt = 0
    # feasibility loop
    while time.time() - t0 < FEAS:
        if step < 2e-05:
            break
        # need at least ~2.5s for one attempt
        if FEAS - (time.time() - t0) < 2.5:
            break
        attempt += 1
        tau = score_best - step
        try:
            # identify 9 rattlers = 9 smallest per-point maxima
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            pm = G0.max(axis=1)
            order = np.argsort(pm)  # ascending: loosest first
            ratt_idx = order[:9]
            core_idx = order[9:]
            Xcore0 = x_best[core_idx].copy()
            # small jitter to leave basin (larger on retries)
            jit = 0.008 if (attempt % 2 == 1) else 0.015
            Xcore0 = _norm(Xcore0 + jit * rng.randn(*Xcore0.shape))
            # core feasibility Adam under tau, T 3e-3 -> 2e-4
            Xcore = _adam_tau(Xcore0, tau, steps=1200,
                              T0=3e-3, T1=2e-4, lr0=0.012, lr1=0.002)
            # reinsert rattlers one by one into largest holes
            C = Xcore.copy()
            for k in range(9):
                z, _ = _find_hole(C, rng, n_samples=20000)
                z = _refine_hole(z, C, steps=200, T=0.005, lr=0.03)
                C = np.vstack([C, z.reshape(1, -1)])
            # joint re-polish all 131 under tau penalty
            C = _adam_tau(C, tau, steps=800,
                          T0=1e-3, T1=2e-4, lr0=0.006, lr1=0.0012)
            s = _max_cos(C)
            if s < score_best - 1e-12:
                # accept only if true improvement; check feasibility pressure worked
                score_best = s
                x_best = _norm(C).copy()
                # keep step or slightly grow
                step = min(0.0005, step * 1.15)
            else:
                # reject, decrease step
                step *= 0.5
        except Exception:
            step *= 0.5
            continue

    # final LP polish with remaining time (about 20s)
    remain = BUDGET - (time.time() - t0)
    if remain > 4.0:
        try:
            x_best = _lp_polish(x_best, seconds=remain - 1.0, t_start=time.time())
        except Exception:
            pass
    x_best = _norm(x_best)
    if not np.all(np.isfinite(x_best)):
        x_best = _norm(best_x)
    return x_best

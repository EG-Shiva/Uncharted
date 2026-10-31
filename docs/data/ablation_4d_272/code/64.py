import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _smooth(X, steps, a0, a1, s0, s1, mu=0.85):
    X = _norm(X.copy())
    n = X.shape[0]
    vel = np.zeros_like(X)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            X = X - (smax / nrm) * vel
            X = _norm(X)
    return X

def _lp_polish(x, deadline, cap=4000):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 2.0
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def _refine_hole(Y, z0, steps=180, alpha=80.0, s0=0.05, mu=0.6):
    z = np.asarray(z0, dtype=np.float64).copy()
    nz = float(np.linalg.norm(z))
    if nz < 1e-12 or not np.isfinite(nz):
        z = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        z = z / nz
    vel = np.zeros(4)
    for it in range(steps):
        dots = Y @ z
        m = float(dots.max())
        w = np.exp(np.clip(alpha * (dots - m), -50, 0))
        sw = float(w.sum()) + 1e-300
        g = (w @ Y) / sw
        gt = g - float(g @ z) * z
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = mu * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            continue
        step = s0 / (1.0 + it / 60.0)
        z = z - (step / vn) * vel
        nz = float(np.linalg.norm(z))
        if nz < 1e-12 or not np.isfinite(nz):
            continue
        z = z / nz
    return z / (np.linalg.norm(z) + 1e-300)

def _find_hole(Y, rng, n_samples=12000):
    d = Y.shape[1]
    C = rng.standard_normal((n_samples, d))
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-300)
    bs = 3000
    best_s = 2.0
    best_c = C[0].copy()
    for a in range(0, n_samples, bs):
        Cb = C[a:a + bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k = int(np.argmin(sm))
        if float(sm[k]) < best_s:
            best_s = float(sm[k])
            best_c = Cb[k].copy()
    best_c = _refine_hole(Y, best_c)
    return best_c

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    n, d = 272, 4
    rng = np.random.RandomState(20261031)
    best_X = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = _norm(rng.randn(n, d))
        best_val = _max_cos(best_X)
    # initial quick LP polish of incumbent (8s)
    try:
        Xp, vp = _lp_polish(best_X, t0 + 10.0, cap=4000)
        if vp < best_val and np.all(np.isfinite(Xp)):
            best_X, best_val = Xp, vp
    except Exception:
        pass
    it = 0
    while time.time() < HARD_END - 12.0:
        it += 1
        remain = HARD_END - time.time()
        if remain < 14.0:
            break
        base = best_X.copy()
        r = rng.rand()
        if r < 0.35:
            sigma = float(rng.uniform(0.008, 0.03))
            base = base + sigma * rng.randn(n, d)
            base = _norm(base)
            a0 = float(rng.uniform(20.0, 90.0))
            a1 = float(rng.uniform(250.0, 550.0))
            s0 = float(rng.uniform(0.008, 0.018))
            s1 = float(rng.uniform(0.001, 0.004))
            steps = int(rng.randint(300, 550))
            mu = float(rng.uniform(0.78, 0.9))
            try:
                cand = _smooth(base, steps, a0, a1, s0, s1, mu)
            except Exception:
                continue
        elif r < 0.65:
            # stressed kick on subset
            try:
                rm, m0 = _row_max(base)
                order = np.argsort(-rm)
                k = int(rng.randint(6, 16))
                idx = order[:max(k * 2, 10)]
                # pick k among top with randomness
                sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
                sigma = float(rng.uniform(0.06, 0.20))
                base[sel] = base[sel] + sigma * rng.randn(len(sel), d)
                # also jitter neighbors slightly
                base = _norm(base)
            except Exception:
                base = _norm(base + 0.02 * rng.randn(n, d))
            a0 = float(rng.uniform(40.0, 100.0))
            a1 = float(rng.uniform(300.0, 600.0))
            s0 = float(rng.uniform(0.008, 0.016))
            s1 = float(rng.uniform(0.001, 0.003))
            steps = int(rng.randint(350, 600))
            mu = float(rng.uniform(0.8, 0.9))
            try:
                cand = _smooth(base, steps, a0, a1, s0, s1, mu)
            except Exception:
                continue
        else:
            # hole relocation: replace worst point(s) by hole
            try:
                Y = base.copy()
                rm, m0 = _row_max(Y)
                nrep = int(rng.randint(1, 3))
                worst = np.argsort(-rm)[:nrep]
                # build hole set sequentially
                keep = np.ones(n, dtype=bool)
                keep[worst] = False
                Yr = Y[keep]
                for w in worst:
                    z = _find_hole(Yr, rng, n_samples=10000)
                    # small jitter
                    z = z + 0.005 * rng.randn(d)
                    z = z / (np.linalg.norm(z) + 1e-300)
                    Y[w] = z
                    Yr = Y[keep] if False else np.delete(Y, worst, axis=0)
                    # update Yr to include new point for next hole (avoid same hole)
                    Yr = np.concatenate([Yr, z[None, :]], axis=0) if nrep > 1 else Yr
                base = _norm(Y)
            except Exception:
                base = _norm(base + 0.015 * rng.randn(n, d))
            a0 = float(rng.uniform(20.0, 80.0))
            a1 = float(rng.uniform(250.0, 500.0))
            s0 = float(rng.uniform(0.006, 0.014))
            s1 = float(rng.uniform(0.001, 0.003))
            steps = int(rng.randint(300, 550))
            mu = float(rng.uniform(0.8, 0.9))
            try:
                cand = _smooth(base, steps, a0, a1, s0, s1, mu)
            except Exception:
                continue
        try:
            vc = _max_cos(cand)
        except Exception:
            continue
        # only LP-polish if promising (within 0.008 of best) to save time
        if vc > best_val + 0.008:
            continue
        # short polish
        polish_time = min(6.0, max(2.5, (HARD_END - time.time()) * 0.15))
        deadline = time.time() + polish_time
        if deadline > HARD_END - 6.0:
            deadline = HARD_END - 6.0
        try:
            Xp, vp = _lp_polish(cand, deadline, cap=4000)
        except Exception:
            continue
        if vp < best_val and np.all(np.isfinite(Xp)):
            best_X, best_val = Xp, vp
    # final polish with remaining time
    try:
        deadline = HARD_END - 0.5
        if time.time() < deadline - 1.0:
            Xp, vp = _lp_polish(best_X, deadline, cap=5000)
            if vp < best_val and np.all(np.isfinite(Xp)):
                best_X, best_val = Xp, vp
    except Exception:
        pass
    best_X = _norm(best_X)
    return best_X

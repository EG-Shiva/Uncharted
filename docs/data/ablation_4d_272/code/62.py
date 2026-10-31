import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _normr(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _lp_polish(x, deadline, maxpairs=4000):
    x = _normr(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 2.0
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    itc = 0
    while time.time() < deadline and radius > 1e-14 and itc < 60:
        itc += 1
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > maxpairs:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], maxpairs - 1)[:maxpairs]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _normr(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85, deadline=None):
    x = _normr(x.copy())
    vel = np.zeros_like(x)
    for it in range(steps):
        if deadline is not None and (it % 40 == 0) and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _normr(x)
    return x

def _refine_hole(Y, z0, alpha=80.0, steps=220, s0=0.06, mu=0.6):
    z = z0 / (np.linalg.norm(z0) + 1e-300)
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
        step = s0 / (1.0 + it / 80.0)
        z = z - (step / vn) * vel
        nz = float(np.linalg.norm(z))
        if nz < 1e-12 or not np.isfinite(nz):
            z = z0 / (np.linalg.norm(z0) + 1e-300)
            vel *= 0.5
            continue
        z = z / nz
    return z / (np.linalg.norm(z) + 1e-300)

def _find_hole(Y, rng, n_samples=30000):
    d = Y.shape[1]
    best_s = 2.0
    best_c = None
    bs = 5000
    for a in range(0, n_samples, bs):
        nb = min(bs, n_samples - a)
        Cb = rng.standard_normal((nb, d))
        Cb = Cb / np.linalg.norm(Cb, axis=1, keepdims=True)
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k = int(np.argmin(sm))
        if float(sm[k]) < best_s:
            best_s = float(sm[k])
            best_c = Cb[k].copy()
    if best_c is None:
        best_c = rng.standard_normal(d)
        best_c /= np.linalg.norm(best_c)
    best_c = _refine_hole(Y, best_c)
    return best_c

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    rng = np.random.default_rng(272045)
    # load starting point
    X0 = None
    best_val0 = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=np.float64)
            if A.shape == (272, 4):
                A = _normr(A)
                v = _max_cos(A)
                if v < best_val0:
                    best_val0 = v
                    X0 = A.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = _normr(rng.standard_normal((272, 4)))
        best_val0 = _max_cos(X0)
    Xbest = X0.copy()
    best = best_val0
    # initial quick LP lock
    if time.time() < HARD_END - 20:
        Xbest, best = _lp_polish(Xbest, min(HARD_END - 12, time.time() + 12.0))
    Xcur = Xbest.copy()
    cur = best
    stagn = 0
    it = 0
    # main basin hopping
    while time.time() < HARD_END - 12:
        it += 1
        if stagn >= 6:
            Xcur = Xbest.copy(); cur = best; stagn = 0
        mode = int(rng.integers(0, 5))
        n = 272
        try:
            if mode == 0:
                sig = 0.0035
                noise = rng.standard_normal((n, 4)) * sig
                noise -= np.sum(noise * Xcur, axis=1, keepdims=True) * Xcur
                Xt = _normr(Xcur + noise)
                Xt = _smooth_gd(Xt, steps=650, a0=30.0, a1=550.0, s0=0.02, s1=0.002, mu=0.85, deadline=HARD_END - 11)
            elif mode == 1:
                rm, _ = _row_max(Xcur)
                order = np.argsort(-rm)
                k = 20 + int(rng.integers(0, 12))
                tgt = order[:k]
                noise = rng.standard_normal((n, 4)) * 0.002
                noise[tgt] = rng.standard_normal((len(tgt), 4)) * 0.030
                noise -= np.sum(noise * Xcur, axis=1, keepdims=True) * Xcur
                Xt = _normr(Xcur + noise)
                Xt = _smooth_gd(Xt, steps=850, a0=25.0, a1=600.0, s0=0.025, s1=0.002, mu=0.85, deadline=HARD_END - 11)
            elif mode == 2:
                rm, _ = _row_max(Xcur)
                order = np.argsort(-rm)
                # pick victim among top 6 randomly
                vi = int(order[int(rng.integers(0, 6))])
                Y = np.delete(Xcur, vi, axis=0)
                z = _find_hole(Y, rng, n_samples=25000)
                Xt = np.vstack([Y, z])
                # small shake of neighbours of new point to unlock
                dots = (Y @ z)
                nn_idx = np.argsort(-dots)[:12]
                # map to Xt indices (0..270); new point is index 271
                noise = rng.standard_normal(Xt.shape) * 0.0015
                # stronger on neighbours (Y part)
                noise[nn_idx] *= 6.0
                noise -= np.sum(noise * Xt, axis=1, keepdims=True) * Xt
                Xt = _normr(Xt + noise)
                Xt = _smooth_gd(Xt, steps=900, a0=20.0, a1=550.0, s0=0.025, s1=0.002, mu=0.85, deadline=HARD_END - 11)
            elif mode == 3:
                rm, mv = _row_max(Xcur)
                # rattlers: far below max
                ratt = np.where(rm < mv - 1e-3)[0]
                if len(ratt) > 0:
                    vi = int(ratt[int(rng.integers(0, len(ratt)))])
                else:
                    # lowest degree / lowest row max
                    vi = int(np.argmin(rm))
                Y = np.delete(Xcur, vi, axis=0)
                z = _find_hole(Y, rng, n_samples=25000)
                Xt = np.vstack([Y, z])
                Xt = _smooth_gd(Xt, steps=750, a0=25.0, a1=500.0, s0=0.02, s1=0.002, mu=0.85, deadline=HARD_END - 11)
            else:
                sig = 0.07
                noise = rng.standard_normal((n, 4)) * sig
                noise -= np.sum(noise * Xcur, axis=1, keepdims=True) * Xcur
                Xt = _normr(Xcur + noise)
                Xt = _smooth_gd(Xt, steps=1100, a0=12.0, a1=450.0, s0=0.03, s1=0.002, mu=0.88, deadline=HARD_END - 11)
        except Exception:
            stagn += 1
            continue
        try:
            v = _max_cos(Xt)
        except Exception:
            stagn += 1
            continue
        # walk acceptance (allow small uphill from cur)
        if v < cur or (v < cur + 0.002 and float(rng.random()) < 0.25):
            Xcur = Xt; cur = v; stagn = 0 if v < cur else stagn + 1
            # stagn handling: if improved cur, reset partially
            if v < cur + 1e-12:
                pass
        else:
            stagn += 1
        if v < best - 1e-12:
            # lock new record with LP
            try:
                Xp, vp = _lp_polish(Xt, min(HARD_END - 5, time.time() + 4.0))
            except Exception:
                Xp, vp = Xt, v
            if vp < best:
                Xbest, best = Xp, vp
                Xcur, cur = Xp.copy(), vp
                stagn = 0
            elif v < best:
                Xbest, best = Xt, v
                Xcur, cur = Xt.copy(), v
                stagn = 0
        # opportunistic: if very close, try LP rarely
        elif v < best + 2e-6 and (it % 12 == 0) and time.time() < HARD_END - 25:
            try:
                Xp, vp = _lp_polish(Xt, min(HARD_END - 5, time.time() + 3.0))
                if vp < best:
                    Xbest, best = Xp, vp
                    Xcur, cur = Xp.copy(), vp
                    stagn = 0
            except Exception:
                pass
    # final polish with remaining time
    try:
        if time.time() < HARD_END - 2:
            Xbest, best = _lp_polish(Xbest, HARD_END - 1.0)
    except Exception:
        pass
    Xbest = _normr(Xbest)
    return Xbest

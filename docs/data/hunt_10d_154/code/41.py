import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _smooth_refine(X, steps, a0, a1, lr0, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    for t in range(1, steps + 1):
        if (t % 250) == 0 and time.time() > deadline:
            break
        frac = t / steps
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            break
        Grad = (2.0 / S) * (E @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mhat = m / c1
        vhat = v / c2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints to avoid huge LP
        if p > 6000:
            # keep most tight
            idx = np.argpartition(-c[sel], 6000)[:6000]
            I, J = I[idx], J[idx]
            p = 6000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != np.sum(sel) else -c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

def _align_rotate(donor, target):
    # rotate donor to best align with target (Procrustes, d x d)
    M = donor.T @ target
    try:
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return donor @ R
    except Exception:
        return donor

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 118.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    Xbest = best.copy()
    # pool of (val, X)
    pool = [(best_val, Xbest.copy())]
    # add a couple quickly refined randoms to pool for crossover diversity if time permits later
    Xcur = Xbest.copy()
    cur_val = best_val

    it = 0
    while time.time() < HOP_END and time.time() < DEAD - 50:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(HOP_END - t0, 1.0)
        Tacc = 0.0045 * (1.0 - frac) + 0.00025
        r = rng.random()
        if r < 0.22:
            # fresh random: new basin
            Xtry = _norm(rng.standard_normal((n, d)))
            steps, a0, a1, lr0 = 900, 22.0, 550.0, 0.016
        elif r < 0.42:
            # big shake of current wandering state
            sigma = float(rng.uniform(0.25, 0.55))
            Xtry = Xcur + sigma * rng.standard_normal((n, d))
            Xtry = _norm(Xtry)
            steps, a0, a1, lr0 = 900, 25.0, 600.0, 0.015
        elif r < 0.62:
            # massive resample k points
            k = int(rng.integers(15, 41))
            Xtry = Xcur.copy()
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = rng.standard_normal((k, d))
            Xtry = _norm(Xtry + 0.01 * rng.standard_normal((n, d)))
            steps, a0, a1, lr0 = 1000, 22.0, 600.0, 0.015
        elif r < 0.75:
            # spatial cluster resample
            Gc = Xcur @ Xcur.T
            np.fill_diagonal(Gc, -2.0)
            c0 = int(rng.integers(0, n))
            K = int(rng.integers(12, 30))
            nn = np.argsort(-Gc[c0])[:K]
            Xtry = Xcur.copy()
            Xtry[nn] = rng.standard_normal((K, d))
            Xtry = _norm(Xtry + 0.01 * rng.standard_normal((n, d)))
            steps, a0, a1, lr0 = 1000, 22.0, 600.0, 0.015
        elif r < 0.88:
            # crossover with pool member after rotation alignment
            if len(pool) >= 2 and rng.random() < 0.8:
                j = int(rng.integers(0, len(pool)))
                donor = pool[j][1]
                drot = _align_rotate(donor, Xcur)
                mask = rng.random(n) < 0.5
                # ensure nontrivial mix
                if mask.sum() == 0 or mask.sum() == n:
                    mask[rng.integers(0, n)] = not mask[rng.integers(0, n)]
                Xtry = Xcur.copy()
                Xtry[mask] = drot[mask]
                Xtry = _norm(Xtry + 0.02 * rng.standard_normal((n, d)))
            else:
                Xtry = _norm(Xcur + 0.35 * rng.standard_normal((n, d)))
            steps, a0, a1, lr0 = 1000, 24.0, 600.0, 0.014
        else:
            # medium shake
            sigma = float(rng.uniform(0.06, 0.14))
            Xtry = _norm(Xcur + sigma * rng.standard_normal((n, d)))
            steps, a0, a1, lr0 = 800, 28.0, 550.0, 0.015
        # randomize hyperparams slightly
        a1 = a1 * float(rng.uniform(0.85, 1.25))
        lr0 = lr0 * float(rng.uniform(0.85, 1.2))

        Xref = _smooth_refine(Xtry, steps, a0, a1, lr0, deadline=HOP_END)
        val = _max_cos(Xref)

        # update best / pool (no LP here so topology can change)
        if val < best_val - 1e-12:
            best_val = val
            Xbest = Xref.copy()
            pool.append((val, Xref.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:6]
        elif val < best_val + 0.015:
            # keep diverse near-best in pool
            if len(pool) < 6:
                pool.append((val, Xref.copy()))
                pool = sorted(pool, key=lambda z: z[0])[:6]
            elif val < pool[-1][0]:
                pool[-1] = (val, Xref.copy())
                pool = sorted(pool, key=lambda z: z[0])[:6]

        # tempered acceptance to wander (allows uphill)
        if val < cur_val or rng.random() < np.exp((cur_val - val) / max(Tacc, 1e-9)):
            Xcur = Xref.copy()
            cur_val = val
        elif cur_val > best_val + 0.012 and rng.random() < 0.05:
            # restart wandering from best if drifted too far
            Xcur = Xbest.copy()
            cur_val = best_val

    # --- intensive phase: long smooth + LP on top candidates ---
    # deduplicate pool by value
    pool = sorted(pool, key=lambda z: z[0])[:4]
    # ensure Xbest included
    if not any(np.shares_memory(p[1], Xbest) for p in pool):
        pool = [(best_val, Xbest.copy())] + pool[:3]
    final_best = Xbest.copy()
    final_val = best_val
    # budget for intensive: leave ~12s for final polish
    for k, (pv, px) in enumerate(list(pool[:3])):
        if time.time() > DEAD - 18:
            break
        # long high-alpha refine
        try:
            Xr = _smooth_refine(px.copy(), 1600, 25.0, 900.0, 0.012, deadline=DEAD - 14)
        except Exception:
            Xr = px.copy()
        vr = _max_cos(Xr)
        # short LP snap
        tl = min(5.0, max(1.5, (DEAD - 14 - time.time()) / max(3 - k, 1)))
        if tl > 0.8 and time.time() < DEAD - 14:
            try:
                Xp = _lp_polish(Xr, tl, deadline=DEAD - 12)
                vp = _max_cos(Xp)
                if vp < vr:
                    Xr, vr = Xp, vp
            except Exception:
                pass
        if vr < final_val:
            final_val = vr
            final_best = Xr.copy()
    Xbest, best_val = final_best, final_val
    remaining = DEAD - time.time() - 1.0
    if remaining > 2.0:
        try:
            Xp = _lp_polish(Xbest, min(remaining, 12.0), deadline=DEAD - 0.5)
            vp = _max_cos(Xp)
            if vp < best_val:
                Xbest, best_val = Xp, vp
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest

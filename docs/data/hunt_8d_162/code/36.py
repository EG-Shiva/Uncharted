import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(X, iters, a0, a1, s0, s1):
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = a0 * (a1 / a0) ** frac if iters > 1 else a0
        step = s0 * (s1 / s0) ** frac if iters > 1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(alpha * (G - gmax))
        W[W < 1e-12] = 0.0
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        nrm = float(np.mean(np.linalg.norm(Ft, axis=1))) + 1e-12
        X = X - (step / nrm) * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X

def _refine_hole(v0, X, iters=200, beta=120.0, lr0=0.08):
    v = v0.copy()
    v /= np.linalg.norm(v) + 1e-30
    for it in range(iters):
        d = X @ v
        m = float(d.max())
        w = np.exp(beta * (d - m))
        g = w @ X
        gt = g - float(g @ v) * v
        n = float(np.linalg.norm(gt)) + 1e-12
        gt /= n
        step = lr0 * (0.15 ** (it / max(1, iters)))
        v = v - step * gt
        v /= np.linalg.norm(v) + 1e-30
    return v

def _hole_vector(X, rng, n_sample=35000):
    n, d = X.shape
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    dots = C @ X.T
    m = dots.max(axis=1)
    # take a few best diverse
    order = np.argpartition(m, 8)[:8]
    best_v = None
    best_s = 1e9
    for k in order:
        v = _refine_hole(C[k], X, iters=180, beta=100.0, lr0=0.08)
        s = float((X @ v).max())
        if s < best_s:
            best_s = s
            best_v = v
    return best_v

def _grow_one(n_start, n_target, seed, deadline):
    rng = np.random.default_rng(seed)
    d = 8
    X = rng.standard_normal((n_start, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    # optimised starting code: batched soft-max descent from random seed
    X = _anneal(X, 1200, 15.0, 250.0, 0.03, 0.003)
    if time.time() > deadline:
        return None
    for ncur in range(n_start, n_target):
        if time.time() > deadline:
            break
        hv = _hole_vector(X, rng, n_sample=35000)
        X = np.vstack([X, hv[None, :]])
        # re-anneal all points with increasing sharpness after each insertion
        # sharpness grows as we approach target
        frac = (ncur - n_start + 1) / max(1, n_target - n_start)
        a1 = 250.0 + 250.0 * frac
        X = _anneal(X, 500, 20.0, a1, 0.02, 0.002)
    # final longer anneal with high sharpness
    X = _anneal(X, 1200, 30.0, 500.0, 0.012, 0.001)
    return X

def _contact_hist(X, tol=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tol, axis=1)
    return m, cnt

def _lp_polish(X, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1.0:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 4000:
            thr = np.sort(c[sel])[-4000]
            sel = c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 175.0
    # load incumbent
    inc_X = None
    inc_v = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if c.shape == (162, 8) and np.all(np.isfinite(c)):
                v = _max_cos(c)
                if v < inc_v:
                    inc_v = v
                    inc_X = c.copy()
        except Exception:
            pass
    if inc_X is None:
        rng0 = np.random.default_rng(999)
        inc_X = rng0.standard_normal((162, 8))
        inc_X /= np.linalg.norm(inc_X, axis=1, keepdims=True)
        inc_v = _max_cos(inc_X)

    # incremental growth: 2-3 different insertion orders / seeds within ~120s
    growth_deadline = t0 + 115.0
    configs = [(142, 1001), (146, 2002), (150, 3003)]
    results = []
    for n_start, seed in configs:
        if time.time() > growth_deadline - 5.0:
            break
        try:
            Xg = _grow_one(n_start, 162, seed, growth_deadline)
        except Exception:
            continue
        if Xg is None:
            continue
        Xg = Xg / np.linalg.norm(Xg, axis=1, keepdims=True)
        # ensure exactly 162
        if Xg.shape[0] > 162:
            Xg = Xg[:162]
        while Xg.shape[0] < 162:
            # pad rarely if deadline cut inserts
            rngp = np.random.default_rng(seed + Xg.shape[0])
            v = rngp.standard_normal(8); v /= np.linalg.norm(v)
            Xg = np.vstack([Xg, v[None, :]])
            Xg = _anneal(Xg, 300, 25.0, 300.0, 0.015, 0.002)
        v, cnt = _contact_hist(Xg)
        results.append((v, cnt.copy(), Xg.copy()))
        # early exit if time over
        if time.time() > growth_deadline:
            break

    if len(results) == 0:
        return inc_X / np.linalg.norm(inc_X, axis=1, keepdims=True)

    # compare basins by score and contact-count distribution
    results.sort(key=lambda t: t[0])
    # pick best one or two distinct results (distinct seed => distinct topology;
    # also require different contact histogram or score gap to avoid duplicates)
    picked = [results[0]]
    for v, cnt, Xg in results[1:]:
        vm0, cnt0 = picked[0][0], picked[0][1]
        hist_gap = float(np.mean(np.abs(np.sort(cnt) - np.sort(cnt0))))
        if abs(v - vm0) > 1e-6 or hist_gap > 0.5:
            picked.append((v, cnt, Xg))
            break
    if len(picked) == 1 and len(results) > 1:
        picked.append(results[1])
    # LP-polish only best one or two for remaining time
    remain = BUDGET - (time.time() - t0) - 2.0
    per = max(4.0, remain / max(1, len(picked)))
    # also give a short soft-max pre-polish before LP
    polished = []
    for v, cnt, Xg in picked[:2]:
        if time.time() - t0 > BUDGET - 6.0:
            polished.append((v, Xg))
            continue
        try:
            Xp = _anneal(Xg.copy(), 600, 40.0, 500.0, 0.008, 0.001)
            vp = _max_cos(Xp)
            if vp < v:
                v, Xg = vp, Xp
        except Exception:
            pass
        lp_sec = min(per, BUDGET - (time.time() - t0) - 2.0)
        if lp_sec > 3:
            try:
                Xl = _lp_polish(Xg.copy(), lp_sec)
                vl = _max_cos(Xl)
                if vl < v:
                    v, Xg = vl, Xl
            except Exception:
                pass
        polished.append((v, Xg))
    polished.sort(key=lambda t: t[0])
    best_grown_v, best_grown_X = polished[0]
    # return better of grown result and loaded incumbent
    if best_grown_v < inc_v:
        out = best_grown_X / np.linalg.norm(best_grown_X, axis=1, keepdims=True)
    else:
        out = inc_X / np.linalg.norm(inc_X, axis=1, keepdims=True)
    # strict validity: no zero rows, no NaN, unit rows
    out = np.asarray(out, dtype=float)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[~np.isfinite(nrm)] = 1.0
    nrm[nrm < 1e-12] = 1.0
    out = out / nrm
    out[~np.isfinite(out)] = 0.125
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out

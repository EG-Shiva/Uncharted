import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _build_full(seeds):
    # seeds (20,9) -> (180,9) by 9 cyclic rolls each
    seeds = np.asarray(seeds, dtype=float)
    assert seeds.shape == (20, 9)
    X = np.empty((180, 9), dtype=float)
    k = 0
    for s in range(20):
        v = seeds[s]
        for r in range(9):
            X[k] = np.roll(v, r)
            k += 1
    assert X.shape == (180, 9) and X.shape[0] == 180
    return X

def _sym_adam(seeds_init, steps, T0, T1, lr0, rng=None):
    seeds = seeds_init.copy()
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    m = np.zeros_like(seeds)
    v = np.zeros_like(seeds)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Xb = _build_full(seeds)
    bb = _max_cos(Xb)
    best_seeds = seeds.copy()
    best_X = Xb.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        # forward
        X = _build_full(seeds)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E)) + 1e-30
        Gfull = (E @ X) / (S * 0.5)  # (180,9) raw grad
        # pullback to seeds via inverse rolls
        gs = np.zeros_like(seeds)
        k = 0
        for si in range(20):
            acc = np.zeros(9)
            for r in range(9):
                acc += np.roll(Gfull[k], -r)
                k += 1
            gs[si] = acc
        # tangent projection on seeds
        dot = np.sum(gs * seeds, axis=1, keepdims=True)
        gt = gs - dot * seeds
        # adam
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        bc1 = 1 - b1 ** (s + 1)
        bc2 = 1 - b2 ** (s + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        # clip extreme
        seeds = seeds - lr * step
        seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
        if (s & 127) == 127 or s == steps - 1:
            Xc = _build_full(seeds)
            vc = _max_cos(Xc)
            if vc < bb:
                bb = vc
                best_seeds = seeds.copy()
                best_X = Xc.copy()
    return best_seeds, best_X, bb

def _free_descent(x_init, steps, T0, T1, lr0, mu):
    n, d = x_init.shape
    x = x_init.copy()
    bv = _max_cos(x)
    best = x.copy()
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.5 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (s & 255) == 255 or s == steps - 1:
            vv = _max_cos(x)
            if vv < bv:
                bv = vv
                best = x.copy()
    return best, bv

def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(7)
    # load fallback
    base = None
    base_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                vv = _max_cos(c)
                if vv < base_val:
                    base_val = vv
                    base = c.copy()
        except Exception:
            pass
    if base is None:
        base = rng.randn(180, 9)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_val = _max_cos(base)
    # Phase 1: symmetry-reduced search 20x9 -> 180 via cyclic rolls
    sym_best_X = None
    sym_best_val = 1e9
    sym_best_seeds = None
    construction_ok = False
    try:
        # deterministic schedule list for diversity
        cfgs = [
            (0.010, 0.0004, 0.020),
            (0.008, 0.0003, 0.025),
            (0.012, 0.0005, 0.015),
            (0.006, 0.0002, 0.030),
        ]
        t_sym_end = t0 + 45.0
        trial = 0
        for (T0, T1, lr0) in cfgs:
            if time.time() > t_sym_end or time.time() > deadline - 90:
                break
            seeds0 = rng.randn(20, 9)
            seeds0 /= np.linalg.norm(seeds0, axis=1, keepdims=True)
            steps = 1200 if trial < 2 else 900
            bs, bX, bv = _sym_adam(seeds0, steps, T0, T1, lr0)
            # immediate shape/count assertion
            assert bX.shape == (180, 9) and bX.shape[0] == 180
            assert np.all(np.isfinite(bX))
            nr = np.linalg.norm(bX, axis=1)
            assert np.all(nr > 1e-12)
            construction_ok = True
            if bv < sym_best_val:
                sym_best_val = bv
                sym_best_X = bX.copy()
                sym_best_seeds = bs.copy()
            trial += 1
        # one extra longer run from best seeds if time
        if sym_best_seeds is not None and time.time() < t_sym_end - 8 and time.time() < deadline - 100:
            bs2, bX2, bv2 = _sym_adam(sym_best_seeds, 1200, 0.005, 0.0002, 0.012)
            assert bX2.shape == (180, 9) and bX2.shape[0] == 180
            if bv2 < sym_best_val:
                sym_best_val = bv2
                sym_best_X = bX2.copy()
                sym_best_seeds = bs2.copy()
    except Exception:
        construction_ok = False
    if (not construction_ok) or sym_best_X is None:
        # fallback to best.npy path
        try:
            fb = np.load("best.npy")
            fb = np.asarray(fb, dtype=float)
            assert fb.shape == (180, 9)
            sym_best_X = fb / np.linalg.norm(fb, axis=1, keepdims=True)
            sym_best_val = _max_cos(sym_best_X)
        except Exception:
            sym_best_X = base.copy()
            sym_best_val = base_val
    # Phase 2: break symmetry, free optimisation
    free_best = base.copy()
    free_val = base_val
    # consider sym structure as start even if worse (new basin)
    # deterministic symmetry-breaking perturbations 1e-3..1e-2
    sigmas = [0.002, 0.005, 0.010]
    # quick free runs
    for si, sig in enumerate(sigmas):
        if time.time() > deadline - 50:
            break
        X0 = sym_best_X + sig * rng.randn(180, 9)
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # annealed params varied deterministically
        T0 = 0.006 - 0.001 * (si % 3)
        if T0 < 0.002:
            T0 = 0.003
        steps = 900 if time.time() < deadline - 90 else 500
        try:
            Xc, vc = _free_descent(X0, steps, T0, 0.00035, 0.055, 0.85)
            if vc < free_val:
                free_val = vc
                free_best = Xc.copy()
            # second stage cooler refine on same branch
            if time.time() < deadline - 55:
                Xc2, vc2 = _free_descent(Xc, 600, 0.002, 0.00015, 0.035, 0.88)
                if vc2 < free_val:
                    free_val = vc2
                    free_best = Xc2.copy()
        except Exception:
            continue
        # also keep sym itself if it happens to be good
        if sym_best_val < free_val:
            free_val = sym_best_val
            free_best = sym_best_X.copy()
    # pick overall starter for polish: best of free vs base
    if free_val <= base_val:
        xpol = free_best.copy()
    else:
        xpol = free_best.copy() if free_val < sym_best_val + 0.005 else base.copy()
        # prefer free exploration even slightly worse to allow LP to improve
        # but ensure we polish the better one; polish both if time? polish free first
    # Phase 3: LP polish with remaining time
    remain = deadline - time.time() - 1.0
    if remain > 4:
        try:
            secs = min(remain, 45.0)
            xpol = _lp_polish(xpol, secs, deadline - 0.5)
            xpol = xpol / np.linalg.norm(xpol, axis=1, keepdims=True)
            vpol = _max_cos(xpol)
            # compare to base, return best
            if vpol < base_val:
                out = xpol
            else:
                # try polishing base with leftover time
                rem2 = deadline - time.time() - 1.0
                if rem2 > 5:
                    try:
                        xb2 = _lp_polish(base, min(rem2, 20.0), deadline - 0.5)
                        v2 = _max_cos(xb2)
                        out = xb2 if v2 < vpol else xpol
                        if min(v2, vpol) > base_val:
                            out = base
                    except Exception:
                        out = base if base_val < vpol else xpol
                else:
                    out = base if base_val < vpol else xpol
        except Exception:
            out = base if base_val < free_val else free_best
    else:
        out = base if base_val < free_val else free_best
    out = np.asarray(out, dtype=float)
    # strict guards
    out = np.where(np.isnan(out), 0.0, out)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    out = out / nr
    assert out.shape == (180, 9)
    return out

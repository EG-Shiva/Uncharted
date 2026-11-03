import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, init_radius=0.008):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds or radius < 1e-14:
            break
        if seconds - (now - start) < 0.12:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if len(sel_idx) > 2200:
            part = np.argpartition(c_all, -2200)[-2200:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        csel = c_all[sel_idx]
        p = len(sel_idx)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _exp_steps_full(x, steps, alpha, lr, max_step=0.02):
    n, d = x.shape
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _dihedral_perms():
    perms = []
    for k in range(9):
        p = np.array([(i + k) % 9 for i in range(9)], dtype=int)
        perms.append(p)
    for k in range(9):
        p = np.array([(-i + k) % 9 for i in range(9)], dtype=int)
        perms.append(p)
    return perms


def _cyclic_perms():
    perms = []
    for k in range(9):
        p = np.array([(i + k) % 9 for i in range(9)], dtype=int)
        perms.append(p)
    return perms


def _sym_build(reps, perms, fixed):
    K = reps.shape[0]
    P = len(perms)
    d = reps.shape[1]
    X = np.empty((K * P + 1, d))
    for pi, p in enumerate(perms):
        X[pi * K:(pi + 1) * K] = reps[:, p]
    X[-1] = fixed
    return X


def _sym_optimize(K, perms, fixed, rng, schedule, steps_seed, t_deadline, t0):
    P = len(perms)
    d = 9
    pinvs = [np.argsort(p) for p in perms]
    best_val = np.inf
    best_X = None
    # number of restarts depends on time
    n_rest = 3 if K <= 10 else 2
    for rs in range(n_rest):
        if time.time() - t0 > t_deadline:
            break
        reps = rng.standard_normal(size=(K, d))
        reps /= np.linalg.norm(reps, axis=1, keepdims=True)
        # small deterministic offset to break accidental symmetries
        reps += 1e-6 * rng.standard_normal(size=reps.shape)
        reps /= np.linalg.norm(reps, axis=1, keepdims=True)
        for (alpha, steps) in schedule:
            if time.time() - t0 > t_deadline:
                break
            lr = 0.12 / alpha
            for _ in range(steps):
                # build full orbit
                X = np.empty((K * P + 1, d))
                for pi, p in enumerate(perms):
                    X[pi * K:(pi + 1) * K] = reps[:, p]
                X[K * P] = fixed
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                m = float(G.max())
                W = np.exp(alpha * (G - m))
                np.fill_diagonal(W, 0.0)
                gradX = W @ X
                # pullback to reps
                gradR = np.zeros_like(reps)
                for pi in range(P):
                    Gb = gradX[pi * K:(pi + 1) * K]
                    gradR += Gb[:, pinvs[pi]]
                dot = np.sum(gradR * reps, axis=1, keepdims=True)
                gt = gradR - dot * reps
                step = lr * gt
                nm = np.linalg.norm(step, axis=1).max() if K > 0 else 0
                if nm > 0.02 and nm > 0:
                    step *= (0.02 / nm)
                reps = reps - step
                reps /= np.linalg.norm(reps, axis=1, keepdims=True)
        Xf = _sym_build(reps, perms, fixed)
        v = _max_cos(Xf)
        if v < best_val:
            best_val = v
            best_X = Xf.copy()
    return best_X, best_val


def solve():
    t0 = time.time()
    LIMIT = 171.0
    rng = np.random.default_rng(12345)
    n, d = 181, 9
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        xb = rng.standard_normal(size=(n, d))
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    vb = _max_cos(xb)
    try:
        xr = np.load("record.npy").astype(float)
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (n, d):
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            vr = _max_cos(xr)
            if vr < vb:
                xb, vb = xr.copy(), vr
    except Exception:
        pass
    x_best = xb.copy()
    best_val = float(vb)

    fixed = np.ones(9) / 3.0  # norm 1, invariant under permutations

    # ---- Phase S: symmetry-constrained orbit search ----
    sym_deadline = t0 + 32.0
    sym_candidates = []
    # D9: 10 orbits of 18 + fixed = 181
    try:
        perms_d = _dihedral_perms()
        sched_d = [(16.0, 220), (35.0, 220), (80.0, 220), (160.0, 180)]
        Xs, vs = _sym_optimize(10, perms_d, fixed, rng, sched_d, 0, sym_deadline - 12.0, t0)
        if Xs is not None:
            sym_candidates.append((vs, Xs))
    except Exception:
        pass
    # C9: 20 orbits of 9 + fixed = 181
    if time.time() - t0 < sym_deadline:
        try:
            perms_c = _cyclic_perms()
            sched_c = [(16.0, 180), (40.0, 180), (100.0, 180)]
            Xs2, vs2 = _sym_optimize(20, perms_c, fixed, rng, sched_c, 0, sym_deadline, t0)
            if Xs2 is not None:
                sym_candidates.append((vs2, Xs2))
        except Exception:
            pass
    sym_candidates.sort(key=lambda t: t[0])
    if len(sym_candidates) > 0:
        xsym = sym_candidates[0][1].copy()
    else:
        xsym = x_best.copy()

    # ---- Phase B: break symmetry, multi-scale kicks + polish ----
    # quick smooth of symmetric seed itself (asymmetric freedom)
    if time.time() - t0 < LIMIT - 60.0:
        try:
            y0 = xsym.copy()
            y0 = _exp_steps_full(y0, steps=200, alpha=30.0, lr=0.12 / 30.0)
            y0 = _exp_steps_full(y0, steps=200, alpha=120.0, lr=0.12 / 120.0)
            v0 = _max_cos(y0)
            # keep as candidate seed even if worse than best (different basin)
            sym_seed_smoothed = y0
            sym_seed_val = v0
        except Exception:
            sym_seed_smoothed = xsym.copy()
            sym_seed_val = _max_cos(xsym)
    else:
        sym_seed_smoothed = xsym.copy()
        sym_seed_val = _max_cos(xsym)

    seeds = [(sym_seed_smoothed, sym_seed_val), (xsym, _max_cos(xsym)), (x_best, best_val)]
    # order seeds: try symmetric broken seeds first, then incumbent
    # perturbation scales from 1e-4 to 1e-2 as required
    sigmas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.0005, 0.002, 0.006, 0.01, 0.001]
    # interleave seeds
    seed_cycle = [0, 0, 1, 0, 2, 0, 1, 2, 0, 1]
    lp_budget_total = 0
    for it, (sigma, sc) in enumerate(zip(sigmas, seed_cycle)):
        if time.time() - t0 > LIMIT - 22.0:
            break
        base, _bv = seeds[sc % len(seeds)]
        # refresh incumbent seed value
        if sc == 2:
            base = x_best.copy()
        y = base + sigma * rng.standard_normal(size=base.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed smoothing: low-T then high-T
        a1 = float([22.0, 35.0, 50.0][it % 3])
        y = _exp_steps_full(y, steps=220, alpha=a1, lr=0.12 / a1)
        if it % 2 == 0:
            a2 = float([90.0, 150.0][it % 2])
            y = _exp_steps_full(y, steps=200, alpha=a2, lr=0.12 / a2)
        else:
            y = _exp_steps_full(y, steps=160, alpha=70.0, lr=0.12 / 70.0)
        v = _max_cos(y)
        # LP polish if promising or if exploring symmetric basin early
        thresh = best_val + (0.004 if it < 4 else 0.0015)
        # always polish first two symmetric kicks to map basin
        if it < 2:
            thresh = max(thresh, v)
        if v < thresh:
            remain = LIMIT - 22.0 - (time.time() - t0)
            if remain > 2.0:
                b = min(1.8, remain)
                # smaller initial radius for tiny sigma
                r0 = 0.008 if sigma > 0.0005 else 0.003
                y = _lp_polish(y, seconds=b, init_radius=r0)
                v = _max_cos(y)
                lp_budget_total += b
        if v < best_val - 1e-12:
            x_best = y.copy()
            best_val = v
            seeds[2] = (x_best.copy(), best_val)
        if time.time() - t0 > LIMIT - 22.0:
            break

    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        try:
            x_best = _lp_polish(x_best, seconds=min(rem, 18.0), init_radius=0.004)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

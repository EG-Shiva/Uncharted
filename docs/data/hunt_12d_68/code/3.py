import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def soft_opt(x, steps, T0, T1, lr0, lr1, mom, rng, noise_amp=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    # precompute log ratio for geometric schedule
    lt0 = np.log(T0)
    lt1 = np.log(T1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = np.max(G)
        # softmax weights, (G-m)<=0 safe
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        Z = np.sum(E)
        if Z <= 0 or not np.isfinite(Z):
            Z = 1.0
        grad = (E @ x) / Z
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mom * vel + gt
        x = x - lr * vel
        if noise_amp > 0:
            amp = noise_amp * (T / T0)
            nz = rng.standard_normal(size=x.shape)
            # tangential component
            nd = np.sum(nz * x, axis=1, keepdims=True)
            nz -= nd * x
            x = x + amp * nz
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        x = x / nr
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for linprog
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thr = m - max(20 * radius, 1e-9)
        sel = c > thr
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        # cap constraints for speed
        if p > 2500:
            vals_c = c[sel]
            # keep tightest 2500
            idx = np.argpartition(-vals_c, 2500)[:2500]
            I = I[idx]; J = J[idx]
            p = 2500
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != np.sum(sel) else -c[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    # load starting points
    X0 = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (68, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.standard_normal(size=(68, 12))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val = max_cos(X0)
    global_best = X0.copy()
    global_val = best_val

    HOP_BUDGET = 140  # seconds for hopping
    # quick param ranges
    sigmas = np.array([0.01, 0.03, 0.08, 0.15, 0.25])
    trial = 0
    # precompute worst indices of current best periodically
    while time.time() - t0 < HOP_BUDGET:
        trial += 1
        remain = HOP_BUDGET - (time.time() - t0)
        if remain < 0.5:
            break
        r = rng.random()
        if r < 0.68:
            sigma = float(sigmas[rng.integers(len(sigmas))])
            if rng.random() < 0.55:
                G = global_best @ global_best.T
                np.fill_diagonal(G, -2.0)
                mx = np.max(G, axis=1)
                worst = np.argsort(-mx)[:8]
                Xtry = global_best.copy()
                noise = rng.standard_normal(size=Xtry.shape) * sigma
                boost = np.ones((68, 1))
                boost[worst, 0] = 3.0
                Xtry = Xtry + noise * boost
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            else:
                Xtry = global_best + sigma * rng.standard_normal(size=global_best.shape)
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.86:
            k = int(rng.integers(1, 6))
            Xtry = global_best.copy()
            idx = rng.choice(68, size=k, replace=False)
            Xtry[idx] = rng.standard_normal(size=(k, 12))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            Xtry = rng.standard_normal(size=(68, 12))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        steps = 1300 + int(rng.integers(0, 900))
        T0p = 0.015 + rng.random() * 0.015
        T1p = 0.0003 + rng.random() * 0.0007
        lr0 = 0.2 + rng.random() * 0.35
        lr1 = 0.02 + rng.random() * 0.05
        mom = 0.80 + rng.random() * 0.15
        nzamp = 0.001 if rng.random() < 0.3 else 0.0

        Xopt = soft_opt(Xtry, steps, T0p, T1p, lr0, lr1, mom, rng, noise_amp=nzamp)
        v = max_cos(Xopt)
        if v < global_val - 5e-6:
            # promising: short LP to get true optimum
            Xpol = lp_polish(Xopt, seconds=min(3.0, max(0.8, HOP_BUDGET - (time.time() - t0)) * 0.2 + 1.0))
            vp = max_cos(Xpol)
            if vp < global_val:
                global_best = Xpol
                global_val = vp
            # also if soft alone is better but polish failed, keep polished version if better
        elif v < global_val + 0.004 and (trial % 25 == 0):
            # occasional probe of near-miss basins
            if time.time() - t0 < HOP_BUDGET - 2:
                Xpol = lp_polish(Xopt, seconds=1.5)
                vp = max_cos(Xpol)
                if vp < global_val:
                    global_best = Xpol
                    global_val = vp
        # hard time escape
        if time.time() - t0 > HOP_BUDGET:
            break

    # final squeeze
    time_left = 178.0 - (time.time() - t0)
    if time_left > 3:
        try:
            global_best = lp_polish(global_best, seconds=min(time_left - 1.0, 20.0))
            global_val = max_cos(global_best)
        except Exception:
            pass
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    # clean non-finite
    global_best = np.nan_to_num(global_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(global_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    global_best = global_best / nr
    return global_best

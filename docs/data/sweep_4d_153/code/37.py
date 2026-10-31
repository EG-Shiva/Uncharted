import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def gen_from_reps(reps, c1, s1, c2, s2):
    R = reps.shape[0]
    m = len(c1)
    X = np.empty((m * R, 4), dtype=float)
    r0 = reps[:, 0]; r1 = reps[:, 1]; r2 = reps[:, 2]; r3 = reps[:, 3]
    for k in range(m):
        a1 = c1[k]; b1 = s1[k]; a2 = c2[k]; b2 = s2[k]
        sl = slice(k * R, (k + 1) * R)
        X[sl, 0] = r0 * a1 - r1 * b1
        X[sl, 1] = r0 * b1 + r1 * a1
        X[sl, 2] = r2 * a2 - r3 * b2
        X[sl, 3] = r2 * b2 + r3 * a2
    return X


def sym_optimize(reps0, c1, s1, c2, s2, steps, T0, T1, lr0, lr1, mu=0.9, eval_every=100):
    reps = reps0.copy()
    reps /= np.linalg.norm(reps, axis=1, keepdims=True)
    R = reps.shape[0]
    m = len(c1)
    c1b = c1[:, None]; s1b = s1[:, None]
    c2b = c2[:, None]; s2b = s2[:, None]
    vel = np.zeros_like(reps)
    X0 = gen_from_reps(reps, c1, s1, c2, s2)
    best_val = max_cos(X0)
    best = reps.copy()
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        X = gen_from_reps(reps, c1, s1, c2, s2)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S == 0:
            break
        W = E / S
        gradX = W @ X
        gx = gradX.reshape(m, R, 4)
        g00 = gx[:, :, 0]; g01 = gx[:, :, 1]
        g02 = gx[:, :, 2]; g03 = gx[:, :, 3]
        gr0 = ((g00 * c1b + g01 * s1b).sum(axis=0)) / m
        gr1 = ((-g00 * s1b + g01 * c1b).sum(axis=0)) / m
        gr2 = ((g02 * c2b + g03 * s2b).sum(axis=0)) / m
        gr3 = ((-g02 * s2b + g03 * c2b).sum(axis=0)) / m
        gradR = np.stack([gr0, gr1, gr2, gr3], axis=1)
        dot = np.sum(gradR * reps, axis=1, keepdims=True)
        tg = gradR - dot * reps
        vel = mu * vel + tg
        reps = reps - lr * vel
        nr = np.linalg.norm(reps, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        reps /= nr
        if (s + 1) % eval_every == 0 or s == steps - 1:
            X2 = gen_from_reps(reps, c1, s1, c2, s2)
            v = max_cos(X2)
            if v < best_val:
                best_val = v
                best = reps.copy()
    Xb = gen_from_reps(best, c1, s1, c2, s2)
    best_val = max_cos(Xb)
    return best, best_val, Xb


def softmax_anneal(v0, steps, T0, T1, lr0, lr1, mu=0.9):
    v = v0.copy()
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    vel = np.zeros_like(v)
    best = v.copy()
    best_val = max_cos(v)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = v @ v.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S == 0:
            break
        W = E / S
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        tg = grad - dot * v
        vel = mu * vel + tg
        v = v - lr * vel
        nr = np.linalg.norm(v, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        v /= nr
        if (s % 150 == 149) or (s == steps - 1):
            cv = max_cos(v)
            if cv < best_val:
                best_val = cv
                best = v.copy()
    return best


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    deadline = time.time() + seconds
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        r_eq = np.repeat(np.arange(n), d)
        c_eq = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (r_eq, c_eq)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel],
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
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
    t_start = time.time()
    TIME_LIMIT = 172.0
    FINAL_POLISH = 22.0
    rng = np.random.default_rng(20261031)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    best = None
    best_val = 1e9
    for c in [xb, xr]:
        if c is None:
            continue
        try:
            c = np.asarray(c, dtype=float)
            if c.shape != (153, 4):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            v = max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((153, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = max_cos(best)

    # symmetry configs: (m, R, a, b)
    configs = []
    for b in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        configs.append((17, 9, 1, b))
    for b in [1, 2, 5, 8, 20]:
        configs.append((51, 3, 1, b))
    for b in [1, 2, 4]:
        configs.append((9, 17, 1, b))
    configs.append((3, 51, 1, 0))
    configs.append((3, 51, 1, 1))

    sym_results = []
    try:
        for (m, R, a, b) in configs:
            if R * m != 153:
                continue
            if time.time() - t_start > 70.0:
                break
            th1 = 2 * np.pi * a * np.arange(m) / m
            th2 = 2 * np.pi * b * np.arange(m) / m
            c1 = np.cos(th1); s1 = np.sin(th1)
            c2 = np.cos(th2); s2 = np.sin(th2)
            n_init = 6
            for _ in range(n_init):
                if time.time() - t_start > 70.0:
                    break
                reps0 = rng.standard_normal((R, 4))
                reps0 /= np.linalg.norm(reps0, axis=1, keepdims=True)
                br, bv, bX = sym_optimize(reps0, c1, s1, c2, s2,
                                         steps=400, T0=0.02, T1=0.003,
                                         lr0=0.8, lr1=0.08, mu=0.9, eval_every=200)
                sym_results.append((bv, br, (m, R, a, b), (c1, s1, c2, s2)))
                if bv < best_val - 1e-9:
                    pass
        sym_results.sort(key=lambda t: t[0])
        # fully relax top 3
        relaxed = []
        for k in range(min(3, len(sym_results))):
            if time.time() - t_start > 95.0:
                break
            bv, br, cfg, cs = sym_results[k]
            c1, s1, c2, s2 = cs
            br2, bv2, bX2 = sym_optimize(br, c1, s1, c2, s2,
                                        steps=2500, T0=0.015, T1=0.0015,
                                        lr0=0.6, lr1=0.04, mu=0.9, eval_every=250)
            relaxed.append((bv2, bX2))
        # pick best symmetric X
        sym_best_X = None
        sym_best_val = 1e9
        for bv2, bX2 in relaxed:
            if bv2 < sym_best_val:
                sym_best_val = bv2
                sym_best_X = bX2
        if sym_best_X is None and len(sym_results) > 0:
            # generate from cheapest best
            bv, br, cfg, cs = sym_results[0]
            c1, s1, c2, s2 = cs
            sym_best_X = gen_from_reps(br, c1, s1, c2, s2)
            sym_best_val = max_cos(sym_best_X)
        # gradual symmetry breaking
        if sym_best_X is not None and np.all(np.isfinite(sym_best_X)):
            if sym_best_val < best_val + 0.06:
                amps = [0.0, 0.02, 0.05, 0.10, 0.15]
                for amp in amps:
                    if time.time() - t_start > TIME_LIMIT - FINAL_POLISH - 25.0:
                        break
                    y = sym_best_X + amp * rng.standard_normal(sym_best_X.shape)
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    y2 = softmax_anneal(y, steps=1500, T0=0.02, T1=0.002, lr0=0.6, lr1=0.05, mu=0.9)
                    v2 = max_cos(y2)
                    if v2 < best_val:
                        best_val = v2
                        best = y2.copy()
                    # second seed for larger amps
                    if amp >= 0.05:
                        if time.time() - t_start > TIME_LIMIT - FINAL_POLISH - 25.0:
                            break
                        y = sym_best_X + amp * rng.standard_normal(sym_best_X.shape)
                        y /= np.linalg.norm(y, axis=1, keepdims=True)
                        y2 = softmax_anneal(y, steps=1200, T0=0.03, T1=0.002, lr0=0.7, lr1=0.05, mu=0.9)
                        v2 = max_cos(y2)
                        if v2 < best_val:
                            best_val = v2
                            best = y2.copy()
                # polish the broken best a bit
                if time.time() - t_start < TIME_LIMIT - FINAL_POLISH - 5.0:
                    try:
                        yp = lp_polish(best, seconds=2.0)
                        vp = max_cos(yp)
                        if vp < best_val:
                            best_val = vp
                            best = yp
                    except Exception:
                        pass
    except Exception:
        pass

    # extra generic hops from current best to use remaining time
    try:
        while time.time() - t_start < TIME_LIMIT - FINAL_POLISH - 2.0:
            base = best
            y = base + 0.03 * rng.standard_normal(base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y2 = softmax_anneal(y, steps=1200, T0=0.02, T1=0.0025, lr0=0.6, lr1=0.06, mu=0.9)
            try:
                y2 = lp_polish(y2, seconds=0.8)
            except Exception:
                pass
            v2 = max_cos(y2)
            if v2 < best_val - 1e-12:
                best_val = v2
                best = y2
            if time.time() - t_start > TIME_LIMIT - FINAL_POLISH - 1.0:
                break
            # only a few iterations to save time
            break
    except Exception:
        pass

    remain = TIME_LIMIT - (time.time() - t_start) - 1.0
    if remain < 2.0:
        remain = 2.0
    remain = min(remain, 178.0 - (time.time() - t_start) - 1.0)
    if remain > 0.5:
        try:
            best = lp_polish(best, seconds=remain)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

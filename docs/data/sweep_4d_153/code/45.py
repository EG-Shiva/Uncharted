import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, budget, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        try:
            cmat = x @ x.T
            c = cmat[iu]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(12345)

    # load baseline record / best
    baseline = None
    baseline_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (153, 4) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-9):
                    a = a / nr
                    v = _max_cos(a)
                    if v < baseline_val:
                        baseline_val = v
                        baseline = a.copy()
        except Exception:
            continue
    if baseline is None:
        baseline = rng.normal(size=(153, 4))
        baseline /= np.linalg.norm(baseline, axis=1, keepdims=True)
        baseline_val = _max_cos(baseline)

    try:
        # ---- helpers for symmetry phase ----
        def expand_reps(reps, m, aval):
            R = reps.shape[0]
            k = np.arange(m)
            ang1 = 2.0 * np.pi * k / float(m)
            ang2 = 2.0 * np.pi * float(aval) * k / float(m)
            c1 = np.cos(ang1)
            s1 = np.sin(ang1)
            c2 = np.cos(ang2)
            s2 = np.sin(ang2)
            X = np.empty((R * m, 4), dtype=float)
            Xb = X.reshape(R, m, 4)
            r0 = reps[:, 0][:, None]
            r1 = reps[:, 1][:, None]
            r2 = reps[:, 2][:, None]
            r3 = reps[:, 3][:, None]
            Xb[:, :, 0] = r0 * c1 - r1 * s1
            Xb[:, :, 1] = r0 * s1 + r1 * c1
            Xb[:, :, 2] = r2 * c2 - r3 * s2
            Xb[:, :, 3] = r2 * s2 + r3 * c2
            return X

        def run_sym(m, R, aval, steps, seed, T0=0.03, T1=0.002, lr0=0.02, lr1=0.003):
            lrng = np.random.default_rng(seed)
            reps = lrng.normal(size=(R, 4))
            reps /= np.linalg.norm(reps, axis=1, keepdims=True)
            k = np.arange(m)
            ang1 = 2.0 * np.pi * k / float(m)
            ang2 = 2.0 * np.pi * float(aval) * k / float(m)
            c1 = np.cos(ang1)
            s1 = np.sin(ang1)
            c2 = np.cos(ang2)
            s2 = np.sin(ang2)
            mt = np.zeros_like(reps)
            vt = np.zeros_like(reps)
            b1, b2, eps = 0.9, 0.999, 1e-8
            best_val = 1e9
            best_X = None
            for it in range(steps):
                frac = it / max(1, steps - 1)
                T = T0 * (T1 / T0) ** frac
                lr = lr0 * (lr1 / lr0) ** frac
                # expand
                Xb = np.empty((R, m, 4), dtype=float)
                r0 = reps[:, 0][:, None]
                r1 = reps[:, 1][:, None]
                r2 = reps[:, 2][:, None]
                r3 = reps[:, 3][:, None]
                Xb[:, :, 0] = r0 * c1 - r1 * s1
                Xb[:, :, 1] = r0 * s1 + r1 * c1
                Xb[:, :, 2] = r2 * c2 - r3 * s2
                Xb[:, :, 3] = r2 * s2 + r3 * c2
                X = Xb.reshape(R * m, 4)
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                cmax = float(G.max())
                # track best occasionally
                if (it % 100 == 0) or (it == steps - 1):
                    if cmax < best_val:
                        best_val = cmax
                        best_X = X.copy()
                W = np.exp((G - cmax) / T)
                np.fill_diagonal(W, 0.0)
                GradX = (W @ X) / T
                Gb = GradX.reshape(R, m, 4)
                g = np.empty_like(reps)
                g[:, 0] = np.sum(Gb[:, :, 0] * c1 + Gb[:, :, 1] * s1, axis=1)
                g[:, 1] = np.sum(-Gb[:, :, 0] * s1 + Gb[:, :, 1] * c1, axis=1)
                g[:, 2] = np.sum(Gb[:, :, 2] * c2 + Gb[:, :, 3] * s2, axis=1)
                g[:, 3] = np.sum(-Gb[:, :, 2] * s2 + Gb[:, :, 3] * c2, axis=1)
                # tangential projection
                dot = np.sum(g * reps, axis=1, keepdims=True)
                g = g - dot * reps
                # Adam
                mt = b1 * mt + (1.0 - b1) * g
                vt = b2 * vt + (1.0 - b2) * (g * g)
                mhat = mt / (1.0 - b1 ** (it + 1))
                vhat = vt / (1.0 - b2 ** (it + 1))
                step = mhat / (np.sqrt(vhat) + eps)
                reps = reps - lr * step
                nr = np.linalg.norm(reps, axis=1, keepdims=True) + 1e-12
                reps = reps / nr
                if not np.all(np.isfinite(reps)):
                    break
                if time.time() > DEADLINE - 30.0:
                    break
            if best_X is None:
                best_X = expand_reps(reps, m, aval)
                best_val = _max_cos(best_X)
            return best_X, float(best_val)

        # ---- symmetry sweep ----
        results = []
        # C17 with 9 reps, a=1..8
        for ai, aval in enumerate(range(1, 9)):
            if time.time() > t0 + 85.0:
                break
            try:
                Xc, vc = run_sym(17, 9, aval, steps=900, seed=1000 + aval)
                if Xc.shape == (153, 4) and np.all(np.isfinite(Xc)):
                    results.append((vc, Xc))
            except Exception:
                continue
        # C9 with 17 reps
        for aval in (1, 2, 4):
            if time.time() > t0 + 100.0:
                break
            try:
                Xc, vc = run_sym(9, 17, aval, steps=900, seed=2000 + aval)
                if Xc.shape == (153, 4) and np.all(np.isfinite(Xc)):
                    results.append((vc, Xc))
            except Exception:
                continue

        results.sort(key=lambda t: t[0])
        # keep top few to polish
        top = results[:3]

        overall_best = baseline.copy()
        overall_val = baseline_val

        # also consider raw symmetric codes
        for vc, Xc in top:
            if vc < overall_val:
                overall_best = Xc.copy()
                overall_val = vc

        # break symmetry with 1e-3 tangential noise + LP polish
        remaining = DEADLINE - time.time() - 2.0
        if remaining < 4.0:
            # no time to polish; return best so far
            overall_best /= np.linalg.norm(overall_best, axis=1, keepdims=True)
            return overall_best
        per = min(10.0, remaining / max(1, len(top) + 1))
        for vc, Xc in top:
            if time.time() > DEADLINE - 6.0:
                break
            try:
                y = Xc + rng.normal(size=Xc.shape) * 1e-3
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                z = _lp_polish(y, budget=per, t_deadline=DEADLINE - 3.0)
                vz = _max_cos(z)
                if vz < overall_val:
                    overall_best = z.copy()
                    overall_val = vz
            except Exception:
                continue
        # final polish of overall best with leftover time
        leftover = DEADLINE - time.time() - 1.5
        if leftover > 3.0:
            try:
                z = _lp_polish(overall_best, budget=min(leftover, 20.0), t_deadline=DEADLINE - 1.0)
                vz = _max_cos(z)
                if vz < overall_val and np.all(np.isfinite(z)):
                    overall_best = z
                    overall_val = vz
            except Exception:
                pass
        overall_best /= np.linalg.norm(overall_best, axis=1, keepdims=True)
        # validate, else fallback
        if overall_best.shape != (153, 4) or not np.all(np.isfinite(overall_best)):
            return baseline
        return overall_best
    except Exception:
        try:
            b = baseline / np.linalg.norm(baseline, axis=1, keepdims=True)
            return b
        except Exception:
            x = np.random.default_rng(0).normal(size=(153, 4))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            return x

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def soft_refine(Xc, steps=650):
    """Annealed exp(beta*G) repulsion with normalized tangent GD + momentum.
    Returns best encountered."""
    n, d = Xc.shape
    Xw = Xc.copy()
    V = np.zeros_like(Xw)
    mu = 0.9
    best_X = Xw.copy()
    best_v = max_cos(Xw)
    for s in range(steps):
        frac = s / float(max(1, steps))
        beta = 30.0 * ((400.0 / 30.0) ** frac)
        step = 0.02 * (1.0 - 0.7 * frac)
        G = Xw @ Xw.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        grad = E @ Xw
        gdot = np.sum(grad * Xw, axis=1, keepdims=True)
        grad -= gdot * Xw
        gn = float(np.linalg.norm(grad, axis=1).max())
        if gn > 1e-12:
            grad /= gn
        V = mu * V + (1.0 - mu) * grad
        Xw = Xw - step * V
        Xw /= np.linalg.norm(Xw, axis=1, keepdims=True)
        if (s + 1) % 50 == 0 or s == steps - 1:
            v = max_cos(Xw)
            if v < best_v:
                best_v = v
                best_X = Xw.copy()
    return best_X, best_v


def solve():
    t0 = time.time()

    # load starting point
    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            X0 = np.load(fn)
            break
        except Exception:
            continue
    if X0 is None:
        rng0 = np.random.default_rng(0)
        X0 = rng0.standard_normal((95, 13))
    X0 = np.asarray(X0, dtype=np.float64)
    # ensure shape (95,13)
    if X0.shape != (95, 13):
        # fallback: truncate / pad deterministically
        rng0 = np.random.default_rng(1)
        tmp = rng0.standard_normal((95, 13))
        r = min(95, X0.shape[0])
        c = min(13, X0.shape[1] if X0.ndim > 1 else 13)
        tmp[:r, :c] = X0[:r, :c]
        X0 = tmp
    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

    best = X0.copy()
    best_val = max_cos(best)
    rng = np.random.default_rng(0)
    n, d = best.shape

    attempt = 0
    # main basin-hopping loop, reserve time for final polish
    while True:
        elapsed = time.time() - t0
        if elapsed > 156:
            break
        attempt += 1
        mode = attempt % 7

        # contact analysis of current best
        G0 = best @ best.T
        np.fill_diagonal(G0, -10.0)
        m0 = float(G0.max())
        contacts = np.sum(G0 > m0 - 1e-7, axis=1)
        rattlers = np.where(contacts == 0)[0]
        worst_order = np.argsort(-contacts)

        Xc = best.copy()
        if mode == 0:
            Xc = Xc + 0.010 * rng.standard_normal((n, d))
        elif mode == 1:
            Xc = Xc + 0.030 * rng.standard_normal((n, d))
        elif mode == 2:
            Xc = Xc + 0.070 * rng.standard_normal((n, d))
        elif mode == 3:
            noise = 0.005 * rng.standard_normal((n, d))
            worst = worst_order[:8]
            noise[worst] = 0.25 * rng.standard_normal((len(worst), d))
            if len(rattlers) > 0:
                noise[rattlers[:2]] = 0.50 * rng.standard_normal((min(2, len(rattlers)), d))
            Xc = Xc + noise
        elif mode == 4:
            # replace rattlers + worst point by best voids
            to_replace = []
            if len(rattlers) > 0:
                to_replace.extend(list(rattlers[:2]))
            w0 = int(worst_order[0])
            if w0 not in to_replace:
                to_replace.append(w0)
            # candidate voids
            cand = rng.standard_normal((8000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            for idx in to_replace:
                dots = cand @ Xc.T
                dots[:, idx] = -2.0
                vals = dots.max(axis=1)
                bi = int(np.argmin(vals))
                Xc[idx] = cand[bi]
            Xc = Xc + 0.010 * rng.standard_normal((n, d))
        elif mode == 5:
            Xc = Xc + 0.150 * rng.standard_normal((n, d))
        else:
            # large jump, partly random restart biased to best
            Xc = Xc + 0.300 * rng.standard_normal((n, d))

        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
        steps = 900 if mode == 6 else 650
        Xr, vr = soft_refine(Xc, steps=steps)

        if vr < best_val - 1e-12:
            remaining = 165 - (time.time() - t0)
            if remaining > 4:
                Xr2 = lp_polish(Xr, min(4.0, remaining - 1.0))
                v2 = max_cos(Xr2)
                if v2 < best_val:
                    best, best_val = Xr2, v2
                elif vr < best_val:
                    best, best_val = Xr, vr
            else:
                best, best_val = Xr, vr
        elif vr < best_val + 0.0008 and (attempt % 12 == 0):
            # occasional polish of near-miss (cheap)
            remaining = 165 - (time.time() - t0)
            if remaining > 5:
                Xr2 = lp_polish(Xr, 2.0)
                v2 = max_cos(Xr2)
                if v2 < best_val:
                    best, best_val = Xr2, v2

    # final squeeze of global best
    remaining = 178 - (time.time() - t0)
    if remaining > 2.5:
        try:
            best = lp_polish(best, min(max(2.0, remaining - 1.0), 14.0))
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

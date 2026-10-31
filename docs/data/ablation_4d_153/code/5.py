import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
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
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_refine(x_init, steps, a0, a1, s0, s1, n0, n1, mu, rng):
    x = x_init.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    v = np.zeros_like(x)
    for it in range(steps):
        frac = it / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** frac if a0 > 0 and a1 > 0 else a0
        if s0 > 0 and s1 > 0:
            step = s0 * (s1 / s0) ** frac
        else:
            step = s0 + (s1 - s0) * frac
        if n0 > 0:
            if n1 > 0:
                noise = n0 * (n1 / n0) ** frac
            else:
                noise = n0 * (1.0 - frac)
        else:
            noise = 0.0
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        M = float(np.max(G))
        # E = exp(alpha*(G-M)) reusing G buffer
        np.exp(alpha * (G - M), out=G)
        E = G
        sumU = float(E.sum() * 0.5)
        if not np.isfinite(sumU) or sumU <= 0:
            continue
        E /= sumU
        grad = E @ x
        dots = np.sum(grad * x, axis=1, keepdims=True)
        Gt = grad - dots * x
        rms = float(np.sqrt(np.mean(Gt * Gt) + 1e-24))
        direction = Gt / rms
        v = mu * v + (1.0 - mu) * direction
        if noise > 1e-13:
            Zn = rng.standard_normal((n, d))
            Zdot = np.sum(Zn * x, axis=1, keepdims=True)
            Zt = Zn - Zdot * x
            zrms = float(np.sqrt(np.mean(Zt * Zt) + 1e-24))
            Zt /= zrms
            x = x - step * v + noise * Zt
        else:
            x = x - step * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def solve():
    t_start = time.time()
    HARD = t_start + 172.0
    n, d = 153, 4
    rng = np.random.default_rng(0)
    np.random.seed(0)

    def normed(a):
        a = np.asarray(a, dtype=float)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a

    best = None
    best_val = 2.0
    try:
        xb = normed(np.load("best.npy"))
        if xb.shape == (n, d) and np.all(np.isfinite(xb)):
            best = xb
            best_val = max_cos(xb)
    except Exception:
        pass
    try:
        xr = normed(np.load("record.npy"))
        if xr.shape == (n, d) and np.all(np.isfinite(xr)):
            rv = max_cos(xr)
            if rv < best_val:
                best = xr
                best_val = rv
    except Exception:
        pass
    if best is None:
        best = normed(rng.standard_normal((n, d)))
        best_val = max_cos(best)

    trial = 0
    # reserve ~25s for final polish
    while True:
        now = time.time()
        if now > HARD - 25:
            break
        if trial >= 80:
            break
        cfg = trial % 5
        if cfg == 0:
            sigma = 0.012
            xtry = best + rng.standard_normal((n, d)) * sigma
            xtry /= np.linalg.norm(xtry, axis=1, keepdims=True)
            pars = (2000, 15.0, 180.0, 0.012, 0.0012, 0.003, 0.0, 0.85)
        elif cfg == 1:
            xtry = best + rng.standard_normal((n, d)) * 0.004
            k = 6
            idx = rng.choice(n, k, replace=False)
            xtry[idx] += rng.standard_normal((k, d)) * 0.13
            xtry /= np.linalg.norm(xtry, axis=1, keepdims=True)
            pars = (2000, 25.0, 250.0, 0.010, 0.0010, 0.002, 0.0, 0.90)
        elif cfg == 2:
            sigma = 0.030
            xtry = best + rng.standard_normal((n, d)) * sigma
            xtry /= np.linalg.norm(xtry, axis=1, keepdims=True)
            pars = (1800, 10.0, 120.0, 0.015, 0.0015, 0.005, 0.0, 0.80)
        elif cfg == 3:
            xtry = best + rng.standard_normal((n, d)) * 0.005
            k = 12
            idx = rng.choice(n, k, replace=False)
            xtry[idx] = rng.standard_normal((k, d))
            xtry /= np.linalg.norm(xtry, axis=1, keepdims=True)
            pars = (2000, 12.0, 160.0, 0.013, 0.0012, 0.003, 0.0, 0.85)
        else:
            sigma = 0.055
            xtry = best + rng.standard_normal((n, d)) * sigma
            xtry /= np.linalg.norm(xtry, axis=1, keepdims=True)
            pars = (1600, 18.0, 200.0, 0.014, 0.0010, 0.004, 0.0, 0.85)

        steps, a0, a1, s0, s1, n0, n1, mu = pars
        xopt = smooth_refine(xtry, steps, a0, a1, s0, s1, n0, n1, mu, rng)
        v = max_cos(xopt)
        if v < best_val + 0.003:
            tleft = HARD - time.time()
            if tleft > 26:
                alloc = min(3.0, tleft - 22.0)
                try:
                    xpol = lp_polish(xopt, seconds=alloc)
                except Exception:
                    xpol = xopt
                vp = max_cos(xpol)
                # take the better of polished / raw
                if vp < v:
                    if vp < best_val - 1e-12:
                        best = xpol
                        best_val = vp
                    elif v < best_val - 1e-12:
                        best = xopt
                        best_val = v
                else:
                    if v < best_val - 1e-12:
                        best = xopt
                        best_val = v
                    elif vp < best_val - 1e-12:
                        best = xpol
                        best_val = vp
            else:
                if v < best_val - 1e-12:
                    best = xopt
                    best_val = v
        trial += 1

    tleft = HARD - time.time() - 0.5
    if tleft > 2:
        try:
            best = lp_polish(best, seconds=tleft)
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())


def _gd_refine(X, steps, rng_jitter=0.0, rng=None, iso0=0.25):
    n, d = X.shape
    Xw = X.copy()
    V = np.zeros_like(Xw)
    mu = 0.88
    best_X = Xw.copy()
    best_v = _max_cos(Xw)
    I_d = np.eye(d) / d
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = 25.0 * ((500.0 / 25.0) ** frac)
        step = 0.035 * (1.0 - 0.65 * frac)
        iso_w = iso0 * (1.0 - frac)  # decay to pure minimax
        G = Xw @ Xw.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        esum = float(E.sum())
        if esum > 0:
            E /= esum
        Gr = E @ Xw
        # tangent project + normalize soft part
        gdot = np.sum(Gr * Xw, axis=1, keepdims=True)
        Gr = Gr - gdot * Xw
        gn = float(np.linalg.norm(Gr, axis=1).max())
        if gn > 1e-14:
            Gr /= gn
        if iso_w > 0:
            mean = Xw.mean(axis=0, keepdims=True)  # 1xd
            C = (Xw.T @ Xw) / n
            Em = C - I_d
            Ga = Xw @ Em
            # center part: push opposite centroid
            Gc = np.repeat(mean, n, axis=0)
            gdot2 = np.sum(Gc * Xw, axis=1, keepdims=True)
            Gc = Gc - gdot2 * Xw
            gdot3 = np.sum(Ga * Xw, axis=1, keepdims=True)
            Ga = Ga - gdot3 * Xw
            na = float(np.linalg.norm(Ga, axis=1).max())
            if na > 1e-14:
                Ga /= na
            nc = float(np.linalg.norm(Gc, axis=1).max())
            if nc > 1e-14:
                Gc /= nc
            Gr = Gr + iso_w * (0.6 * Ga + 0.4 * Gc)
            gdot = np.sum(Gr * Xw, axis=1, keepdims=True)
            Gr = Gr - gdot * Xw
            gn2 = float(np.linalg.norm(Gr, axis=1).max())
            if gn2 > 1e-14:
                Gr = Gr / gn2
        V = mu * V + (1.0 - mu) * Gr
        Xw = Xw - step * V
        Xw /= np.linalg.norm(Xw, axis=1, keepdims=True)
        if (s + 1) % 60 == 0 or s == steps - 1:
            v = _max_cos(Xw)
            if v < best_v:
                best_v = v
                best_X = Xw.copy()
    return best_X, best_v


def _greedy_voids(Xkeep, k, rng, pool=12000):
    d = Xkeep.shape[1]
    out = []
    cur = Xkeep.copy()
    for _ in range(k):
        cand = rng.standard_normal((pool, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ cur.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        out.append(cand[bi])
        cur = np.vstack([cur, cand[bi][None, :]])
    return np.array(out)


def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.3:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1600:
            idx = np.argpartition(c, -1600)[-1600:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    DEAD = t0 + 176.0
    EXPL = t0 + 158.0
    rng = np.random.default_rng(0)
    n, d = 95, 13

    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (95, 13):
                X0 = a / np.linalg.norm(a, axis=1, keepdims=True)
                break
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

    best = X0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    # quick baseline refine of seed with isotropy off to gauge
    # (cheap, keeps determinism)
    trial = 0
    sigmas = [0.02, 0.06, 0.14, 0.28, 0.45]

    # a few fresh random starts first for different basin
    n_fresh = 4
    for fi in range(n_fresh):
        if time.time() > EXPL - 20:
            break
        trial += 1
        Xr = rng.standard_normal((n, d))
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        iso = 0.35 if fi < 2 else 0.15
        Xo, vo = _gd_refine(Xr, steps=700, iso0=iso)
        if vo < best_val:
            best, best_val = Xo, vo
            cur, cur_val = Xo.copy(), vo
        elif vo < cur_val + 0.004 and rng.random() < 0.4:
            cur, cur_val = Xo.copy(), vo

    while time.time() < EXPL:
        if time.time() > DEAD - 25:
            break
        trial += 1
        mode = trial % 9
        Xt = cur.copy()
        if mode in (0, 1):
            sig = sigmas[(trial // 9) % len(sigmas)]
            # subset shake: coherent block + global jitter
            Xt = Xt + sig * rng.standard_normal((n, d))
        elif mode == 2:
            # shake worst-contact hub region hard
            rm, m = _row_max(cur)
            order = np.argsort(-rm)
            k = int(rng.integers(6, 14))
            idx = order[:k]
            Xt[idx] += 0.35 * rng.standard_normal((k, d))
            Xt += 0.015 * rng.standard_normal((n, d))
        elif mode == 3:
            # reinsert 2-5 worst points into voids
            rm, m = _row_max(cur)
            order = np.argsort(-rm)
            k = int(rng.integers(2, 6))
            rep = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[rep] = False
            Xkeep = cur[keep_mask]
            voids = _greedy_voids(Xkeep, k, rng, pool=8000)
            Xt[rep] = voids + 0.005 * rng.standard_normal((k, d))
            Xt += 0.008 * rng.standard_normal((n, d))
        elif mode == 4:
            # random kill + void fill (topology jump)
            k = int(rng.integers(3, 8))
            rep = rng.choice(n, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[rep] = False
            Xkeep = cur[keep_mask]
            voids = _greedy_voids(Xkeep, k, rng, pool=8000)
            Xt[rep] = voids
            Xt += 0.01 * rng.standard_normal((n, d))
        elif mode == 5:
            # mirror / rotation kick: random orthogonal in a 2D subspace + shake
            Xt = Xt + 0.10 * rng.standard_normal((n, d))
            # flip a random subset through origin complement
            k = int(rng.integers(8, 20))
            idx = rng.choice(n, size=k, replace=False)
            # reflect those points across random hyperplane
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            Xt[idx] = Xt[idx] - 2.0 * np.outer(Xt[idx] @ v, v) * 0.5
        elif mode == 6:
            # large global jump biased to best
            Xt = best + 0.30 * rng.standard_normal((n, d))
        elif mode == 7:
            # crossover between best and cur
            mask = rng.random(n) < 0.5
            Xt[mask] = best[mask]
            Xt += 0.05 * rng.standard_normal((n, d))
        else:
            Xt = Xt + 0.03 * rng.standard_normal((n, d))

        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        # short vs long refine alternating
        steps = 750 if (trial % 3 == 0) else 450
        iso0 = 0.25 if (trial % 4 < 2) else 0.05
        Xo, vo = _gd_refine(Xt, steps=steps, iso0=iso0)

        if vo < best_val - 1e-12:
            best, best_val = Xo.copy(), vo
            cur, cur_val = Xo.copy(), vo
        elif vo < cur_val:
            cur, cur_val = Xo.copy(), vo
        elif vo < cur_val + 0.002 and rng.random() < 0.25:
            cur, cur_val = Xo.copy(), vo
        elif trial % 15 == 0 and rng.random() < 0.15:
            # occasional random walk to escape
            cur, cur_val = Xo.copy(), vo

        # early exit if we already beat record comfortably: keep pushing
        if time.time() > EXPL - 5:
            break

    # final squeeze
    if time.time() < DEAD - 3:
        try:
            secs = min(max(2.0, DEAD - time.time() - 1.0), 15.0)
            Bp = _lp_polish(best, secs, deadline=DEAD - 0.5)
            v = _max_cos(Bp)
            if v < best_val:
                best, best_val = Bp, v
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best

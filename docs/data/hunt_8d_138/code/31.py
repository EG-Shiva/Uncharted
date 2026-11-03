import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
             shape=(n, nv)
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _refine_hole(v0, X, steps=250, beta=150.0, lr=0.12):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    d = v.shape[0]
    for t in range(steps):
        dots = X @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ X) / S
        gt = g - np.dot(g, v) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        nrm = np.linalg.norm(v) + 1e-30
        v = v / nrm
        if t == steps // 2:
            beta = beta * 1.5
    return v


def _find_hole_candidates(X, rng, n_sample=50000):
    d = X.shape[1]
    cand = rng.standard_normal((n_sample, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    smax = (cand @ X.T).max(axis=1)
    # 5 best holes (smallest smax)
    part = np.argpartition(smax, 5)[:5]
    # sort them for determinism
    part = part[np.argsort(smax[part])]
    return cand, smax, part


def _prune_to(X, target):
    X = X.copy()
    while X.shape[0] > target:
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # for each k, max without k
        best_k = -1
        best_v = 1e9
        # vectorized: precompute top two maxima? brute is fine for n~142
        for k in range(n):
            # max over pairs not involving k
            # mask rows/cols != k: just compute max of G with row k col k excluded
            # quick: copy row/col? use masking via max of submatrix
            # G is symmetric; max without k = max of G[mask][:,mask]
            # do efficient: max of full unless argmax involves k
            # fallback brute
            mask = np.ones(n, dtype=bool)
            mask[k] = False
            sub = G[np.ix_(mask, mask)]
            v = float(sub.max())
            if v < best_v:
                best_v = v
                best_k = k
        X = np.delete(X, best_k, axis=0)
    return X


def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    n_target, d = 138, 8
    rng_master = np.random.default_rng(0)

    # load incumbent
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n_target, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        r = rng_master.standard_normal((n_target, d))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        best = r
        best_val = _max_cos(best)

    # identify rattlers / core
    try:
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        thr = 1e-9
        is_contact = G > (m0 - thr)
        per = is_contact.sum(axis=1)
        order = np.argsort(per)  # ascending: rattlers first
        # 30 rattlers with 0 contacts expected
        n_ratt = 30
        # if fewer zeros, take 30 lowest
        ratt_idx = order[:n_ratt]
        core_idx = order[n_ratt:]  # 108
        core = best[core_idx].copy()
    except Exception:
        core = best[:108].copy()

    # polish core subcode briefly
    try:
        core = _lp_polish(core, min(deadline - 10.0, time.time() + 2.0))
    except Exception:
        pass

    base_core = core.copy()
    core_val = _max_cos(base_core)

    # growth function
    def grow_once(seed, core_init, polish_per_add=0.35):
        rng = np.random.default_rng(seed)
        X = core_init.copy()
        # small perturbation to diversify basins (except seed 0)
        if seed != 1000:
            sig = 0.005 + 0.015 * ((seed % 5) / 4.0)
            X = X + sig * rng.standard_normal(X.shape)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
            # quick re-polish after kick if time allows
            X = _lp_polish(X, time.time() + 0.3)
        inter138 = None
        inter138_val = 1e9
        # grow 109..142
        for nn in range(X.shape[0] + 1, 143):
            if time.time() > deadline - 8.0:
                break
            cand, smax, part = _find_hole_candidates(X, rng, n_sample=50000)
            # randomize among five best
            pick = int(rng.integers(0, len(part)))
            v0 = cand[part[pick]].copy()
            v0 = _refine_hole(v0, X, steps=250, beta=120.0, lr=0.10)
            X = np.vstack([X, v0.reshape(1, -1)])
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
            # LP polish after every addition
            try:
                X = _lp_polish(X, time.time() + polish_per_add)
            except Exception:
                pass
            if X.shape[0] == 138:
                inter138 = X.copy()
                inter138_val = _max_cos(X)
        return X, inter138, inter138_val

    run_id = 0
    seeds = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
             1010, 1011, 1012, 1013, 1014]
    for sd in seeds:
        if time.time() > deadline - 8.0:
            break
        # alternate core perturbation source: every other run start from base_core directly
        try:
            X142, X138m, v138m = grow_once(sd, base_core, polish_per_add=0.35)
        except Exception:
            continue
        # check intermediate 138
        if X138m is not None:
            try:
                # extra short polish of intermediate
                if time.time() < deadline - 6.0:
                    X138m = _lp_polish(X138m, time.time() + 0.6)
                    v138m = _max_cos(X138m)
                if v138m < best_val - 1e-12:
                    best_val = v138m
                    best = X138m.copy()
            except Exception:
                pass
        # prune 142 -> 138
        if X142.shape[0] >= 138:
            try:
                # if growth stopped early (<142), only handle if >=138
                if X142.shape[0] > 138:
                    Xp = _prune_to(X142, 138)
                else:
                    Xp = X142.copy()
                Xp /= (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-30)
                if time.time() < deadline - 5.0:
                    Xp = _lp_polish(Xp, time.time() + 0.8)
                vp = _max_cos(Xp)
                if vp < best_val - 1e-12:
                    best_val = vp
                    best = Xp.copy()
            except Exception:
                pass
        run_id += 1

    # final tight polish of incumbent
    try:
        if time.time() < deadline:
            best = _lp_polish(best, min(deadline, time.time() + 4.0))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

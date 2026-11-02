import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _contacts(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G >= m - tol
    # per-point counts (each pair counted once per endpoint)
    cnt = C.sum(axis=1)
    return m, cnt


def _lp_polish(x, deadline, max_solves=100):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
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
        solves += 1
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
    return x, best


def _smooth_anneal_all(x, steps, T0, T1, lrfac):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lrfac * T
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _moved_target_anneal(y, moved, t_target, steps, T0, T1, lrfac):
    moved = np.asarray(moved, dtype=int)
    m = len(moved)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = y @ y.T
        Gm = G[moved]  # m x n
        E = np.exp(np.clip((Gm - t_target) / T, -20.0, 20.0))
        for k, idx in enumerate(moved):
            E[k, idx] = 0.0
        grad_m = E @ y
        ym = y[moved]
        dot = np.sum(grad_m * ym, axis=1, keepdims=True)
        gt = grad_m - dot * ym
        row = E.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lrfac * T
        ym = ym - lr * gt
        ym /= np.linalg.norm(ym, axis=1, keepdims=True)
        y[moved] = ym
    return y


def _find_void(x_keep, rng, ns=20000):
    R = rng.standard_normal(size=(ns, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ x_keep.T
    mm = S.max(axis=1)
    bi = int(np.argmin(mm))
    v = R[bi].copy()
    best_m = float(mm[bi])
    sigma = 0.05
    for _ in range(6):
        C = v + sigma * rng.standard_normal(size=(2000, 4))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        s = C @ x_keep.T
        m2 = s.max(axis=1)
        j = int(np.argmin(m2))
        if float(m2[j]) < best_m:
            best_m = float(m2[j])
            v = C[j].copy()
        sigma *= 0.6
    # few projected softmax steps vs keep set only
    for _ in range(40):
        T = 0.003
        s = x_keep @ v  # K
        mx = s.max()
        w = np.exp((s - mx) / T)
        g = w @ x_keep
        g = g - np.dot(g, v) * v
        g /= (w.sum() + 1e-12)
        v = v - 0.004 * g
        v /= np.linalg.norm(v)
    return v


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    explore_end = t0 + 138.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])

    patch_ks = [7, 11, 15]
    deltas = [1e-4, 3e-4, 1e-3]

    trial = 0
    # precompute contacts of best for hub/rattler
    while time.time() < explore_end:
        trial += 1
        if time.time() > explore_end - 3.0:
            break
        _, cnt = _contacts(best_x, tol=1e-7)
        rattler = int(np.argmin(cnt))
        hub = int(np.argmax(cnt))
        # cycle centres: rattler first, then random, then hub
        mod = trial % 3
        if trial == 1:
            centre = rattler
        elif mod == 2:
            centre = int(rng.integers(0, 153))
        elif mod == 0:
            centre = hub
        else:
            # random point (mod==1, trial>1): mix rattler/random
            centre = int(rng.integers(0, 153)) if (trial % 2 == 0) else rattler

        k = patch_ks[(trial - 1) % 3]
        delta = deltas[((trial - 1) // 3) % 3]
        t_target = min(best_val, 0.84910235828) - delta

        # patch = centre + k largest-cosine neighbours
        sims = best_x @ best_x[centre]
        sims[centre] = -2.0
        nbrs = np.argsort(-sims)[:k]
        patch = np.concatenate([[centre], nbrs]).astype(int)
        # randomize insertion order slightly
        order = rng.permutation(len(patch))
        patch_ordered = patch[order]

        base = best_x
        keep_mask = np.ones(153, dtype=bool)
        keep_mask[patch] = False
        cur_set = base[keep_mask].copy()

        # reinsert one at a time at largest voids
        new_pts = []
        for _ in range(len(patch_ordered)):
            v = _find_void(cur_set, rng, ns=20000)
            cur_set = np.vstack([cur_set, v / np.linalg.norm(v)])
            new_pts.append(cur_set[-1].copy())
            if time.time() > explore_end:
                break
        y = base.copy()
        # map new_pts back to patch indices (patch_ordered order matches insertion)
        for pi, pt in zip(patch_ordered, new_pts):
            y[pi] = pt
        y /= np.linalg.norm(y, axis=1, keepdims=True)

        # optimize only reinserted points with target penalty
        # anneal params depend on delta: larger delta -> hotter/longer
        if delta >= 9e-4:
            y = _moved_target_anneal(y, patch, t_target, steps=380, T0=0.010, T1=0.0012, lrfac=1.3)
        elif delta >= 2e-4:
            y = _moved_target_anneal(y, patch, t_target, steps=340, T0=0.008, T1=0.0010, lrfac=1.2)
        else:
            y = _moved_target_anneal(y, patch, t_target, steps=300, T0=0.006, T1=0.0008, lrfac=1.1)

        # unfreeze all for short soft-max anneal
        y = _smooth_anneal_all(y, steps=160, T0=0.004, T1=0.0012, lrfac=1.0)
        v = _max_cos(y)

        # only LP-polish if within 0.002 (changed graph candidate) or better
        if v < best_val + 0.002:
            if time.time() > deadline - 12.0:
                break
            y2, v2 = _lp_polish(y, min(deadline - 8.0, time.time() + 1.6), max_solves=18)
            if v2 < best_val - 1e-12:
                best_x, best_val = y2, v2
                continue
            # extra all-free anneal since graph changed usefully
            if v2 < best_val + 0.002:
                # one extra all-free anneal
                y3 = _smooth_anneal_all(y2.copy(), steps=380, T0=0.010, T1=0.0015, lrfac=1.2)
                y3 = _smooth_anneal_all(y3, steps=120, T0=0.003, T1=0.0010, lrfac=1.0)
                v3 = _max_cos(y3)
                if v3 < best_val + 0.001:
                    if time.time() < deadline - 10.0:
                        y4, v4 = _lp_polish(y3, min(deadline - 8.0, time.time() + 1.6), max_solves=18)
                        if v4 < best_val - 1e-12:
                            best_x, best_val = y4, v4
                            continue
                        # keep wandering state if it improved over y2
                        if v4 < v2 and v4 < best_val + 0.002:
                            pass
            # small chance: keep near-best as base via perturbation? we always base on best,
            # so nothing to do
        # loop continues; base stays best_x (large-neighbourhood search)

    if time.time() < deadline - 1.0:
        best_x, best_val = _lp_polish(best_x, deadline - 0.5, max_solves=200)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

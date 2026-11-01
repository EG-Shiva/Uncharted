import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

_N = 173
_D = 5
_IU0 = None
_IU1 = None

def _iu():
    global _IU0, _IU1
    if _IU0 is None:
        _IU0, _IU1 = np.triu_indices(_N, 1)
    return _IU0, _IU1

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _void_fill(rem, k, rng, n_cand=3000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s_max = (cand @ rem.T).max(axis=1)
    chosen = []
    for _ in range(k):
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        chosen.append(pt)
        upd = cand @ pt
        np.maximum(s_max, upd, out=s_max)
        s_max[idx] = 2.0
    return np.stack(chosen, axis=0)

def _repel(x, alphas, iters_list, lrs, mu, decay):
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for _ in range(int(iters)):
            G = x @ x.T
            W = np.exp(alpha * (G - 0.75))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-300
            A = (W @ x) / rs
            dot = (A * x).sum(axis=1, keepdims=True)
            Ft = A - dot * x
            v = mu * v + (1.0 - mu) * Ft
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            lr *= decay
    return x

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = _iu()
    # use only if n matches
    if n != _N:
        iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
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

def _contact_edges(x, thresh=1e-7):
    iu0, iu1 = _iu()
    G = x @ x.T
    c = G[iu0, iu1]
    m = float(c.max())
    return (m - c) < thresh

def _quick_repel(x_try, rng):
    a1 = 30.0 + rng.rand() * 20.0
    a2 = 80.0 + rng.rand() * 60.0
    a3 = 180.0 + rng.rand() * 140.0
    it1, it2, it3 = 150, 200, 250
    l1 = 0.06 + rng.rand() * 0.10
    l2 = 0.03 + rng.rand() * 0.05
    l3 = 0.015 + rng.rand() * 0.025
    mu = 0.5 + rng.rand() * 0.35
    decay = 0.997 + rng.rand() * 0.002
    if decay > 0.9995:
        decay = 0.9995
    return _repel(x_try, [a1, a2, a3], [it1, it2, it3], [l1, l2, l3], mu, decay)

def _mutate(parent_x, rng, n=173, d=5):
    x_try = parent_x.copy()
    r = rng.rand()
    if r < 0.55:
        k = int(rng.randint(2, 9))  # 2..8
        G = x_try @ x_try.T
        np.fill_diagonal(G, -2.0)
        mi = G.max(axis=1)
        order = np.argsort(-mi)
        pool = order[:30]
        sel = rng.choice(pool, size=k, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[sel] = False
        rem = x_try[mask]
        try:
            new_pts = _void_fill(rem, k, rng, n_cand=3000)
        except Exception:
            new_pts = rng.standard_normal((k, d))
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        x_try[sel] = new_pts
        sigma = 0.005 + 0.015 * rng.rand()
        x_try = x_try + sigma * rng.standard_normal(x_try.shape)
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
    elif r < 0.85:
        sigma = 0.04 + 0.20 * rng.rand()
        if rng.rand() < 0.15:
            sigma = 0.30 + 0.20 * rng.rand()
        x_try = x_try + sigma * rng.standard_normal(x_try.shape)
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
    else:
        # patch restart: replace a cap region
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        proj = x_try @ u
        k = int(rng.randint(6, 18))
        order = np.argsort(-proj)  # top cap
        sel = order[:k]
        mask = np.ones(n, dtype=bool)
        mask[sel] = False
        rem = x_try[mask]
        try:
            new_pts = _void_fill(rem, k, rng, n_cand=4000)
        except Exception:
            new_pts = rng.standard_normal((k, d))
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        x_try[sel] = new_pts
        sigma = 0.01 * rng.rand()
        x_try = x_try + sigma * rng.standard_normal(x_try.shape)
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
    return x_try

def _recombine(A, B, rng):
    # orthogonal Procrustes: align B to A
    try:
        M = B.T @ A  # 5x5
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            # keep orthogonal (allow reflection, still valid isometry of sphere)
            pass
        Ba = B @ R
    except Exception:
        Ba = B.copy()
    # match
    try:
        D = np.linalg.norm(A[:, None, :] - Ba[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(D)
        Br = np.empty_like(Ba)
        # ri should be 0..n-1
        Br[ri] = Ba[ci]
    except Exception:
        Br = Ba.copy()
    n, d = A.shape
    child = np.empty_like(A)
    r = rng.rand()
    if r < 0.6:
        # halfspace splice
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u) + 1e-300
        pa = A @ u
        # random threshold to get ~30-70% from A
        q = rng.rand() * 0.6 + 0.2
        thr = np.quantile(pa, q)
        takeA = pa > thr
        if rng.rand() < 0.5:
            takeA = ~takeA
        child[takeA] = A[takeA]
        child[~takeA] = Br[~takeA]
    elif r < 0.85:
        # ball splice around random center
        cidx = int(rng.randint(0, n))
        cpt = A[cidx]
        dists = np.linalg.norm(A - cpt, axis=1)
        k = int(rng.randint(n // 4, 3 * n // 4))
        order = np.argsort(dists)
        takeB = np.zeros(n, dtype=bool)
        takeB[order[:k]] = True
        if rng.rand() < 0.5:
            child[takeB] = Br[takeB]
            child[~takeB] = A[~takeB]
        else:
            child[takeB] = A[takeB]
            child[~takeB] = Br[~takeB]
    else:
        # random mask
        m = rng.rand(n) < 0.5
        child[m] = A[m]
        child[~m] = Br[~m]
    # small jitter to break exact ties at seams
    sigma = 0.003 + 0.01 * rng.rand()
    child = child + sigma * rng.standard_normal(child.shape)
    child /= np.linalg.norm(child, axis=1, keepdims=True)
    return child

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    deadline_search = deadline - 13.0
    rng = np.random.RandomState(7)
    n, d = _N, _D
    POOL_N = 14

    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape != (n, d):
            xb = None
        else:
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape != (n, d):
            xr = None
        else:
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    except Exception:
        xr = None

    base = xb if xb is not None else (xr if xr is not None else None)
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)

    pool_x = []
    pool_v = []
    pool_e = []

    def pool_best_idx():
        return int(np.argmin(pool_v))

    def pool_worst_idx():
        return int(np.argmax(pool_v))

    # ---- initial pool ----
    init_starts = []
    if xb is not None:
        init_starts.append(("base", xb.copy()))
    if xr is not None:
        # avoid duplicate
        if xb is None or _max_cos(xb - xr) > 1e-12 or True:
            # check difference
            try:
                diff = float(np.abs(xb - xr).max()) if xb is not None else 1.0
            except Exception:
                diff = 1.0
            if xb is None or diff > 1e-8:
                init_starts.append(("base2", xr.copy()))
    # fill remaining with perturbed / random
    while len(init_starts) < POOL_N:
        k = len(init_starts)
        if k % 3 == 0:
            sig = 0.25 + 0.25 * rng.rand()
            x0 = base + sig * rng.standard_normal(base.shape)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            init_starts.append(("kick", x0))
        elif k % 3 == 1:
            # void rebuild of worst points
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            mi = G.max(axis=1)
            order = np.argsort(-mi)
            kk = int(rng.randint(8, 20))
            sel = order[:kk]
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            rem = base[mask]
            try:
                new_pts = _void_fill(rem, kk, rng, n_cand=3000)
            except Exception:
                new_pts = rng.standard_normal((kk, d))
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x0 = base.copy()
            x0[sel] = new_pts
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            init_starts.append(("void", x0))
        else:
            x0 = rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            init_starts.append(("rand", x0))

    for _, x_init in init_starts:
        if time.time() > deadline_search - 20.0:
            break
        try:
            x_opt = _quick_repel(x_init.copy(), rng)
            # short LP
            x_opt = _lp_polish(x_opt, 1.0)
        except Exception:
            x_opt = x_init.copy()
        try:
            vv = _max_cos(x_opt)
            ee = _contact_edges(x_opt, thresh=1e-7)
        except Exception:
            continue
        pool_x.append(x_opt)
        pool_v.append(vv)
        pool_e.append(ee)
        # keep pool bounded during init (should equal POOL_N)
    # trim to POOL_N best-diverse? just keep all, then enforce size
    if len(pool_x) > POOL_N:
        order = np.argsort(np.array(pool_v))
        pool_x = [pool_x[i] for i in order[:POOL_N]]
        pool_v = [pool_v[i] for i in order[:POOL_N]]
        pool_e = [pool_e[i] for i in order[:POOL_N]]

    if len(pool_x) == 0:
        # fallback
        x0 = base.copy()
        pool_x = [x0]
        pool_v = [_max_cos(x0)]
        pool_e = [_contact_edges(x0)]

    t_start_anneal = time.time()
    total_anneal = max(deadline_search - t_start_anneal, 1.0)
    it = 0
    while time.time() < deadline_search:
        it += 1
        frac = (time.time() - t_start_anneal) / total_anneal
        if frac < 0:
            frac = 0.0
        if frac > 1:
            frac = 1.0
        T = 2e-4 * (0.005 ** frac)  # 2e-4 -> 1e-6
        # select parents
        nb = len(pool_x)
        # bias to best sometimes
        if rng.rand() < 0.35:
            p1 = pool_best_idx()
        else:
            p1 = int(rng.randint(0, nb))
        if rng.rand() < 0.55:
            # mutation
            parent = pool_x[p1]
            x_try = _mutate(parent, rng, n, d)
        else:
            # recombination
            p2 = int(rng.randint(0, nb))
            if nb > 1:
                while p2 == p1:
                    p2 = int(rng.randint(0, nb))
            A = pool_x[p1]
            B = pool_x[p2]
            # random order
            if rng.rand() < 0.5:
                A, B = B, A
            try:
                x_try = _recombine(A, B, rng)
            except Exception:
                x_try = _mutate(pool_x[p1], rng, n, d)
        # anneal briefly
        try:
            x_opt = _quick_repel(x_try, rng)
            # shorter LP during search to save time; longer occasionally
            lp_t = 0.9 if (it % 4 != 0) else 1.6
            # reduce if little time left
            if deadline_search - time.time() < 5:
                lp_t = 0.4
            x_opt = _lp_polish(x_opt, lp_t)
            vv = _max_cos(x_opt)
            ee = _contact_edges(x_opt, thresh=1e-7)
        except Exception:
            continue
        # diversity insertion
        try:
            # compute hamming distances
            dists = [int(np.count_nonzero(ee ^ pe)) for pe in pool_e]
            dmin = min(dists) if dists else 10**9
            imin = int(np.argmin(dists)) if dists else 0
            if len(pool_x) < POOL_N:
                if dmin < 20:
                    # only if beats that member
                    if vv < pool_v[imin] - 1e-12:
                        pool_x[imin] = x_opt
                        pool_v[imin] = vv
                        pool_e[imin] = ee
                else:
                    pool_x.append(x_opt)
                    pool_v.append(vv)
                    pool_e.append(ee)
            else:
                if dmin < 20:
                    if vv < pool_v[imin] - 1e-12:
                        pool_x[imin] = x_opt
                        pool_v[imin] = vv
                        pool_e[imin] = ee
                else:
                    iw = pool_worst_idx()
                    if vv < pool_v[iw]:
                        pool_x[iw] = x_opt
                        pool_v[iw] = vv
                        pool_e[iw] = ee
                    else:
                        bv = float(min(pool_v))
                        # Metropolis accept worse
                        try:
                            prob = np.exp(-(vv - bv) / max(T, 1e-12))
                        except Exception:
                            prob = 0.0
                        if prob > 1.0:
                            prob = 1.0
                        if rng.rand() < prob * 0.9:  # slight damping
                            pool_x[iw] = x_opt
                            pool_v[iw] = vv
                            pool_e[iw] = ee
        except Exception:
            continue
        # early exit if big win (unlikely)
        if min(pool_v) < 0.74:
            break

    # final: best member + polish
    try:
        bi = int(np.argmin(pool_v))
        best = pool_x[bi].copy()
    except Exception:
        best = base.copy()
    remain = deadline - time.time() - 1.5
    if remain < 2.0:
        remain = 2.0
    if remain > 14.0:
        remain = 14.0
    try:
        best = _lp_polish(best, remain)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

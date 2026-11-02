import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rigidity_stats(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    # near contacts within 1e-3 of max
    thr = m - 1e-3
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    near = int(np.count_nonzero(c > thr))
    # per-point degrees
    n = X.shape[0]
    deg = np.zeros(n, dtype=int)
    # thresholded adjacency
    # build mask
    Gm = G > thr
    # Gm includes symmetric; sum rows (diag is -2 so not counted)
    deg = np.count_nonzero(Gm, axis=1)
    deg_sorted = np.sort(deg)
    nratt = int(np.count_nonzero(deg == 0))
    # spectrum
    Gn = X @ X.T
    try:
        ev = np.linalg.eigvalsh(Gn)
        ev = np.sort(ev)[::-1]
    except Exception:
        ev = np.zeros(n)
    return m, near, nratt, deg_sorted, ev, G

def _quench_adam(X0, stages, lr_base=0.008, check_time=None, t_end=None):
    # stages: list of (T,p,steps)
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for si, (T, p, steps) in enumerate(stages):
        lr = lr_base * (0.7 ** si)
        for _ in range(steps):
            t += 1
            G = X @ X.T
            np.fill_diagonal(G, 0.0)
            # hinge power
            D = G - T
            mask = D > 0
            if not np.any(mask):
                # nothing to do, small random nudge to keep moving
                continue
            # W = p * D^{p-1} where mask
            if p == 2:
                W = np.zeros_like(G)
                W[mask] = 2.0 * D[mask]
            else:
                W = np.zeros_like(G)
                W[mask] = float(p) * np.power(D[mask], float(p - 1))
            # gradient
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            # adam update
            m1 = b1 * m1 + (1 - b1) * Gt
            v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
            c1 = 1 - b1 ** t
            c2 = 1 - b2 ** t
            mh = m1 / c1
            vh = v1 / c2
            upd = mh / (np.sqrt(vh) + eps)
            X = X - lr * upd
            X = _norm(X)
            if (t % 100) == 0 and t_end is not None:
                if time.time() > t_end:
                    return X
    return X

def _lp_polish(X, t_end, radius_init=3e-4, cap=1500, max_it=10):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    Xbest = X.copy()
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 5e-8)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        C = np.empty((p, 2 * d + 1), dtype=int)
        C[:, :d] = I[:, None] * d + cd
        C[:, d:2 * d] = J[:, None] * d + cd
        C[:, 2 * d] = n * d
        cols = C.ravel()
        M = np.empty((p, 2 * d + 1))
        M[:, :d] = X[J]
        M[:, d:2 * d] = X[I]
        M[:, 2 * d] = -1.0
        vals = M.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            Xbest = Y.copy()
            radius = min(radius * 1.5, 5e-3)
        else:
            radius *= 0.5
    return Xbest

def solve():
    t0 = time.time()
    HARD = t0 + 132.0
    n, d = 82, 14
    rng = np.random.default_rng(12345)
    # fallback
    Xfb = None; vfb = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < vfb:
                        vfb = v; Xfb = a.copy()
        except Exception:
            pass
    if Xfb is None:
        Xfb = _norm(rng.standard_normal((n, d)))
        vfb = _max_cos(Xfb)
    Xbest_overall = Xfb.copy(); vbest = vfb

    # ---- diverse seeds ----
    seeds = []
    # 6 random
    for _ in range(6):
        seeds.append(_norm(rng.standard_normal((n, d))))
    # 3 greedy far-hole
    for _ in range(3):
        if time.time() > HARD - 40:
            break
        Xg = np.zeros((n, d))
        Xg[0] = _norm(rng.standard_normal((1, d)))[0]
        for k in range(1, n):
            cand = _norm(rng.standard_normal((500, d)))
            dots = cand @ Xg[:k].T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            Xg[k] = cand[bi]
        seeds.append(Xg)
        if time.time() > HARD - 30:
            break
    # 3 perturbed lattice-like (hypercube vertices + noise)
    for _ in range(3):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, d)) / np.sqrt(float(d))
        noise = 0.15 * rng.standard_normal((n, d))
        seeds.append(_norm(signs + noise))

    stages_full = [(0.205, 2, 300), (0.200, 3, 300), (0.197, 4, 300), (0.194, 6, 250)]
    pool = []  # (X,m,near,nratt,deg,ev)
    for si, s0 in enumerate(seeds):
        if time.time() > HARD - 35:
            break
        # per-seed time cap ~4s
        seed_end = min(HARD - 30, time.time() + 4.0)
        Xq = _quench_adam(s0, stages_full, lr_base=0.008, t_end=seed_end)
        m, near, nratt, deg_sorted, ev, _ = _rigidity_stats(Xq)
        pool.append([Xq, m, near, nratt, deg_sorted, ev])
        if m < vbest - 1e-12:
            vbest = m; Xbest_overall = Xq.copy()

    if len(pool) == 0:
        return _norm(Xbest_overall)

    # ---- diversity archive (up to 6) ----
    # score for jammed: prefer high near, low rattlers, with max penalty
    def jam_score(e):
        # lower better
        return e[1] - 2e-6 * float(e[2]) + 1e-5 * float(e[3])
    # sort by jam_score and take diverse
    pool_sorted = sorted(pool, key=jam_score)
    archive = []
    for e in pool_sorted:
        # check diversity vs archive: spectrum dist
        keep = True
        for a in archive:
            ds = float(np.linalg.norm(e[5] - a[5]))
            dd = float(np.linalg.norm(e[4].astype(float) - a[4].astype(float)))
            if ds < 0.05 and dd < 3.0:
                keep = False
                break
        if keep:
            archive.append(e)
        if len(archive) >= 6:
            break
    if len(archive) == 0:
        archive = pool_sorted[:6]
    # ensure best-max included
    pool_by_max = sorted(pool, key=lambda e: e[1])
    if pool_by_max[0] not in archive:
        if len(archive) >= 6:
            archive[-1] = pool_by_max[0]
        else:
            archive.append(pool_by_max[0])

    # ---- crossover of archive members with different fingerprints ----
    # Procrustes alignment + point exchange, then short requench
    stages_short = [(0.200, 3, 150), (0.197, 4, 150), (0.194, 6, 150)]
    # pairwise
    na = len(archive)
    for ii in range(na):
        for jj in range(ii + 1, na):
            if time.time() > HARD - 28:
                break
            A = archive[ii][0]; B = archive[jj][0]
            evA = archive[ii][5]; evB = archive[jj][5]
            ds = float(np.linalg.norm(evA - evB))
            dd = float(np.linalg.norm(archive[ii][4].astype(float) - archive[jj][4].astype(float)))
            if ds < 0.02 and dd < 2.0:
                continue  # same fingerprint, skip
            # Procrustes: align B to A assuming current order
            try:
                Cmat = B.T @ A
                U, _, Vt = np.linalg.svd(Cmat)
                Q = U @ Vt
                Balt = B @ Q
            except Exception:
                Balt = B
            # point exchange: half from A, half from Balt by index mask
            mask = rng.random(n) < 0.5
            Ch = A.copy()
            Ch[~mask] = Balt[~mask]
            Ch = _norm(Ch)
            # small perturbation to break symmetry
            Ch = _norm(Ch + 1e-3 * rng.standard_normal((n, d)))
            child_end = min(HARD - 25, time.time() + 3.0)
            Chq = _quench_adam(Ch, stages_short, lr_base=0.006, t_end=child_end)
            m, near, nratt, deg_sorted, ev, _ = _rigidity_stats(Chq)
            pool.append([Chq, m, near, nratt, deg_sorted, ev])
            if m < vbest - 1e-12:
                vbest = m; Xbest_overall = Chq.copy()
        if time.time() > HARD - 28:
            break

    # ---- pick single most jammed candidate for final LP polish ----
    # filter within 0.008 of best scratch max
    if len(pool) > 0:
        best_pool_max = min(e[1] for e in pool)
        cands = [e for e in pool if e[1] < best_pool_max + 0.008]
        if len(cands) == 0:
            cands = pool
        # most jammed = max near, tie break lower max and fewer rattlers
        cands.sort(key=lambda e: (-e[2], e[1], e[3]))
        Xjam = cands[0][0]
    else:
        Xjam = Xbest_overall

    # ---- final 20s LP polish ----
    remain = HARD - time.time()
    if remain > 4:
        polish_time = min(remain - 1.0, 18.0)
        t_end = time.time() + polish_time
        try:
            Xp = _lp_polish(Xjam, t_end, radius_init=3e-4, cap=1500, max_it=10)
            vp = _max_cos(Xp)
            if vp < vbest:
                vbest = vp; Xbest_overall = Xp.copy()
        except Exception:
            pass

    # fallback safety
    if not np.all(np.isfinite(Xbest_overall)):
        Xbest_overall = Xfb.copy()
    Xbest_overall = _norm(np.asarray(Xbest_overall, dtype=float))
    return Xbest_overall

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0.02:
        return X, best
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if deadline is not None and now >= deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        ns = int(sel.sum())
        if ns == 0:
            radius *= 0.5
            continue
        if ns > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def _contact_info(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G >= m - tol
    cnt = mask.sum(axis=1)
    return m, cnt, mask, G

def _orbit_classes(X, tol=1e-9):
    m, cnt, mask, G = _contact_info(X, tol)
    n = X.shape[0]
    deg = cnt.astype(int)
    keys = []
    for i in range(n):
        nb = np.where(mask[i])[0]
        nd = tuple(sorted(deg[nb].tolist())) if len(nb) > 0 else ()
        # include count and neighbour-degree multiset; also quantised self max
        keys.append((int(cnt[i]), nd))
    uniq = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, []).append(i)
    groups = [np.array(v, dtype=np.int64) for v in uniq.values()]
    # sort by size descending for cycling
    groups.sort(key=lambda a: -len(a))
    return groups, m, cnt

def _communities_greedy(X, tol=1e-9, target=10):
    m, cnt, mask, G = _contact_info(X, tol)
    n = X.shape[0]
    # adjacency (symmetric, no self); mask includes both triangles + symmetric
    A = mask.astype(np.int64)
    # ensure symmetric
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    tot = float(A.sum())
    if tot <= 0:
        # no contacts: random partition
        return [np.arange(n, dtype=np.int64)]
    two_m = tot  # sum of degrees = 2m
    k = A.sum(axis=1).astype(float)
    comms = [{i} for i in range(n)]
    # greedy merge until target or no positive gain
    while len(comms) > target:
        C = len(comms)
        # precompute a_c
        a = np.array([k[list(s)].sum() / two_m for s in comms])
        best_gain = 0.0
        best_pair = None
        # brute force pairs
        for ii in range(C):
            si = comms[ii]
            # adjacency of si to others: for speed use boolean vectors
            for jj in range(ii + 1, C):
                sj = comms[jj]
                # count edges between si and sj
                # use A submatrix sum
                li = list(si); lj = list(sj)
                e = float(A[np.ix_(li, lj)].sum()) / two_m
                gain = 2.0 * (e - a[ii] * a[jj])
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (ii, jj)
        if best_pair is None or best_gain <= 1e-12:
            break
        ii, jj = best_pair
        comms[ii] = comms[ii] | comms[jj]
        del comms[jj]
    groups = [np.array(sorted(s), dtype=np.int64) for s in comms]
    groups.sort(key=lambda a: -len(a))
    return groups

def _random_plane(rng):
    # two random orthonormal vectors in R5
    A = rng.normal(size=(5, 2))
    Q, _ = np.linalg.qr(A)
    return Q[:, 0], Q[:, 1]

def _apply_rotation(X, idx, p, q, phi):
    Y = X.copy()
    c = np.cos(phi) - 1.0
    s = np.sin(phi)
    # R = I + c(pp'+qq') + s(qp' - pq')
    # y = R x = x + c((x.p)p + (x.q)q) + s((x.p)q - (x.q)p)
    Xs = X[idx]
    dp = Xs @ p
    dq = Xs @ q
    Y[idx] = Xs + c * (dp[:, None] * p[None, :] + dq[:, None] * q[None, :]) + s * (dp[:, None] * q[None, :] - dq[:, None] * p[None, :])
    nrm = np.linalg.norm(Y[idx], axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Y[idx] = Y[idx] / nrm
    return Y

def _apply_tangent(X, idx, g, alpha):
    Y = X.copy()
    Xs = X[idx]
    dot = Xs @ g
    T = g[None, :] - dot[:, None] * Xs
    Y[idx] = Xs + alpha * T
    nrm = np.linalg.norm(Y[idx], axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Y[idx] = Y[idx] / nrm
    return Y

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(20261101)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                if not np.all(np.isfinite(an)):
                    continue
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # initial groups from record basin
    orbits, _, _ = _orbit_classes(best)
    try:
        comms = _communities_greedy(best, target=10)
    except Exception:
        comms = [np.arange(n, dtype=np.int64)]
    # Build combined group list: tag 0=orbit, 1=community
    def build_groups():
        glist = []
        for g in orbits:
            glist.append((0, g))
        for g in comms:
            if len(g) >= 3 and len(g) <= 120:
                glist.append((1, g))
        return glist
    groups = build_groups()
    if len(groups) == 0:
        groups = [(0, np.arange(n, dtype=np.int64))]
    tabu = set()
    trial = 0
    # reserve final polish time
    # main loop
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        # quick polish budget
        pt = 1.2
        if rem < 25:
            pt = max(0.5, rem - 8.0)
        if trial > 120:
            pt = min(pt, 0.8)
        # cycle over groups and moves
        if len(groups) == 0:
            break
        gi = trial % len(groups)
        kind, gidx = groups[gi]
        move = (trial // max(1, len(groups))) % 4  # 0 rot-class,1 tang-class,2 rot-complement/break,3 tang-break
        key = (int(gi), int(move), int(kind), int(len(gidx)))
        if key in tabu and (trial % 7) != 0:
            trial += 1
            if trial > 400:
                break
            continue
        amp = float(rng.uniform(0.02, 0.15))
        Y = best.copy()
        try:
            if move == 0:
                p, q = _random_plane(rng)
                Y = _apply_rotation(best, gidx, p, q, amp)
            elif move == 1:
                gv = rng.normal(size=(d,)); gv /= np.linalg.norm(gv)
                Y = _apply_tangent(best, gidx, gv, amp)
            elif move == 2:
                # break-one-class: split class in two, move one half; if size 1 move complement slightly
                if len(gidx) > 1:
                    perm = rng.permutation(gidx)
                    half = len(perm) // 2
                    sub = perm[:max(1, half)]
                    p, q = _random_plane(rng)
                    Y = _apply_rotation(best, sub, p, q, amp)
                    # tiny individual jitter inside moved half to break symmetry
                    Y[sub] += 0.005 * rng.normal(size=(len(sub), d))
                    Y[sub] /= np.linalg.norm(Y[sub], axis=1, keepdims=True)
                else:
                    # preserve-one-class: rotate everything except gidx
                    allidx = np.ones(n, dtype=bool); allidx[gidx] = False
                    comp = np.where(allidx)[0]
                    p, q = _random_plane(rng)
                    Y = _apply_rotation(best, comp, p, q, amp * 0.5)
            else:
                if len(gidx) > 1:
                    perm = rng.permutation(gidx)
                    half = len(perm) // 2
                    sub = perm[:max(1, half)]
                    gv = rng.normal(size=(d,)); gv /= np.linalg.norm(gv)
                    Y = _apply_tangent(best, sub, gv, amp)
                    Y[sub] += 0.005 * rng.normal(size=(len(sub), d))
                    Y[sub] /= np.linalg.norm(Y[sub], axis=1, keepdims=True)
                else:
                    gv = rng.normal(size=(d,)); gv /= np.linalg.norm(gv)
                    allidx = np.ones(n, dtype=bool); allidx[gidx] = False
                    comp = np.where(allidx)[0]
                    Y = _apply_tangent(best, comp, gv, amp * 0.5)
        except Exception:
            trial += 1
            continue
        # polish candidate
        try:
            Yp, vp = _lp_polish(Y, pt, deadline - 7.0)
        except Exception:
            trial += 1
            continue
        if vp < best_val - 1e-12:
            best, best_val = Yp, vp
            # recompute topology
            try:
                orbits, _, _ = _orbit_classes(best)
                comms = _communities_greedy(best, target=10)
                groups = build_groups()
            except Exception:
                pass
            tabu.clear()
        else:
            tabu.add(key)
            # keep tabu bounded
            if len(tabu) > 300:
                # drop arbitrary
                tabu.pop()
        trial += 1
        if trial > 500:
            break
    # final polish of best with remaining time
    rem = deadline - time.time()
    if rem > 2.0:
        try:
            best, best_val = _lp_polish(best, min(rem - 1.0, 25.0), deadline - 0.5)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # tiny shrink not needed; ensure finite
    best[~np.isfinite(best)] = 0.2
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

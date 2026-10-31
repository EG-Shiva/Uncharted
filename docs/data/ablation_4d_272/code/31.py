import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def row_max_cnt(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    m = float(rm.max())
    cnt = np.sum(G > (m - 1e-4), axis=1)
    return rm, m, cnt, G

def repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm(x)
    return x

def lp_polish(x, deadline, cap=5000):
    x = _norm(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
            cnt = cap
        if cnt == 0:
            radius *= 0.5
            continue
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def random_orthonormal_pair(rng):
    A = rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]

def global_twist(x, rng, s=None):
    # position-dependent rotation in u-v plane by angle s*(x.w)
    u, v, w, _ = random_orthonormal_pair(rng)
    if s is None:
        s = float(rng.uniform(0.25, 0.65))
    t = (x @ w) * s
    cu = np.cos(t); su = np.sin(t)
    xu = x @ u
    xv = x @ v
    # rotate (xu,xv)
    yu = cu * xu - su * xv
    yv = su * xu + cu * xv
    y = x + np.outer(yu - xu, u) + np.outer(yv - xv, v)
    return _norm(y)

def cap_rotate(x, rng, k=None, ang=None):
    n = x.shape[0]
    a = rng.standard_normal(4); a /= np.linalg.norm(a)
    dots = x @ a
    if k is None:
        k = int(rng.integers(20, 65))
    S = np.argsort(-dots)[:k]
    p = rng.standard_normal(4); p /= np.linalg.norm(p)
    q = rng.standard_normal(4)
    q = q - np.dot(q, p) * p
    q /= np.linalg.norm(q)
    if ang is None:
        ang = float(rng.uniform(0.08, 0.35)) * (1 if rng.random() < 0.5 else -1)
    c = np.cos(ang); s_ = np.sin(ang)
    # R = I + (c-1)(pp'+qq') + s(qp'-pq')
    y = x.copy()
    Xs = x[S]
    Xp = Xs @ p
    Xq = Xs @ q
    Yp = c * Xp - s_ * Xq
    Yq = s_ * Xp + c * Xq
    dP = Yp - Xp
    dQ = Yq - Xq
    y[S] = Xs + np.outer(dP, p) + np.outer(dQ, q)
    return _norm(y)

def cluster_rotate(x, Gcontact, rng, k=None, ang=None):
    n = x.shape[0]
    seed = int(rng.integers(0, n))
    if k is None:
        k = int(rng.integers(20, 60))
    # BFS on contact graph
    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    frontier = [seed]
    order = [seed]
    while len(order) < k and frontier:
        nxt = []
        for u in frontier:
            nbrs = np.where(Gcontact[u])[0]
            # shuffle-ish: sort by rng? use random order via rng permutation of nbrs
            if len(nbrs) > 1:
                perm = rng.permutation(len(nbrs))
                nbrs = nbrs[perm]
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    order.append(int(w))
                    nxt.append(int(w))
                    if len(order) >= k:
                        break
            if len(order) >= k:
                break
        frontier = nxt
        if not frontier and len(order) < k:
            # add geographically nearest to seed to fill
            dots = x @ x[seed]
            dots[visited] = -2
            cand = int(np.argmax(dots))
            visited[cand] = True
            order.append(cand)
            frontier = [cand]
    S = np.array(order, dtype=int)
    p = rng.standard_normal(4); p /= np.linalg.norm(p)
    q = rng.standard_normal(4)
    q = q - np.dot(q, p) * p
    q /= np.linalg.norm(q)
    if ang is None:
        ang = float(rng.uniform(0.08, 0.35)) * (1 if rng.random() < 0.5 else -1)
    c = np.cos(ang); s_ = np.sin(ang)
    y = x.copy()
    Xs = x[S]
    Xp = Xs @ p
    Xq = Xs @ q
    Yp = c * Xp - s_ * Xq
    Yq = s_ * Xp + c * Xq
    y[S] = Xs + np.outer(Yp - Xp, p) + np.outer(Yq - Xq, q)
    return _norm(y)

def solve():
    t0 = time.time()
    BUDGET = 170.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(20261031)
    cands = []
    try:
        bx = np.load("best.npy")
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (272, 4):
            cands.append(_norm(bx))
    except Exception:
        pass
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (272, 4):
            cands.append(_norm(rx))
    except Exception:
        pass
    if not cands:
        r = rng.standard_normal((272, 4))
        cands.append(_norm(r))
    best_x = min(cands, key=lambda z: max_cos(z)).copy()
    best_v = max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    # quick diverse pool: a few random-healed configs to seed contact diversity
    # use very little time
    try:
        for _ in range(3):
            if time.time() > t_end - 150:
                break
            r = _norm(rng.standard_normal((272, 4)))
            r = repulse_gd(r, steps=250, a0=40.0, a1=400.0, s0=0.08, s1=0.008, mu=0.85)
            v = max_cos(r)
            if v < best_v:
                best_x = r.copy(); best_v = v
                cur_x = r.copy(); cur_v = v
    except Exception:
        pass

    # reset walker to best
    cur_x = best_x.copy(); cur_v = best_v
    it = 0
    # main basin-hopping with collective moves + GD healing (no LP inside loop)
    while time.time() < t_end - 28:
        it += 1
        frac = (time.time() - t0) / max(1e-9, (t_end - 28 - t0))
        temp = 0.004 * (1 - frac) + 0.0004  # annealing tolerance
        u = rng.random()
        try:
            if u < 0.40:
                y = global_twist(cur_x, rng)
            elif u < 0.75:
                y = cap_rotate(cur_x, rng)
            else:
                # build contact adjacency for cluster move
                try:
                    rm, m, cnt, G = row_max_cnt(cur_x)
                    adj = G > (m - 2e-3)
                    np.fill_diagonal(adj, False)
                except Exception:
                    adj = None
                if adj is None:
                    y = cap_rotate(cur_x, rng)
                else:
                    y = cluster_rotate(cur_x, adj, rng)
            # small extra jitter to avoid exact symmetry trap
            y = y + 0.0015 * rng.standard_normal(y.shape)
            y = _norm(y)
            # heal: two-stage GD, first broad then sharp
            y = repulse_gd(y, steps=200, a0=50.0, a1=450.0, s0=0.05, s1=0.006, mu=0.85)
            # occasional stronger sharpening
            if it % 3 == 0:
                y = repulse_gd(y, steps=120, a0=150.0, a1=700.0, s0=0.02, s1=0.004, mu=0.8)
            v = max_cos(y)
            if v < best_v - 1e-12:
                best_v = v; best_x = y.copy()
                cur_x = y.copy(); cur_v = v
            elif v < cur_v + temp:
                cur_x = y.copy(); cur_v = v
            else:
                # with small prob restart walker at best to intensify
                if rng.random() < 0.05:
                    cur_x = best_x.copy(); cur_v = best_v
            # opportunistic fresh random injection every ~60 iters
            if it % 60 == 0 and time.time() < t_end - 60:
                r = _norm(rng.standard_normal((272, 4)))
                r = repulse_gd(r, steps=200, a0=50.0, a1=450.0, s0=0.06, s1=0.006)
                rv = max_cos(r)
                # merge: keep best half? just consider as walker if promising
                if rv < cur_v + 0.01:
                    cur_x = r.copy(); cur_v = rv
                    if rv < best_v:
                        best_v = rv; best_x = r.copy()
        except Exception:
            continue

    # final lock: LP polish best and a twisted variant, keep winner
    try:
        rem = t_end - time.time()
        if rem > 3:
            # first GD sharpen best
            best_x = repulse_gd(best_x, steps=150, a0=150.0, a1=800.0, s0=0.02, s1=0.003, mu=0.8)
            best_v = max_cos(best_x)
            # LP with most of remaining time
            deadline = t_end - 0.8
            # give 70% to best, keep 30% for runner
            mid = time.time() + (deadline - time.time()) * 0.7
            best_x = lp_polish(best_x, deadline=mid)
            best_v = max_cos(best_x)
            # try one more collective kick + heal + LP if time allows
            if time.time() < deadline - 6:
                y2 = global_twist(best_x, rng, s=0.25)
                y2 = repulse_gd(y2, steps=200, a0=80.0, a1=600.0, s0=0.04, s1=0.005)
                y2 = lp_polish(y2, deadline=deadline)
                v2 = max_cos(y2)
                if v2 < best_v:
                    best_x = y2; best_v = v2
                else:
                    # finish polishing best with leftover
                    best_x = lp_polish(best_x, deadline=deadline)
                    best_v = max_cos(best_x)
    except Exception:
        pass

    best_x = _norm(best_x)
    return best_x

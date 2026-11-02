import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 5e-3
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds or radius < 1e-14:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.5, seconds - (time.time() - start))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return bx / np.linalg.norm(bx, axis=1, keepdims=True)

def _adjacency(x, m, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    n = x.shape[0]
    iuI, iuJ = np.triu_indices(n, 1)
    sel = (G[iuI, iuJ] > m - tol)
    adj = [[] for _ in range(n)]
    I = iuI[sel]; J = iuJ[sel]
    for a, b in zip(I, J):
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj

def _bfs_cluster(adj, start, size, rng=None, randomize=False):
    n = len(adj)
    seen = set([int(start)])
    q = [int(start)]
    order = [int(start)]
    # BFS
    head = 0
    while head < len(q) and len(order) < size:
        u = q[head]; head += 1
        nbrs = adj[u]
        if randomize and rng is not None and len(nbrs) > 1:
            perm = rng.permutation(len(nbrs))
            nbrs = [nbrs[i] for i in perm]
        for w in nbrs:
            if w not in seen:
                seen.add(w)
                q.append(w)
                order.append(w)
                if len(order) >= size:
                    break
    if len(order) < size:
        # fill with random / remaining nodes to reach size
        extra = [i for i in range(n) if i not in seen]
        if rng is not None:
            perm = rng.permutation(len(extra))
            extra = [extra[i] for i in perm]
        for w in extra:
            order.append(int(w))
            if len(order) >= size:
                break
    return np.array(order[:size], dtype=np.int64)

def _maxmin_reinsert(y, cluster, rng, n_cand=50000):
    n, d = y.shape
    cset = set(int(c) for c in cluster)
    # fixed points
    fixed_idx = np.array([i for i in range(n) if i not in cset], dtype=np.int64)
    Xf = y[fixed_idx]
    # random insertion order
    order = cluster.copy()
    # shuffle order deterministically via rng permutation
    order = order[rng.permutation(len(order))]
    inserted = []
    CH = 10000
    for k in order:
        k = int(k)
        if inserted:
            Xa = np.vstack([Xf] + inserted)
        else:
            Xa = Xf
        best_pt = None
        best_w = np.inf
        # generate in chunks to limit memory but total n_cand
        for _ in range(n_cand // CH):
            cand = rng.standard_normal(size=(CH, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ Xa.T
            worst = dots.max(axis=1)
            bi = int(np.argmin(worst))
            if worst[bi] < best_w:
                best_w = float(worst[bi])
                best_pt = cand[bi].copy()
        y[k] = best_pt
        inserted.append(best_pt[None, :])
    return y

def _frozen_hillclimb(y, cluster, rng, sweeps=2, n_trial=2000, sigma=0.03):
    n, d = y.shape
    for _ in range(sweeps):
        for k in cluster:
            k = int(k)
            cur = y[k].copy()
            others = np.delete(y, k, axis=0)
            cur_w = float((others @ cur).max())
            # trials around cur
            tr = rng.standard_normal(size=(n_trial, d)) * sigma + cur[None, :]
            tr /= np.linalg.norm(tr, axis=1, keepdims=True)
            w = (tr @ others.T).max(axis=1)
            bi = int(np.argmin(w))
            if w[bi] < cur_w:
                y[k] = tr[bi]
                cur_w = float(w[bi])
                # second small refinement around new best
                tr2 = rng.standard_normal(size=(n_trial // 2, d)) * (sigma * 0.4) + y[k][None, :]
                tr2 /= np.linalg.norm(tr2, axis=1, keepdims=True)
                w2 = (tr2 @ np.delete(y, k, axis=0).T).max(axis=1)
                bj = int(np.argmin(w2))
                if w2[bj] < cur_w:
                    y[k] = tr2[bj]
    return y

def _smooth_relax(x, steps=60, step0=0.015, window=0.01):
    n, d = x.shape
    step = float(step0)
    iuI, iuJ = np.triu_indices(n, 1)
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        c = G[iuI, iuJ]
        sel = c > (m - window)
        if int(sel.sum()) == 0:
            break
        I = iuI[sel]; J = iuJ[sel]; cv = c[sel]
        w = (cv - (m - window)) / window  # (0,1]
        w = w * w  # emphasize closest
        F = np.zeros_like(x)
        # tangential components
        ci = cv[:, None]
        contrib_i = w[:, None] * (x[J] - ci * x[I])
        contrib_j = w[:, None] * (x[I] - ci * x[J])
        np.add.at(F, I, contrib_i)
        np.add.at(F, J, contrib_j)
        # normalize force scale
        fn = np.linalg.norm(F, axis=1, keepdims=True)
        # limit step per point
        F = F / np.maximum(1.0, fn * 5.0)
        x = x + step * F
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        step *= 0.995
    return x

def solve():
    t0 = time.time()
    DEADLINE = 165.0
    rng = np.random.default_rng(0)
    # load best
    best_arr = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_arr = a.copy()
        except Exception:
            pass
    if best_arr is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_arr = a
        best_val = _max_cos(a)
    base = best_arr.copy()
    # contact graph of base best
    m0 = _max_cos(base)
    adj = _adjacency(base, m0, tol=1e-7)
    deg = np.array([len(a) for a in adj], dtype=np.int64)
    hub_order = np.argsort(-deg)
    # build attempt list: hubs + random connected
    attempts = []
    for h in hub_order[:8]:
        attempts.append((int(h), False))
    # add a few random starts
    for r in range(6):
        attempts.append((int(rng.integers(0, 323)), True))
    # interleave done; sizes 12-18
    sizes = [12, 14, 16, 18, 13, 15, 17, 12, 14, 16, 18, 13, 15, 17]
    for ai, (hub, is_rand) in enumerate(attempts):
        if time.time() - t0 > DEADLINE - 8:
            break
        sz = int(sizes[ai % len(sizes)])
        # rebuild adjacency from current global best every few attempts to follow topology
        if ai % 4 == 0:
            adj = _adjacency(best_arr, _max_cos(best_arr), tol=1e-7)
            deg = np.array([len(a) for a in adj], dtype=np.int64)
            # refresh hub if not random
            if not is_rand and ai < len(hub_order):
                hub = int(hub_order[ai % len(hub_order)])
        cl = _bfs_cluster(adj, hub, sz, rng=rng, randomize=is_rand)
        y = best_arr.copy()
        # surgery: reinsert
        y = _maxmin_reinsert(y, cl, rng, n_cand=50000)
        if time.time() - t0 > DEADLINE - 7:
            # snapshot check
            v = _max_cos(y)
            if v < best_val:
                best_val = v; best_arr = (y / np.linalg.norm(y, axis=1, keepdims=True)).copy()
            break
        y = _frozen_hillclimb(y, cl, rng, sweeps=2, n_trial=2000, sigma=0.03)
        y = _smooth_relax(y, steps=50, step0=0.015, window=0.01)
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best_arr = y.copy()
        # keep candidates with pre-LP max below 0.670
        if v < 0.670:
            remain = DEADLINE - (time.time() - t0)
            if remain < 3:
                continue
            alloc = min(6.0, remain - 1.5)
            if alloc < 1.0:
                continue
            y2 = _lp_polish(y, alloc)
            v2 = _max_cos(y2)
            if v2 < best_val:
                best_val = v2
                best_arr = y2.copy()
                # update adjacency quickly
                adj = _adjacency(best_arr, best_val, tol=1e-7)
        # early exit if time low
        if time.time() - t0 > DEADLINE - 2:
            break
    # final safety: ensure normalized, no nan
    best_arr = best_arr / np.linalg.norm(best_arr, axis=1, keepdims=True)
    # if we have time left, quick final polish of best with remaining time (max 3s)
    remain = DEADLINE - (time.time() - t0)
    if remain > 2.5:
        try:
            yp = _lp_polish(best_arr, min(3.0, remain - 1.0))
            vp = _max_cos(yp)
            if vp < best_val and np.all(np.isfinite(yp)):
                best_arr = yp
        except Exception:
            pass
    return best_arr

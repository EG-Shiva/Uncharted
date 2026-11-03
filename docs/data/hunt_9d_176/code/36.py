import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def cheap_gd(X, steps=300, alpha=80.0, lr0=0.01, lr1=0.002):
    X = X.copy()
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        Grad = E @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-30
        Gt = Gt / nr
        X = X - lr * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def lp_polish_deadline(x, deadline, radius_init=4e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def bfs_community(adj, seed, K, rng):
    visited = np.zeros(len(adj), dtype=bool)
    visited[seed] = True
    order = [seed]
    queue = [seed]
    qh = 0
    while len(order) < K and qh < len(queue):
        u = queue[qh]
        qh += 1
        nb = adj[u]
        if len(nb) == 0:
            continue
        perm = rng.permutation(len(nb))
        for k in perm:
            v = int(nb[k])
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                order.append(v)
                if len(order) >= K:
                    break
    return order

def reinsert_cluster(base, comm, rng, ntrials=20000):
    n, d = base.shape
    K = len(comm)
    comm = list(comm)
    frozen_mask = np.ones(n, dtype=bool)
    frozen_mask[np.array(comm)] = False
    frozen = base[frozen_mask].copy()
    Y = frozen.copy()
    new_vecs = []
    for k in range(K):
        trials = rng.standard_normal((ntrials, d))
        tn = np.linalg.norm(trials, axis=1, keepdims=True)
        tn[tn == 0] = 1.0
        trials /= tn
        dots_all = trials @ Y.T
        m_vals = dots_all.max(axis=1)
        # best 20 by max
        if ntrials >= 20:
            idx20 = np.argpartition(m_vals, 20)[:20]
        else:
            idx20 = np.arange(ntrials)
        V = trials[idx20].copy()  # (b,9)
        b = V.shape[0]
        # vectorized tangent refinement
        for _ in range(50):
            D = V @ Y.T  # (b,m)
            dm = D.max(axis=1, keepdims=True)
            W = np.exp(80.0 * (D - dm))
            s = W.sum(axis=1, keepdims=True) + 1e-30
            Gv = (W @ Y) / s  # (b,9)
            proj = np.sum(Gv * V, axis=1, keepdims=True)
            Gt = Gv - proj * V
            V = V - 0.05 * Gt
            vn = np.linalg.norm(V, axis=1, keepdims=True)
            vn[vn == 0] = 1.0
            V /= vn
        Df = V @ Y.T
        mf = Df.max(axis=1)
        # tie-break sum exp(80c) normalized
        tie = np.exp(80.0 * (Df - mf[:, None])).sum(axis=1)
        # lexicographic: primary mf, secondary tie
        order = np.lexsort((tie, mf))
        best_v = V[order[0]]
        new_vecs.append(best_v)
        Y = np.vstack([Y, best_v[None, :]])
    out = base.copy()
    for ci, v in zip(comm, new_vecs):
        out[ci] = v
    return out

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        try:
            xr = np.load("record.npy")
            xr = np.asarray(xr, dtype=float)
            if xr.shape == (176, 9):
                xb = xr
            else:
                xb = rng.standard_normal((176, 9))
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                best_val, best_x = vr, xr.copy()
    except Exception:
        pass

    search_deadline = t0 + 130.0
    hard_end = t0 + 172.0

    screened_best_x = best_x.copy()
    screened_best_v = best_val
    base_x = best_x.copy()
    base_v = best_val

    sizes = [12, 18, 24]
    it = 0
    # precompute nothing; loop
    while True:
        if time.time() >= search_deadline:
            break
        it += 1
        K = sizes[(it - 1) % len(sizes)]
        # contact graph at max-1e-5 on current base
        G = base_x @ base_x.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(np.max(G))
        thr = cur_max - 1e-5
        deg = np.sum(G > thr, axis=1)
        order_deg = np.argsort(-deg)
        # different seed each iteration: cycle through top-10
        seed_pos = (it - 1) % min(10, len(order_deg))
        seed = int(order_deg[seed_pos])
        adj = [np.where(G[i] > thr)[0] for i in range(176)]
        # ensure connectivity: if isolated, add nearest neighbors
        if len(adj[seed]) == 0:
            # take top correlated
            row = G[seed]
            adj[seed] = np.argsort(-row)[:8]
        comm = bfs_community(adj, seed, K, rng)
        if len(comm) < K:
            # fill with highest-contact nodes not in comm
            s = set(comm)
            for cand in order_deg:
                if int(cand) not in s:
                    comm.append(int(cand))
                    s.add(int(cand))
                    if len(comm) >= K:
                        break
        try:
            out = reinsert_cluster(base_x, comm, rng, ntrials=20000)
        except Exception:
            continue
        try:
            out = cheap_gd(out, steps=300, alpha=80.0, lr0=0.01, lr1=0.002)
        except Exception:
            continue
        try:
            v = max_cos(out)
        except Exception:
            continue
        if v < screened_best_v:
            screened_best_v = v
            screened_best_x = out.copy()
            # move base to new best for further search
            base_x = out.copy()
            base_v = v
        # if no improvement, keep base at global best (do not drift)
        # occasional restart base to screened best already done
        if time.time() >= search_deadline:
            break

    # LP polish only the single best screened candidate for remaining time
    deadline = min(hard_end, t0 + 177.0)
    # leave small buffer for return
    lp_deadline = deadline - 1.0
    if screened_best_v < best_val - 1e-12:
        # improved: polish it
        if time.time() < lp_deadline - 0.5:
            try:
                screened_best_x = lp_polish_deadline(screened_best_x, lp_deadline, radius_init=4e-3)
            except Exception:
                pass
        screened_best_x /= np.linalg.norm(screened_best_x, axis=1, keepdims=True)
        return screened_best_x
    else:
        # no improvement found: polish original best briefly to ensure at least record
        if time.time() < lp_deadline - 0.5:
            try:
                best_x = lp_polish_deadline(best_x, lp_deadline, radius_init=1e-3)
            except Exception:
                pass
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        # return whichever is better
        try:
            vb = max_cos(best_x)
        except Exception:
            return screened_best_x / np.linalg.norm(screened_best_x, axis=1, keepdims=True)
        if screened_best_v < vb:
            return screened_best_x / np.linalg.norm(screened_best_x, axis=1, keepdims=True)
        return best_x

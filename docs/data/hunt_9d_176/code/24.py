import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def repulsion_gd(x, steps=300, alpha=80.0, lr0=0.015, lr1=0.002):
    x = x.copy()
    n = x.shape[0]
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        Grad = E @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        lr = lr0 + (lr1 - lr0) * (t / max(steps - 1, 1))
        x = x - lr * (Gt / nr)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def refine_cands(V, cur, alpha=80.0, steps=40, lr=0.03):
    V = V.copy()
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-300)
    for s in range(steps):
        C = V @ cur.T
        cm = C.max(axis=1, keepdims=True)
        W = np.exp(alpha * (C - cm))
        sw = W.sum(axis=1, keepdims=True) + 1e-300
        Grad = (W @ cur) / sw
        dot = np.sum(Grad * V, axis=1, keepdims=True)
        Gt = Grad - dot * V
        lrt = lr * (1.0 - 0.5 * s / max(steps - 1, 1))
        V = V - lrt * Gt
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-300)
    return V

def lp_polish_deadline(x, deadline, radius_init=3e-3, cap=2500):
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

def bfs_community(G, counts, seed, size):
    n = G.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    order = [seed]
    queue = [seed]
    # neighbor lists sorted by cosine desc (precompute on fly)
    while len(order) < size and len(queue) > 0:
        u = queue.pop(0)
        row = G[u].copy()
        row[visited] = -2.0
        row[u] = -2.0
        # candidates with contact first, else any
        # sort descending
        idx = np.argsort(-row)
        for v in idx:
            if len(order) >= size:
                break
            if visited[v]:
                continue
            # only take contact neighbors first; if row[v] very low skip unless needed
            if row[v] < -1.5 and len(queue) == 0:
                # allow fallback fill later
                continue
            visited[v] = True
            order.append(int(v))
            queue.append(int(v))
        if len(queue) == 0 and len(order) < size:
            # disconnected: jump to highest-count unvisited to continue
            rem = np.where(~visited)[0]
            if len(rem) == 0:
                break
            nxt = rem[np.argmax(counts[rem])]
            visited[nxt] = True
            order.append(int(nxt))
            queue.append(int(nxt))
    # fallback fill if still short
    if len(order) < size:
        rem = np.where(~visited)[0]
        top = rem[np.argsort(-counts[rem])][:size - len(order)]
        for v in top:
            visited[v] = True
            order.append(int(v))
    return np.array(order[:size], dtype=int)

def solve():
    t0 = time.time()
    SEARCH_CAP = 130.0
    TOTAL = 172.0
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

    base_x = best_x.copy()
    base_val = best_val
    sizes = [12, 18, 24]
    it = 0
    # precompute contact info of base (refresh periodically if improved)
    G0 = base_x @ base_x.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(np.max(G0))
    thr0 = m0 - 1e-5
    adj_counts0 = np.sum(G0 > thr0, axis=1)
    top_order = np.argsort(-adj_counts0)

    while True:
        if time.time() - t0 > SEARCH_CAP:
            break
        it += 1
        if time.time() - t0 > SEARCH_CAP - 0.5:
            break
        # vary seed and size
        seed_rank = (it - 1) % 24
        if seed_rank < len(top_order):
            seed = int(top_order[seed_rank % len(top_order)])
        else:
            seed = int(rng.integers(0, 176))
        # also occasionally random seed to diversify
        if it % 7 == 0:
            seed = int(rng.integers(0, 176))
        S = sizes[(it - 1) % 3]

        # rebuild contact graph every 10 iters from current base
        if it % 10 == 1:
            G0 = base_x @ base_x.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(np.max(G0))
            thr0 = m0 - 1e-5
            adj_counts0 = np.sum(G0 > thr0, axis=1)
            top_order = np.argsort(-adj_counts0)
            # keep G full for BFS sorting (use true dots)
            Gfull = base_x @ base_x.T
        else:
            Gfull = base_x @ base_x.T

        comm = bfs_community(Gfull, adj_counts0, seed, S)
        mask = np.ones(176, dtype=bool)
        mask[comm] = False
        frozen = base_x[mask].copy()  # (176-S, 9)
        cur_list = [frozen]
        cur = frozen.copy()
        # sequential reinsertion
        for k in range(S):
            if time.time() - t0 > SEARCH_CAP - 0.5:
                break
            NT = 20000
            trials = rng.standard_normal((NT, 9))
            trials /= (np.linalg.norm(trials, axis=1, keepdims=True) + 1e-300)
            dots = trials @ cur.T
            cmax = dots.max(axis=1)
            # best 20 by cmax
            part = np.argpartition(cmax, 20)[:20]
            V0 = trials[part]
            # local tangent refinement
            Vr = refine_cands(V0, cur, alpha=80.0, steps=40, lr=0.03)
            C2 = Vr @ cur.T
            cmax2 = C2.max(axis=1)
            # tie-break sum exp(80c)
            # exp safe (c~0.5 -> exp40 ~ 2e17)
            s2 = np.exp(80.0 * (C2 - cmax2[:, None])).sum(axis=1)
            # lexicographic: primary cmax2, secondary s2 (log)
            # find best: minimal cmax2, break ties within 1e-12 by s2
            b = int(np.argmin(cmax2))
            bv = cmax2[b]
            tied = np.where(cmax2 <= bv + 1e-12)[0]
            if len(tied) > 1:
                b = int(tied[np.argmin(s2[tied])])
            best_v = Vr[b].copy()
            cur = np.vstack([cur, best_v[None, :]])
        if cur.shape[0] != 176:
            continue
        # 300 steps normalized exponential-repulsion GD
        cand = repulsion_gd(cur, steps=300, alpha=80.0, lr0=0.015, lr1=0.002)
        v = max_cos(cand)
        if v < base_val - 1e-12:
            base_val = v
            base_x = cand.copy()
            # refresh contact info immediately
            G0 = base_x @ base_x.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(np.max(G0))
            thr0 = m0 - 1e-5
            adj_counts0 = np.sum(G0 > thr0, axis=1)
            top_order = np.argsort(-adj_counts0)
            Gfull = base_x @ base_x.T
        if v < best_val - 1e-12:
            best_val = v
            best_x = cand.copy()

    # LP polish only the single best screened candidate for remaining time
    final = base_x if base_val <= best_val else best_x
    deadline = t0 + TOTAL
    # leave 1s margin
    if time.time() < deadline - 1.0:
        try:
            final = lp_polish_deadline(final, deadline - 0.5, radius_init=3e-3)
        except Exception:
            pass
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    return final

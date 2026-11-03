import time
import numpy as np
from collections import deque
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_deadline(x, deadline, r_init=0.004, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _repel(X, steps=300, beta=80.0, lr=0.015):
    n, d = X.shape
    for _ in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # exp weights, zero diag
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gn = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        X = X - (lr / gn) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def solve():
    t0 = time.time()
    SEARCH_END = t0 + 130.0
    HARD_END = t0 + 177.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    try:
        Xb = np.load("best.npy")
    except Exception:
        try:
            Xb = np.load("record.npy")
        except Exception:
            Xb = rng.standard_normal((n, d))
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (n, d) or not np.all(np.isfinite(Xb)):
        Xb = rng.standard_normal((n, d))
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best_X = Xb.copy()
    best_v = _max_cos(best_X)
    sizes = [12, 18, 24]
    it = 0
    # search loop: no LP inside
    while time.time() < SEARCH_END:
        it += 1
        if time.time() >= SEARCH_END:
            break
        k = sizes[(it - 1) % 3]
        # contact graph of best_X at max-1e-5
        G = best_X @ best_X.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        thresh = cur_max - 1e-5
        adj = G >= thresh
        deg = adj.sum(axis=1)
        order = np.argsort(-deg)
        # vary seed among top hubs for diversity
        seed_rank = (it - 1) % min(10, n)
        seed = int(order[seed_rank])
        # BFS community
        visited = set([seed])
        q = deque([seed])
        # precompute neighbor lists on demand
        while len(visited) < k and q:
            u = q.popleft()
            nbrs = np.where(adj[u])[0]
            if len(nbrs) > 1:
                nbrs = nbrs[rng.permutation(len(nbrs))]
            for v in nbrs:
                v = int(v)
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                    if len(visited) >= k:
                        break
            if len(visited) >= k:
                break
            if len(q) == 0 and len(visited) < k:
                # disconnected: add next highest-contact outside
                for cand in order:
                    cand = int(cand)
                    if cand not in visited:
                        visited.add(cand)
                        q.append(cand)
                        break
        del_idx = np.array(sorted(visited)[:k], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        frozen = best_X[keep_mask].copy()  # m x d
        cur = frozen.copy()
        failed = False
        for s in range(k):
            if time.time() >= SEARCH_END:
                failed = True
                break
            m = cur.shape[0]
            # 20000 random sphere candidates
            Cand = rng.standard_normal((20000, d))
            Cand /= np.linalg.norm(Cand, axis=1, keepdims=True)
            Dots = Cand @ cur.T  # 20000 x m
            maxc = Dots.max(axis=1)
            # cheap screening: top 200 by max
            part = np.argpartition(maxc, 200)[:200]
            Dp = Dots[part]
            # tie-break sum exp(80c)
            # exp may be large but fits in float64 (exp(80)~5e34)
            sumE = np.exp(80.0 * np.clip(Dp, -2.0, 1.0)).sum(axis=1)
            o2 = np.lexsort((sumE, maxc[part]))
            top20 = part[o2[:20]]
            B = Cand[top20].copy()  # 20 x d
            # local tangent refinement of best 20
            step = 0.05
            for _ in range(15):
                Db = B @ cur.T  # 20 x m
                rmax = Db.max(axis=1, keepdims=True)
                W = np.exp(80.0 * (Db - rmax))
                den = W.sum(axis=1, keepdims=True)
                den[den == 0] = 1.0
                Grad = (W @ cur) / den  # 20 x d
                dot = np.sum(Grad * B, axis=1, keepdims=True)
                Gt = Grad - dot * B
                B = B - step * Gt
                B /= np.linalg.norm(B, axis=1, keepdims=True)
                step *= 0.97
            # evaluate refined
            Dr = B @ cur.T
            maxr = Dr.max(axis=1)
            sumEr = np.exp(80.0 * np.clip(Dr, -2.0, 1.0)).sum(axis=1)
            pick = int(np.lexsort((sumEr, maxr))[0])
            chosen = B[pick]
            cur = np.vstack([cur, chosen[None, :]])
        if failed:
            break
        # cur is now n x d (m+k)
        Xt = cur.copy()
        # only 300 steps of normalised exponential-repulsion GD
        Xt = _repel(Xt, steps=300, beta=80.0, lr=0.015)
        v = _max_cos(Xt)
        if v < best_v and np.all(np.isfinite(Xt)):
            best_v = v
            best_X = Xt.copy()
        # continue loop; time checked at top
    # LP polish only the single best screened candidate for remaining time
    if time.time() < HARD_END:
        secs_left = HARD_END - time.time()
        # reserve 1s for return
        secs_left = max(0.5, secs_left - 0.5)
        # cap LP time to remaining but at least a few seconds if possible
        try:
            best_X = _lp_polish_deadline(best_X, time.time() + secs_left)
        except Exception:
            pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X

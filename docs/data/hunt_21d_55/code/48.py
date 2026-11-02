import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _refine_single(y0, Y, steps=300, a0=20.0, a1=500.0, lr0=0.03, lr1=0.003):
    # annealed LogSumExp Adam for one point y vs fixed set Y (m x d)
    y = y0.copy() / max(np.linalg.norm(y0), 1e-12)
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute ratio logs
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        s = Y @ y  # (m,)
        mx = np.max(s)
        # softmax weights
        e = np.exp(alpha * (s - mx))
        se = np.sum(e)
        if not np.isfinite(se) or se <= 0:
            continue
        w = e / se
        g = w @ Y  # (d,)
        # tangent projection
        gt = g - float(np.dot(g, y)) * y
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        y = y - lr * mhat / (np.sqrt(vhat) + eps)
        ny = np.linalg.norm(y)
        if not np.isfinite(ny) or ny < 1e-12:
            y = y0 / max(np.linalg.norm(y0), 1e-12)
            m[:] = 0; v[:] = 0
            continue
        y = y / ny
    return y

def _lp_polish_deadline(x, deadline, radius_init=1e-3):
    x = _norm_rows(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    T_HARD = t0 + 165.0
    T_SEARCH_END = t0 + 150.0  # reserve final 15s for polishing
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    x_best = best.copy()

    it = 0
    # cheap contact-cluster surgery loop
    while time.time() < T_SEARCH_END:
        it += 1
        if time.time() >= T_HARD - 16.0:
            break
        # contact graph with threshold best-1e-6
        thr = best_score - 1e-6
        G = x_best @ x_best.T
        # adjacency (exclude diag)
        adj_mask = G > thr
        np.fill_diagonal(adj_mask, False)
        counts = np.sum(adj_mask, axis=1)
        # highest-contact hub (randomise among top-5 for diversity)
        order = np.argsort(-counts)
        if rng.random() < 0.8:
            hub = int(order[0])
        else:
            k5 = min(5, n)
            hub = int(rng.choice(order[:k5]))
        # BFS cluster of 6-9 neighbours
        k = int(rng.integers(6, 10))  # 6..9
        visited = [hub]
        seen = np.zeros(n, dtype=bool)
        seen[hub] = True
        queue = [hub]
        qi = 0
        # neighbour order randomised by dot value to vary topology
        while len(visited) < k and qi < len(queue):
            u = queue[qi]; qi += 1
            nbrs = np.where(adj_mask[u])[0]
            if len(nbrs) > 1:
                # sort by dot descending then random tie perturb: shuffle then stable sort
                vals = G[u, nbrs]
                perm = np.argsort(-vals)
                nbrs = nbrs[perm]
                # random shuffle in small blocks to add diversity
                # (cheap: with prob 0.3 shuffle)
                if rng.random() < 0.3:
                    rng.shuffle(nbrs[:min(len(nbrs), 8)])
            for w in nbrs:
                if not seen[w]:
                    seen[w] = True
                    visited.append(int(w))
                    queue.append(int(w))
                    if len(visited) >= k:
                        break
        if len(visited) < k:
            # fill with random non-visited
            extra = rng.choice([i for i in range(n) if not seen[i]], size=k - len(visited), replace=False)
            for e in extra:
                visited.append(int(e))
        cluster = np.array(visited[:k], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[cluster] = False
        # retained points list (will grow as we reinsert)
        Y_list = [x_best[i] for i in range(n) if keep_mask[i]]
        Y_mat = np.stack(Y_list, axis=0) if len(Y_list) > 0 else np.zeros((0, d))
        # random reinsertion order
        rins_order = cluster.copy()
        rng.shuffle(rins_order)
        new_vecs = {}
        ok = True
        for idx_pos, _orig in enumerate(rins_order):
            # time check every inner loop
            if time.time() >= T_SEARCH_END:
                ok = False
                break
            Y_cur = np.stack(Y_list, axis=0)
            # sample 20000 random sphere points, keep minimiser of max dot
            NC = 20000
            cand = rng.standard_normal((NC, d))
            cand = _norm_rows(cand)
            # max dot against retained
            dots = cand @ Y_cur.T
            md = np.max(dots, axis=1)
            bi = int(np.argmin(md))
            y0 = cand[bi]
            # refine with 300 steps vectorised annealed LogSumExp Adam
            yref = _refine_single(y0, Y_cur, steps=300)
            Y_list.append(yref)
            new_vecs[_orig] = yref
        if not ok:
            break
        xcand = x_best.copy()
        for o, v in new_vecs.items():
            xcand[o] = v
        xcand = _norm_rows(xcand)
        raw = _max_cos(xcand)
        if raw < best_score:
            best_score = float(raw)
            x_best = xcand.copy()
            # if raw already beats, still try quick LP if time allows? continue to LP below
        # invoke trust-region LP only for candidates below 0.0675, at most 3s each
        if raw < 0.0675:
            remain_search = T_SEARCH_END - time.time()
            remain_hard = T_HARD - 16.0 - time.time()
            if remain_search <= 0.2 or remain_hard <= 0.2:
                break
            budget = min(3.0, remain_search, remain_hard)
            if budget < 0.3:
                continue
            deadline = time.time() + budget
            try:
                xpol = _lp_polish_deadline(xcand, deadline)
            except Exception:
                continue
            xpol = _norm_rows(xpol)
            s = _max_cos(xpol)
            if s < best_score:
                best_score = float(s)
                x_best = xpol.copy()
        # loop continues; check wall clock at top
    # final 15s polishing of best candidate
    try:
        if time.time() < T_HARD - 0.5:
            deadline = min(T_HARD - 0.3, time.time() + max(0.5, T_HARD - 0.3 - time.time()))
            # use remaining reserve (up to ~15s)
            x_best = _lp_polish_deadline(x_best, deadline)
            x_best = _norm_rows(x_best)
    except Exception:
        pass
    x_best = _norm_rows(np.asarray(x_best, dtype=np.float64))
    return x_best

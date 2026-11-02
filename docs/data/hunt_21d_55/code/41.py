import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _refine_point(v0, Y, steps=300, a0=40.0, a1=450.0, lr0=0.03, lr1=0.002):
    v = v0.copy()
    m = np.zeros_like(v)
    vv = np.zeros_like(v)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = t / steps
        alpha = a0 * (a1 / a0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        dots = Y @ v
        Mx = np.max(dots)
        E = np.exp(alpha * (dots - Mx))
        S = np.sum(E)
        if not np.finite(S) or S <= 0:
            continue
        w = E / S
        grad = w @ Y
        gt = grad - np.dot(grad, v) * v
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        v = v - lr * mhat / (np.sqrt(vhat) + eps)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = v0
        else:
            v = v / nv
    return v

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def solve():
    t0 = time.time()
    T165 = t0 + 165.0
    T_SEARCH_END = t0 + 150.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    x_best = best.copy()
    cur_best = float(best_score)

    NCAND = 20000
    trial = 0
    while time.time() < T_SEARCH_END:
        trial += 1
        if time.time() > T165 - 16:
            break
        # contact graph with threshold best-1e-6
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        thresh = cur_best - 1e-6
        mask = G > thresh
        deg = np.sum(mask, axis=1)
        hub = int(np.argmax(deg))
        # cluster size 6-9
        k = int(rng.integers(6, 10))
        # BFS from hub
        visited = set([hub])
        queue = [hub]
        # pre-sort neighbours by dot descending for determinism + randomness small
        while len(visited) < k and len(queue) > 0:
            u = queue.pop(0)
            nbrs = np.where(mask[u])[0]
            if len(nbrs) == 0:
                continue
            order = np.argsort(-G[u, nbrs])
            nbrs = nbrs[order]
            for w in nbrs:
                w = int(w)
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    if len(visited) >= k:
                        break
        # if disconnected and still small, fill with highest-contact nodes
        if len(visited) < k:
            order_all = np.argsort(-deg)
            for w in order_all:
                w = int(w)
                if w not in visited:
                    visited.add(w)
                    if len(visited) >= k:
                        break
        cluster = np.array(sorted(visited), dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[cluster] = False
        retained = x_best[keep_mask]
        # reinsert one at a time
        Y = retained.copy()
        new_pts = []
        # random order of insertion
        ins_order = rng.permutation(len(cluster))
        ok = True
        for _ii in ins_order:
            if time.time() > T_SEARCH_END:
                ok = False
                break
            cand = rng.standard_normal((NCAND, d))
            cand = _norm(cand)
            dots = cand @ Y.T
            md = np.max(dots, axis=1)
            bi = int(np.argmin(md))
            v0 = cand[bi]
            v = _refine_point(v0, Y, steps=300)
            # soft-minimum check: also evaluate raw max
            Y = np.vstack([Y, v[None, :]])
            new_pts.append(v)
        if not ok:
            break
        # assemble candidate in original ordering: retained + new in cluster positions
        # map: we inserted in ins_order; need to place back; simply fill cluster slots in insertion order
        x_cand = np.empty_like(x_best)
        x_cand[keep_mask] = retained
        # new_pts corresponds to ins_order sequence
        for pos, v in zip(ins_order, new_pts):
            # pos is index into cluster array
            orig_idx = cluster[pos]
            x_cand[orig_idx] = v
        x_cand = _norm(x_cand)
        raw = _max_cos(x_cand)
        # screen by raw max cosine
        if raw < cur_best:
            # immediate improvement without LP
            x_best = x_cand.copy()
            cur_best = raw
            continue
        if raw < 0.0675:
            # gate LP: at most 3 seconds each
            remain_search = T_SEARCH_END - time.time()
            if remain_search < 1.0:
                continue
            budget = min(3.0, remain_search, T165 - time.time() - 1.0)
            if budget <= 0.2:
                continue
            x_pol = _lp_polish(x_cand, budget)
            s_pol = _max_cos(x_pol)
            if s_pol < cur_best:
                x_best = x_pol.copy()
                cur_best = s_pol
        # else discard
    # reserve final 15 seconds for polishing best candidate
    remain = T165 - time.time()
    if remain > 0.5:
        budget = min(remain, 15.0)
        # only polish if worthwhile; always polish briefly
        try:
            x_best = _lp_polish(x_best, budget)
            cur_best = _max_cos(x_best)
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best

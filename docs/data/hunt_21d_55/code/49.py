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

def _refine_single(v0, Y, steps=300, a0=30.0, a1=400.0, lr0=0.03, lr1=0.003):
    v = v0.copy()
    v = v / max(np.linalg.norm(v), 1e-12)
    m = np.zeros_like(v)
    vv = np.zeros_like(v)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        dots = Y @ v
        M = np.max(dots)
        E = np.exp(alpha * (dots - M))
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            break
        w = E / S
        grad = w @ Y
        gt = grad - np.dot(grad, v) * v
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * vv + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        v = v - lr * mhat / (np.sqrt(vhat) + eps)
        nrm = np.linalg.norm(v)
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        v /= nrm
    return v

def _lp_polish_budget(x, deadline):
    x = _norm_rows(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1100]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I = iu[0][sel]
        J = iu[1][sel]
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
            break
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def solve():
    t0 = time.time()
    T_HARD = t0 + 165.0
    T_SEARCH_END = t0 + 150.0
    rng = np.random.default_rng(20261101)
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

    NCAND = 20000
    it = 0
    while time.time() < T_SEARCH_END:
        it += 1
        if time.time() >= T_HARD - 16.0:
            break
        # build contact graph with threshold best-1e-6
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            thr = best_score - 1e-6
            iu_all = np.triu_indices(n, 1)
            mask = G[iu_all] > thr
            deg = np.zeros(n, dtype=int)
            adj = [[] for _ in range(n)]
            ii = iu_all[0][mask]
            jj = iu_all[1][mask]
            for a_, b_ in zip(ii, jj):
                adj[int(a_)].append(int(b_))
                adj[int(b_)].append(int(a_))
                deg[int(a_)] += 1
                deg[int(b_)] += 1
            hub = int(np.argmax(deg))
            # BFS cluster 6-9
            k = int(rng.integers(6, 10))
            visited = set([hub])
            queue = [hub]
            qi = 0
            # randomize neighbor order for diversity
            while len(visited) < k and qi < len(queue):
                u = queue[qi]; qi += 1
                nbrs = adj[u]
                if len(nbrs) > 1:
                    order = rng.permutation(len(nbrs))
                    nbrs = [nbrs[o] for o in order]
                for w in nbrs:
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
                        if len(visited) >= k:
                            break
            # if graph too sparse, fill with high-contact nodes
            if len(visited) < k:
                extra = np.argsort(-deg)
                for e in extra:
                    e = int(e)
                    if e not in visited:
                        visited.add(e)
                    if len(visited) >= k:
                        break
            del_idx = sorted(list(visited))
        except Exception:
            # fallback random cluster
            k = int(rng.integers(6, 10))
            del_idx = sorted(list(rng.choice(n, size=k, replace=False)))
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        X_keep = x_best[keep_mask].copy()
        # reinsert one at a time
        Y = X_keep.copy()
        order_del = rng.permutation(len(del_idx))
        success = True
        for _di in order_del:
            if time.time() >= T_SEARCH_END:
                success = False
                break
            # sample 20000 random sphere points
            try:
                cand = rng.standard_normal((NCAND, d))
                cand = _norm_rows(cand)
                dots = cand @ Y.T
                md = np.max(dots, axis=1)
                bi = int(np.argmin(md))
                v0 = cand[bi].copy()
            except Exception:
                v0 = _norm_rows(rng.standard_normal((1, d)))[0]
            # refine with 300 steps annealed LogSumExp Adam
            try:
                vref = _refine_single(v0, Y, steps=300, a0=30.0, a1=400.0, lr0=0.03, lr1=0.003)
            except Exception:
                vref = v0
            Y = np.vstack([Y, vref[None, :]])
        if not success:
            break
        # Y is full 55 (order scrambled, fine)
        try:
            raw = _max_cos(Y)
        except Exception:
            continue
        # screen by raw max cosine
        if raw < best_score:
            # improvement without LP
            x_best = _norm_rows(Y)
            best_score = raw
            # still try LP to squeeze further if time allows
            if raw < 0.0675 and time.time() < T_SEARCH_END - 3.5:
                dl = min(time.time() + 3.0, T_SEARCH_END)
                try:
                    Yp, sp = _lp_polish_budget(x_best, dl)
                    if sp < best_score:
                        x_best = Yp
                        best_score = sp
                except Exception:
                    pass
        elif raw < 0.0675:
            # gated LP only for candidates below 0.0675, at most 3s each
            if time.time() < T_SEARCH_END - 3.5:
                dl = min(time.time() + 3.0, T_SEARCH_END)
                try:
                    Yp, sp = _lp_polish_budget(_norm_rows(Y), dl)
                    if sp < best_score:
                        x_best = Yp
                        best_score = sp
                except Exception:
                    pass
        # else discard
    # reserve final 15 seconds for polishing best candidate
    try:
        remain = T_HARD - time.time()
        if remain > 0.5:
            polish_budget = min(remain, 15.0)
            dl = time.time() + polish_budget
            Yp, sp = _lp_polish_budget(x_best, dl)
            if sp < best_score:
                x_best = Yp
                best_score = sp
    except Exception:
        pass
    x_best = _norm_rows(x_best)
    return x_best

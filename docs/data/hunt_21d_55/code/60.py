import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_deadline(x, deadline, radius_init=8e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
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
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _refine_one(y0, others, steps=300, tau=0.001, lr0=0.015, lr1=0.003):
    y = y0.copy()
    d = y.shape[0]
    m1 = np.zeros(d)
    m2 = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        dots = others @ y
        mx = dots.max()
        e = np.clip((dots - mx) / tau, -60.0, 0.0)
        w = np.exp(e)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            break
        g = (w @ others) / s
        gt = g - np.dot(g, y) * y
        lr = lr0 + (lr1 - lr0) * (t - 1) / max(steps - 1, 1)
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        mh = m1 / (1.0 - b1 ** t)
        vh = m2 / (1.0 - b2 ** t)
        y = y - lr * mh / (np.sqrt(vh) + eps)
        n = np.linalg.norm(y)
        if n < 1e-12:
            break
        y /= n
    return y

def _joint_quench(Y0, S, steps=600, tau0=0.004, tau1=0.0008, lr0=0.02, lr1=0.003):
    Y = Y0.copy()
    k, d = Y.shape
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = (t - 1) / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        Dys = Y @ S.T
        Dyy = Y @ Y.T
        np.fill_diagonal(Dyy, -10.0)
        max_ys = Dys.max(axis=1)
        max_yy = Dyy.max(axis=1)
        Mx = np.maximum(max_ys, max_yy)[:, None]
        Wys = np.exp(np.clip((Dys - Mx) / tau, -60.0, 0.0))
        Wyy = np.exp(np.clip((Dyy - Mx) / tau, -60.0, 0.0))
        Ssum = Wys.sum(axis=1) + Wyy.sum(axis=1) + 1e-30
        G = (Wys @ S + Wyy @ Y) / Ssum[:, None]
        proj = np.sum(G * Y, axis=1, keepdims=True)
        Gt = G - proj * Y
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1 ** t)
        vh = m2 / (1.0 - b2 ** t)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    T_SEARCH_END = t0 + 145.0
    T_HARD_END = t0 + 172.0
    rng = np.random.default_rng(20261101)
    n, d = 55, 21
    xb = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (55, 21):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    xb = a.copy()
        except Exception:
            continue
    if xb is None:
        xb = rng.standard_normal((n, d))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        best_score = _max_cos(xb)
    x_best = xb.copy()

    N_RAND = 200000
    trial = 0
    while time.time() < T_SEARCH_END:
        trial += 1
        if time.time() > T_HARD_END - 25.0:
            break
        base = x_best
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        thresh = cur_max - 1e-4
        adj = G > thresh
        deg = adj.sum(axis=1)
        hub_order = np.argsort(-deg)
        top = min(10, n)
        hub = int(hub_order[(trial - 1) % top])
        k = 6 + ((trial - 1) % 5)
        # BFS neighbourhood of hub
        visited = np.zeros(n, dtype=bool)
        visited[hub] = True
        queue = [hub]
        order = [hub]
        # pre-sort neighbours by cosine for each node on demand
        while queue and len(order) < k:
            u = queue.pop(0)
            nbrs = np.where(adj[u])[0]
            if len(nbrs) > 1:
                nbrs = nbrs[np.argsort(-G[u, nbrs])]
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    queue.append(int(v))
                    order.append(int(v))
                    if len(order) >= k:
                        break
        if len(order) < k:
            # fill with closest to hub
            extra = np.argsort(-G[hub])
            for v in extra:
                if not visited[v]:
                    visited[v] = True
                    order.append(int(v))
                    if len(order) >= k:
                        break
        del_idx = np.array(order[:k], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        S = base[keep_mask]
        m = S.shape[0]
        # random candidates
        try:
            rand_c = rng.standard_normal((N_RAND, d))
        except Exception:
            break
        rand_c /= np.linalg.norm(rand_c, axis=1, keepdims=True)
        # pairwise averages of survivors
        iu2 = np.triu_indices(m, 1)
        sums = S[iu2[0]] + S[iu2[1]]
        sn = np.linalg.norm(sums, axis=1, keepdims=True)
        sn = np.maximum(sn, 1e-12)
        avg_c = sums / sn
        pool = np.vstack([rand_c, avg_c])
        del rand_c, sums
        P = pool.shape[0]
        # max cosine to survivors, chunked
        md = np.empty(P, dtype=np.float64)
        CH = 20000
        for s0 in range(0, P, CH):
            s1 = min(P, s0 + CH)
            md[s0:s1] = (pool[s0:s1] @ S.T).max(axis=1)
            if time.time() > T_SEARCH_END:
                break
        # greedy insert with per-point Adam (200-500 steps)
        Ylist = []
        Ystack = None
        for ti in range(k):
            bi = int(np.argmin(md))
            y0 = pool[bi].copy()
            if Ystack is None:
                others = S
            else:
                others = np.vstack([S, Ystack])
            yopt = _refine_one(y0, others, steps=300, tau=0.001)
            Ylist.append(yopt)
            Ystack = np.stack(Ylist)
            # update md: max with new point
            dots_pool = pool @ yopt
            np.maximum(md, dots_pool, out=md)
        Y0 = np.stack(Ylist)
        # joint annealed soft-max quench (survivors frozen)
        Yq = _joint_quench(Y0, S, steps=600)
        # light joint relax of everything (short)
        x_try = base.copy()
        x_try[del_idx] = Yq
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        v = _max_cos(x_try)
        if v < 0.0675:
            # gated LP, very short
            remain = T_SEARCH_END - time.time()
            if remain > 1.0:
                budget_end = time.time() + min(1.5, remain - 0.2)
                x_try, v = _lp_polish_deadline(x_try, budget_end, radius_init=8e-4)
        if v < best_score - 1e-12:
            x_best = x_try.copy()
            best_score = v
        # occasional second seed variant: also try flipped sign of deepest hole? skip to save time
        if trial > 250:
            break
    # final LP polish at most 20s
    remain = T_HARD_END - time.time()
    if remain > 2.0:
        fin = min(15.0, remain - 1.0)
        x_best, best_score = _lp_polish_deadline(x_best, time.time() + fin, radius_init=1e-3)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

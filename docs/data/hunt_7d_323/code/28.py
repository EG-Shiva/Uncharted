import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_counts(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _lp_polish(x, seconds, init_radius=8e-3, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t_deadline is not None and now >= t_deadline:
            break
        if radius < 1e-14:
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
            cidx = c[idx]
            part = np.argpartition(cidx, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cols.reshape(p, 2 * d + 1)[:, :d] = I[:, None] * d + np.arange(d)
        cols.reshape(p, 2 * d + 1)[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols.reshape(p, 2 * d + 1)[:, 2 * d] = n * d
        vals.reshape(p, 2 * d + 1)[:, :d] = x[J]
        vals.reshape(p, 2 * d + 1)[:, d:2 * d] = x[I]
        vals.reshape(p, 2 * d + 1)[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        remain = seconds - (time.time() - start)
        if t_deadline is not None:
            remain = min(remain, t_deadline - time.time())
        if remain < 0.2:
            break
        tl = float(min(2.0, max(0.3, remain)))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
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
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if t_deadline is not None and time.time() >= t_deadline:
            break
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    SA_END = t0 + 132.0
    rng = np.random.default_rng(12345)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v
                        x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    n, d = 323, 7
    global_best = x0.copy()
    global_val = float(b0)
    cur_x = x0.copy()
    cur_val = float(b0)

    T0, T1 = 1.5e-4, 2.0e-6
    it = 0
    # preallocate candidate buffer reused
    while time.time() < SA_END and time.time() < DEADLINE - 8:
        it += 1
        elapsed = time.time() - t0
        # temperature decaying over ~120s
        frac = min(max(elapsed / 120.0, 0.0), 1.0)
        # exponential decay T0 -> T1
        Temp = T0 * (T1 / T0) ** frac
        if Temp < 1e-12:
            Temp = 1e-12
        # --- pick hub with >=18 contacts on cur_x ---
        try:
            m_cur = _max_cos(cur_x)
            cc = _contact_counts(cur_x, m_cur, tol=1e-9)
        except Exception:
            cc = np.ones(n, dtype=int) * 10
            m_cur = cur_val
        hubs = np.where(cc >= 18)[0]
        if len(hubs) == 0:
            hub = int(np.argmax(cc))
        else:
            hub = int(hubs[rng.integers(0, len(hubs))])
        # contact graph adjacency for BFS
        try:
            Gc = cur_x @ cur_x.T
            np.fill_diagonal(Gc, -2.0)
            adj = [np.where(Gc[i] > m_cur - 1e-9)[0] for i in range(n)]
        except Exception:
            adj = [np.array([], dtype=int) for _ in range(n)]
        # BFS neighbourhood
        visited = set([hub])
        queue = [hub]
        qi = 0
        # random BFS size target 22-38
        bfs_target = int(rng.integers(22, 39))
        # shuffle adjacency order via rng for diversity
        while qi < len(queue) and len(visited) < bfs_target:
            u = queue[qi]; qi += 1
            nbrs = adj[u]
            if len(nbrs) > 1:
                perm = rng.permutation(len(nbrs))
                nbrs = nbrs[perm]
            for w in nbrs:
                w = int(w)
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    if len(visited) >= bfs_target:
                        break
        K = int(rng.integers(30, 61))  # total delete 30-60
        to_del = set(visited)
        # add both rattlers
        ratt = np.where(cc == 0)[0].tolist()
        for r in ratt:
            to_del.add(int(r))
        # add lowest-contact points until K
        if len(to_del) < K:
            order = np.argsort(cc, kind="stable")
            for o in order:
                if len(to_del) >= K:
                    break
                to_del.add(int(o))
        # if too many, trim keeping hub+rattlers, drop highest-contact extras
        if len(to_del) > K:
            lst = list(to_del)
            # keep hub and rattlers forced
            forced = set([hub] + [int(r) for r in ratt])
            rest = [v for v in lst if v not in forced]
            # sort rest by contacts descending -> drop first
            rest_sorted = sorted(rest, key=lambda v: cc[v], reverse=True)
            ndrop = len(to_del) - K
            for v in rest_sorted[:ndrop]:
                to_del.remove(v)
        to_replace = np.array(sorted(list(to_del)), dtype=int)
        K = len(to_replace)
        fixed_mask = np.ones(n, dtype=bool)
        fixed_mask[to_replace] = False
        X_fixed = cur_x[fixed_mask]

        y = cur_x.copy()
        # --- farthest-point sampling from 50000 candidates ---
        try:
            NC = 50000
            cand = rng.standard_normal(size=(NC, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # initial worst vs fixed, chunked
            worst = np.empty(NC, dtype=np.float64)
            worst.fill(-2.0)
            CH = 10000
            for s in range(0, NC, CH):
                e = min(s + CH, NC)
                D = cand[s:e] @ X_fixed.T
                worst[s:e] = D.max(axis=1)
                if time.time() >= DEADLINE - 5:
                    break
            # sequential greedy
            for k in to_replace:
                if time.time() >= DEADLINE - 5:
                    break
                bi = int(np.argmin(worst))
                bp = cand[bi].copy()
                y[int(k)] = bp
                # update worst and forbid reuse
                dots_new = cand @ bp
                # worst = max(worst, dots_new)
                np.maximum(worst, dots_new, out=worst)
                worst[bi] = np.inf
        except Exception:
            # fallback: random reinsert
            for k in to_replace:
                v0 = rng.standard_normal(size=(d,))
                v0 /= np.linalg.norm(v0)
                y[int(k)] = v0

        # --- 300 steps softmax repulsion, only freed move ---
        try:
            free_idx = to_replace
            Kf = len(free_idx)
            # map global col index for masking self
            # build position map: global -> row in free (or -1)
            g2r = np.full(n, -1, dtype=int)
            g2r[free_idx] = np.arange(Kf)
            X = y.copy()
            XF = X[free_idx].copy()
            vel = np.zeros_like(XF)
            mom = 0.85
            lr0 = 0.02
            for step in range(300):
                if step % 25 == 0 and time.time() >= DEADLINE - 5:
                    break
                tau = 1e-2 * (0.1 ** (step / 299.0))  # 1e-2 -> 1e-3
                # G: Kf x n
                G = XF @ X.T
                # mask self dots: set to -inf via exp=0 later
                # compute rowwise max excluding self for stability
                # temporarily set self to -2
                for ri, g in enumerate(free_idx):
                    G[ri, int(g)] = -2.0
                mrow = G.max(axis=1, keepdims=True)
                E = np.exp((G - mrow) / tau)
                # self already -2 -> exp tiny but not zero; zero it
                for ri, g in enumerate(free_idx):
                    # col g weight zero
                    # need weight of fixed+other freed only; self excluded
                    pass
                # compute sums
                ssum = E.sum(axis=1, keepdims=True) + 1e-30
                W = E / ssum  # Kf x n
                Grad = W @ X  # Kf x d
                # tangential
                dotg = np.sum(Grad * XF, axis=1, keepdims=True)
                Gt = Grad - dotg * XF
                # clip
                gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
                # normalized-ish step with lr
                lr = lr0 * (0.5 + 0.5 * (1.0 - step / 300.0))  # 0.02 -> 0.01
                vel = mom * vel - lr * (Gt / np.maximum(gn, 1.0))
                XF = XF + vel
                XF /= np.linalg.norm(XF, axis=1, keepdims=True)
                # refresh X freed rows for next G (fixed stay)
                X[free_idx] = XF
                # early time exit
            y[free_idx] = XF
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        except Exception:
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # --- capped LP polish ---
        remain_sa = SA_END - time.time()
        remain_dl = DEADLINE - time.time()
        if remain_dl < 3:
            break
        budget = min(3.5, remain_sa - 0.2 if remain_sa > 0.5 else 2.0, remain_dl - 2.0)
        if budget < 0.6:
            budget = min(1.5, remain_dl - 1.0)
        if budget < 0.4:
            # no time for polish, evaluate raw
            v = _max_cos(y)
            ypol = y
        else:
            try:
                ypol = _lp_polish(y, seconds=float(budget), init_radius=8e-3, t_deadline=DEADLINE - 1.0)
                v = _max_cos(ypol)
            except Exception:
                v = _max_cos(y)
                ypol = y
        # --- Metropolis acceptance, allow up to 2e-4 worse ---
        delta = v - cur_val
        accept = False
        if delta < -1e-12:
            accept = True
        elif delta <= 2e-4:
            prob = float(np.exp(-delta / Temp)) if Temp > 0 else 0.0
            if rng.random() < prob:
                accept = True
        if accept:
            cur_x = ypol
            cur_val = float(v)
        if v < global_val - 1e-12:
            global_val = float(v)
            global_best = ypol.copy()
        # loop

    # final capped LP polish on global best
    try:
        remain = DEADLINE - time.time()
        if remain > 2.5:
            global_best = _lp_polish(global_best, seconds=min(remain - 0.5, 22.0), init_radius=2e-3, t_deadline=DEADLINE - 0.5)
            global_val = _max_cos(global_best)
    except Exception:
        pass
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    # ensure finite
    global_best = np.where(np.isfinite(global_best), global_best, 0.0)
    nn = np.linalg.norm(global_best, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    global_best = global_best / nn
    return global_best

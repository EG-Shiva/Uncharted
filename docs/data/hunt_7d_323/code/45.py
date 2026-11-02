import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, init_radius=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3500)[-3500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3500
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
    return x


def _refine_subset(y, subset, rng, sweeps, step0):
    n, d = y.shape
    sub = list(subset)
    if len(sub) == 0:
        return y
    for s in range(sweeps):
        step = step0 * (0.97 ** s)
        if step < 1e-6:
            break
        order = np.array(sub)
        # deterministic shuffle via rng
        perm = rng.permutation(len(order))
        order = order[perm]
        for k in order:
            k = int(k)
            cur = y[k].copy()
            dots = y @ cur
            dots[k] = -2.0
            j = int(np.argmax(dots))
            cur_w = float(dots[j])
            xj = y[j]
            dv = xj - float(xj @ cur) * cur
            nd = float(np.linalg.norm(dv))
            if nd < 1e-12:
                continue
            dv = dv / nd
            # gradient proposal
            prop = cur - step * dv
            pn = np.linalg.norm(prop)
            if pn == 0:
                continue
            prop = prop / pn
            d2 = y @ prop
            d2[k] = -2.0
            w2 = float(d2.max())
            if w2 < cur_w:
                y[k] = prop
                continue
            # second try: gradient + random tangent
            rnd = rng.standard_normal(size=d)
            rnd = rnd - float(rnd @ cur) * cur
            rn = float(np.linalg.norm(rnd))
            if rn < 1e-12:
                continue
            rnd = rnd / rn
            prop2 = cur - step * dv * 0.7 + step * 0.7 * rnd
            # project: actually combine then normalize
            pn2 = np.linalg.norm(prop2)
            if pn2 == 0:
                continue
            prop2 = prop2 / pn2
            d3 = y @ prop2
            d3[k] = -2.0
            w3 = float(d3.max())
            if w3 < cur_w:
                y[k] = prop2
    return y


def _bfs_cluster(adj, start, target, rng):
    n = len(adj)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    q = [int(start)]
    order = [int(start)]
    head = 0
    while len(order) < target and head < len(q):
        u = q[head]
        head += 1
        nbrs = adj[u]
        if len(nbrs) > 1:
            # randomize neighbor order for diversity
            perm = rng.permutation(len(nbrs))
            nbrs = nbrs[perm]
        for v in nbrs:
            v = int(v)
            if not visited[v]:
                visited[v] = True
                q.append(v)
                order.append(v)
                if len(order) >= target:
                    break
    return order


def solve():
    t0 = time.time()
    HARD = 165.0
    rng = np.random.default_rng(0)
    # load best start
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best = a
        best_val = _max_cos(a)
    n, d = best.shape
    all_idx = np.arange(n)
    it = 0
    # main cycle
    while time.time() - t0 < HARD:
        elapsed = time.time() - t0
        remain = HARD - elapsed
        if remain < 12.0:
            break
        # contact graph from current best with tol 1e-7
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        tol = 1e-7
        mask = G > (m - tol)
        counts = np.sum(mask, axis=1)
        # adjacency list
        adj = [np.where(mask[i])[0] for i in range(n)]
        hub_order = np.argsort(-counts)
        top_hubs = hub_order[:12]
        if it < len(top_hubs):
            start = int(top_hubs[it])
        else:
            # random connected cluster start
            start = int(rng.integers(0, n))
        target = int(rng.integers(12, 19))  # 12..18
        cluster = _bfs_cluster(adj, start, target, rng)
        if len(cluster) < 8:
            # fallback: add random nodes
            extra = rng.choice(n, size=(target - len(cluster),), replace=False)
            s = set(cluster)
            for e in extra:
                if int(e) not in s:
                    cluster.append(int(e))
                    s.add(int(e))
        cluster = [int(c) for c in cluster]
        cset = set(cluster)
        fixed_idx = [i for i in range(n) if i not in cset]
        X_fixed = best[fixed_idx].copy()
        y = best.copy()
        # sequential max-min reinsertion from 50000 candidates
        Xa = X_fixed.copy()
        # random insertion order for diversity
        ins_order = np.array(cluster)
        perm = rng.permutation(len(ins_order))
        ins_order = ins_order[perm]
        # map: we fill y positions in ins_order sequence
        cur_Xa = Xa
        new_list = []
        for k in ins_order:
            k = int(k)
            cand = rng.standard_normal(size=(50000, d))
            cn = np.linalg.norm(cand, axis=1, keepdims=True)
            cn[cn == 0] = 1.0
            cand = cand / cn
            dots = cand @ cur_Xa.T
            worst = dots.max(axis=1)
            bi = int(np.argmin(worst))
            bp = cand[bi]
            y[k] = bp
            new_list.append(bp[None, :])
            # grow cur_Xa
            cur_Xa = np.vstack([cur_Xa, bp[None, :]])
        # frozen-core optimisation of inserted points only
        y = _refine_subset(y, [int(c) for c in ins_order], rng, sweeps=60, step0=0.02)
        # short all-point smooth relaxation
        y = _refine_subset(y, all_idx, rng, sweeps=10, step0=0.008)
        # renormalize
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        pre = _max_cos(y)
        it += 1
        if pre > 0.670:
            continue
        # LP-polish only promising
        elapsed2 = time.time() - t0
        remain2 = HARD - elapsed2
        if remain2 < 10.0:
            break
        lp_sec = min(7.0, remain2 - 4.0)
        if lp_sec < 2.0:
            break
        try:
            y2 = _lp_polish(y, lp_sec)
        except Exception:
            continue
        y2 = y2 / np.linalg.norm(y2, axis=1, keepdims=True)
        post = _max_cos(y2)
        if post < best_val:
            best = y2.copy()
            best_val = post
            try:
                np.save("best.npy", best)
            except Exception:
                pass
        # time check
        if time.time() - t0 >= HARD:
            break
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

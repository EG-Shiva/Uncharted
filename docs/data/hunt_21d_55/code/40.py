import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_squeeze(x, end_time, r0=0.003):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = float(r0)
    iu = np.triu_indices(n, 1)
    while time.time() < end_time and radius > 1e-13:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1200:
            # keep tightest
            cs = c[sel]
            o = np.argsort(-cs)[:1200]
            I = I[o]; J = J[o]
            p = 1200
        cb = (x[I] * x[J]).sum(axis=1)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cb, A_eq=A_eq,
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            if v < m + 1e-12:
                x = y
            radius *= 0.5
        if time.time() >= end_time:
            break
    return xb, best

def _deep_hole(Y, rng, n_cand=200000):
    d = Y.shape[1]
    best_c = None
    best_m = 1e9
    # batched to bound memory, still >=200k total
    batch = 50000
    done = 0
    while done < n_cand:
        nb = min(batch, n_cand - done)
        done += nb
        cand = rng.randn(nb, d)
        cand = _norm_rows(cand)
        dots = cand @ Y.T
        md = dots.max(axis=1)
        i = int(np.argmin(md))
        if float(md[i]) < best_m:
            best_m = float(md[i])
            best_c = cand[i].copy()
    return best_c

def _refine_point(x0, Y, tau=0.008, steps=120, lr=0.15):
    x = x0.copy()
    for t in range(steps):
        c = Y @ x
        m = float(c.max())
        w = np.exp((c - m) / tau)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            break
        g = (w[:, None] * Y).sum(axis=0) / s
        gt = g - float(g @ x) * x
        # decay lr slightly
        l = lr * (1.0 - 0.5 * t / steps)
        x = x - l * gt
        n = float(np.linalg.norm(x))
        if n == 0 or not np.isfinite(n):
            x = x0.copy(); break
        x = x / n
    return x

def _connected_cluster(Gmax_mat, k, rng):
    n = Gmax_mat.shape[0]
    # Gmax_mat is Gram with diag -2
    m = float(Gmax_mat.max())
    adj_thr = m - 1e-3
    # adjacency list
    neigh = [np.where(Gmax_mat[i] > adj_thr)[0].tolist() for i in range(n)]
    deg = np.array([len(v) for v in neigh])
    # seed among top-10 degree with randomness
    order = np.argsort(-deg)
    top = order[:10]
    seed = int(top[rng.randint(len(top))])
    cluster = [seed]
    in_cl = np.zeros(n, dtype=bool); in_cl[seed] = True
    # BFS frontier
    frontier = neigh[seed][:]
    rng.shuffle(frontier)
    fi = 0
    # expand until k
    while len(cluster) < k:
        # try frontier
        found = None
        while fi < len(frontier):
            c = frontier[fi]; fi += 1
            if not in_cl[c]:
                found = c; break
        if found is None:
            # add more neighbours of cluster
            cand = []
            for u in cluster:
                for v in neigh[u]:
                    if not in_cl[v] and v not in cand:
                        cand.append(v)
            if len(cand) == 0:
                # disconnected: pick high-degree outside
                for o in order:
                    if not in_cl[int(o)]:
                        found = int(o); break
                if found is None:
                    break
            else:
                rng.shuffle(cand)
                frontier.extend(cand)
                continue
        in_cl[found] = True
        cluster.append(int(found))
        # add its neighbours to frontier
        nn = neigh[found][:]
        rng.shuffle(nn)
        frontier.extend(nn)
        if len(cluster) > n:
            break
    return np.array(cluster[:k], dtype=int)

def solve():
    t0 = time.time()
    DEAD = t0 + 150.0
    SEARCH_END = t0 + 137.0
    rng = np.random.RandomState(12345)
    n, d = 55, 21
    cur = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape != (55, 21):
                continue
            a = _norm_rows(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v; cur = a.copy()
        except Exception:
            continue
    if cur is None:
        cur = _norm_rows(rng.randn(n, d))
        best_val = _max_cos(cur)
    x_best = cur.copy()
    cur_val = best_val
    # quick tiny squeeze of incumbent if time allows
    if time.time() < DEAD - 20:
        xb, vb = _lp_squeeze(cur, min(time.time() + 1.0, SEARCH_END), r0=0.002)
        if vb < best_val:
            best_val = vb; x_best = xb.copy()
        cur = xb.copy(); cur_val = vb

    it = 0
    T0 = 0.002
    while time.time() < SEARCH_END:
        it += 1
        if time.time() >= SEARCH_END - 0.8:
            break
        # temperature annealing
        prog = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
        Temp = T0 * (1.0 - 0.9 * prog) + 1e-5
        # choose cluster size 4-8
        k = int(4 + rng.randint(5))  # 4..8
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        try:
            cluster = _connected_cluster(G, k, rng)
        except Exception:
            cluster = rng.choice(n, size=k, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[cluster] = False
        Yfrozen = cur[keep_mask]  # (n-k, d)
        # random order of reinsertion
        order = cluster.copy()
        rng.shuffle(order)
        Ycur = Yfrozen.copy()
        new_pts = {}
        # jointly reinsert: deep hole + soft-min refine, cycling
        # first pass
        feasible = True
        for pos in order:
            if time.time() >= SEARCH_END - 0.8:
                feasible = False; break
            h = _deep_hole(Ycur, rng, n_cand=200000)
            # local soft-min optimisation vs current others
            h = _refine_point(h, Ycur, tau=0.008, steps=120, lr=0.15)
            new_pts[int(pos)] = h.copy()
            Ycur = np.vstack([Ycur, h[None, :]])
        if not feasible:
            break
        # cycle several times through cluster to jointly optimise
        for cyc in range(3):
            if time.time() >= SEARCH_END - 0.8:
                break
            for pos in order:
                if time.time() >= SEARCH_END - 0.8:
                    break
                # others = all except pos
                others = []
                # frozen
                others.append(Yfrozen)
                for q, pt in new_pts.items():
                    if int(q) != int(pos):
                        others.append(pt[None, :])
                Yoth = np.vstack(others)
                x0 = new_pts[int(pos)]
                # small random kick before refine on later cycles
                tau = 0.008 if cyc == 0 else 0.005
                x1 = _refine_point(x0, Yoth, tau=tau, steps=100, lr=0.12)
                new_pts[int(pos)] = x1
        prop = cur.copy()
        for q, pt in new_pts.items():
            prop[int(q)] = pt
        prop = _norm_rows(prop)
        v_prop = _max_cos(prop)
        # simulated-annealing acceptance on true max
        accept = False
        if v_prop < cur_val:
            accept = True
        else:
            if Temp > 0 and rng.rand() < np.exp((cur_val - v_prop) / Temp):
                accept = True
        if accept:
            cur = prop.copy()
            cur_val = v_prop
            # 0.5-1s LP squeeze after each accepted repair
            secs = 0.5 + rng.rand() * 0.5
            remain = SEARCH_END - time.time()
            if remain < 0.6:
                pass
            else:
                secs = min(secs, remain)
                sq, vs = _lp_squeeze(cur, time.time() + secs, r0=0.004)
                # walk from accepted (squeezed) even if not global best
                cur = sq.copy(); cur_val = vs
                if vs < best_val - 1e-12:
                    best_val = vs; x_best = sq.copy()
        else:
            # also track global best if lucky unaccepted still better than best
            if v_prop < best_val - 1e-12:
                # quick squeeze to confirm
                remain = SEARCH_END - time.time()
                if remain > 0.8:
                    sq, vs = _lp_squeeze(prop, time.time() + 0.6, r0=0.003)
                    if vs < best_val:
                        best_val = vs; x_best = sq.copy()
        # occasional restart walk from best to avoid drift too far
        if it % 40 == 0 and rng.rand() < 0.5:
            cur = x_best.copy(); cur_val = best_val
        if it > 500:
            break

    rem = DEAD - time.time()
    if rem > 1.5:
        end = time.time() + min(rem - 0.6, 10.0)
        sq, vs = _lp_squeeze(x_best, end, r0=0.0015)
        if vs < best_val:
            best_val = vs; x_best = sq
    x_best = _norm_rows(x_best)
    bad = ~np.isfinite(x_best)
    if np.any(bad):
        x_best[bad] = 0.01
        x_best = _norm_rows(x_best)
    return x_best

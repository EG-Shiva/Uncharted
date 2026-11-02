import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _lp_squeeze(x, seconds, radius_init=7e-4):
    # short trust-region LP, strictly time-checked
    t_end = time.time() + seconds
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-13:
        G = x @ x.T
        iu = iu0
        cfull = G[iu]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 900:
            idx = np.where(sel)[0]
            # keep largest
            part = np.argpartition(-cfull[idx], 900)[:900]
            nsel = np.zeros_like(sel)
            nsel[idx[part]] = True
            sel = nsel
        csel = cfull[sel]
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        c1 = np.tile(np.arange(d), p)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        if time.time() >= t_end:
            break
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _deep_hole(others, rng, n_cand=200000, t_end=None):
    d = others.shape[1]
    best_c = None
    best_v = 1e9
    # chunk to bound memory
    chunk = 50000
    done = 0
    while done < n_cand:
        if t_end is not None and time.time() > t_end:
            break
        c = min(chunk, n_cand - done)
        cand = rng.standard_normal((c, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        md = (cand @ others.T).max(axis=1)
        j = int(np.argmin(md))
        if float(md[j]) < best_v:
            best_v = float(md[j]); best_c = cand[j].copy()
        done += c
    return best_c

def _refine_point(y, others, steps=150, beta=350.0, lr0=0.08, t_end=None):
    # minimise soft-max (LSE) of dots on sphere
    y = y / (np.linalg.norm(y) + 1e-12)
    for s in range(steps):
        if t_end is not None and (s % 50 == 0) and time.time() > t_end:
            break
        dots = others @ y  # (m,)
        m = float(np.max(dots))
        w = np.exp(beta * (dots - m))
        sw = np.sum(w)
        if not np.isfinite(sw) or sw <= 0:
            break
        g = (w[:, None] * others).sum(axis=0) / sw  # gradient of LSE
        # tangent projection (descent)
        gt = g - float(np.dot(g, y)) * y
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12:
            break
        gt /= nrm
        f = s / max(steps - 1, 1)
        lr = lr0 * (1.0 - 0.6 * f)
        y = y - lr * gt
        y /= (np.linalg.norm(y) + 1e-12)
    return y

def solve():
    rng = np.random.default_rng(20261101)
    t0 = time.time()
    HARD = t0 + 150.0
    # load incumbent
    xb = None
    bs = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < bs:
                    bs = s; xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.standard_normal((55, 21)))
        bs = _max_cos(xb)
    n, d = 55, 21
    x_best = xb.copy(); best = bs
    cur = xb.copy(); cur_score = bs
    T0, T1 = 0.0015, 1e-6
    it = 0
    while time.time() < HARD - 1.5:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / 150.0
        T = T0 * (T1 / T0) ** min(max(frac, 0.0), 1.0)
        # ---- choose connected cluster 4-8 ----
        k = 4 + int(rng.integers(0, 5))  # 4..8
        # contact graph near max
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        # adaptive threshold: include pairs within 2e-3 of max (dense) fallback
        thr = mx - 2e-3
        adj = G > thr
        deg = adj.sum(axis=1)
        if int(rng.random() * 10) < 7:
            # high-degree seed
            order = np.argsort(-deg)
            # pick among top 12 randomly
            seed = int(order[rng.integers(0, min(12, n))])
        else:
            seed = int(rng.integers(0, n))
        # BFS connected cluster
        cluster = [seed]
        seen = set([seed])
        frontier = [seed]
        # random neighbour order for diversity
        while len(cluster) < k and frontier:
            u = frontier.pop(0)
            nbrs = np.where(adj[u])[0]
            if len(nbrs) == 0:
                continue
            perm = rng.permutation(len(nbrs))
            for pi in perm:
                v = int(nbrs[pi])
                if v not in seen:
                    seen.add(v)
                    cluster.append(v)
                    frontier.append(v)
                    if len(cluster) >= k:
                        break
        if len(cluster) < k:
            # fill with random high-degree points
            order = np.argsort(-deg)
            for v in order:
                v = int(v)
                if v not in seen:
                    seen.add(v); cluster.append(v)
                    if len(cluster) >= k:
                        break
        cluster = np.array(cluster[:k], dtype=int)
        # random repair order
        porder = rng.permutation(k)
        cluster_ord = cluster[porder]
        frozen_mask = np.ones(n, dtype=bool)
        frozen_mask[cluster] = False
        X_frozen = cur[frozen_mask]  # (n-k, d)
        # ---- reinsert jointly ----
        # current values for deleted (will be overwritten)
        new_pts = cur[cluster_ord].copy()
        # cycle several times
        NCYC = 3
        for cyc in range(NCYC):
            if time.time() > HARD - 1.2:
                break
            for si, orig_idx in enumerate(cluster_ord):
                if time.time() > HARD - 1.2:
                    break
                # others = frozen + other new points (current estimates)
                others = np.vstack([X_frozen, np.delete(new_pts, si, axis=0)])
                if cyc == 0:
                    # deep hole sampling >=200k
                    hole = _deep_hole(others, rng, n_cand=200000, t_end=HARD - 1.0)
                    if hole is None:
                        hole = new_pts[si]
                    y0 = hole
                else:
                    y0 = new_pts[si]
                    # small jitter before refine to escape exact tie
                    if rng.random() < 0.5:
                        y0 = y0 + 0.005 * rng.standard_normal(d)
                        y0 /= (np.linalg.norm(y0) + 1e-12)
                yref = _refine_point(y0, others, steps=150, beta=350.0, lr0=0.08, t_end=HARD - 1.0)
                new_pts[si] = yref
        x_rep = cur.copy()
        x_rep[cluster_ord] = new_pts
        x_rep = _norm(x_rep)
        # quick screen on raw repaired (no LP yet)
        # LP squeeze 0.5-1s
        remain = HARD - time.time()
        if remain < 0.7:
            # no time for LP: evaluate raw
            v_raw = _max_cos(x_rep)
            # SA on raw
            if v_raw < cur_score or rng.random() < np.exp((cur_score - v_raw) / max(T, 1e-12)):
                cur, cur_score = x_rep, v_raw
                if v_raw < best - 1e-12:
                    x_best, best = x_rep.copy(), v_raw
            continue
        budget = 0.5 + float(rng.random()) * 0.5  # 0.5-1s
        budget = min(budget, remain - 0.4)
        if budget < 0.35:
            budget = 0.35
        x_pol = _lp_squeeze(x_rep, seconds=budget, radius_init=7e-4)
        v_pol = _max_cos(x_pol)
        # SA acceptance on true max, walk from accepted (not reset to incumbent)
        if v_pol < cur_score or rng.random() < np.exp((cur_score - v_pol) / max(T, 1e-12)):
            cur, cur_score = x_pol, v_pol
        if v_pol < best - 1e-12:
            x_best, best = x_pol.copy(), v_pol
        # occasional restart to best if wandering too high (but keep walking)
        if cur_score > best + 0.008 and rng.random() < 0.3:
            cur, cur_score = x_best.copy(), best
    x_best = _norm(x_best)
    return x_best

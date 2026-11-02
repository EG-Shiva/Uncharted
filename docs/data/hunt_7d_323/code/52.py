import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, init_radius=4e-3):
    if seconds <= 0.2:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    niter = 0
    while time.time() - start < seconds and radius > 1e-14 and niter < 60:
        niter += 1
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
        if cnt > 3000:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -3000)[-3000:]
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
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
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
            radius = min(radius * 1.5, 0.04)
        else:
            radius *= 0.5
    return x


def _relax(y, mask, steps, lr0, temp, rng=None):
    # vectorized LogSumExp tangential repulsion; mask bool or index array
    if isinstance(mask, np.ndarray) and mask.dtype == bool:
        midx = np.where(mask)[0]
    else:
        midx = np.asarray(mask, dtype=np.int64)
    if len(midx) == 0:
        return y
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) / temp)
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        WG = (W * G).sum(axis=1, keepdims=True)
        F = W @ y - WG * y
        F = F / S
        # decay lr slightly
        lr = lr0 / (1.0 + 0.02 * s)
        y[midx] = y[midx] - lr * F[midx]
        nrm = np.linalg.norm(y[midx], axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y[midx] = y[midx] / nrm
    return y


def solve():
    t0 = time.time()
    BUDGET = 160.0
    rng = np.random.default_rng(0)

    def remaining():
        return BUDGET - (time.time() - t0)

    bestX = None
    bestV = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < bestV:
                        bestV = v
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        bestX = a
        bestV = _max_cos(a)

    n, d = 323, 7
    all_idx = np.arange(n)

    # quick initial tiny LP touch if time allows? skip, go to surgery
    hub_cycle = 0
    outer = 0
    # precompute for cluster size schedule
    while remaining() > 12 and outer < 30:
        outer += 1
        # build contact graph at best
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        tol = 1e-7
        adj = [np.where(G[i] > m - tol)[0].tolist() for i in range(n)]
        # contact counts
        deg = np.array([len(a) for a in adj], dtype=np.int64)
        hub_order = np.argsort(-deg, kind="stable")
        # candidate hubs: top 10 + few random
        hubs = list(hub_order[:10])
        # add 2 random connected starts
        for _ in range(2):
            hubs.append(int(rng.integers(0, n)))
        for hi, hub in enumerate(hubs):
            if remaining() < 10:
                break
            # BFS neighbourhood 12-18
            K = 12 + int(rng.integers(0, 7))  # 12..18
            # randomize neighbour order for diversity except hub-priority
            visited = set([int(hub)])
            queue = [int(hub)]
            # BFS
            qpos = 0
            # shuffle adjacency once per hub using rng
            while len(visited) < K and qpos < len(queue):
                cur = queue[qpos]
                qpos += 1
                nbrs = adj[cur]
                if len(nbrs) > 1:
                    perm = rng.permutation(len(nbrs))
                    nbrs = [nbrs[i] for i in perm]
                for nb in nbrs:
                    if nb not in visited:
                        visited.add(int(nb))
                        queue.append(int(nb))
                        if len(visited) >= K:
                            break
            # if still small (isolated), fill with random BFS from visited
            if len(visited) < K:
                # expand by graph distance 2 or random nodes
                while len(visited) < K:
                    # pick random visited node, pick its best (highest cosine) non-visited partner
                    cand_pool = rng.choice(all_idx, size=min(50, n), replace=False)
                    added = False
                    for cp in cand_pool:
                        if int(cp) not in visited:
                            visited.add(int(cp))
                            queue.append(int(cp))
                            added = True
                            break
                    if not added:
                        break
            cluster = np.array(sorted(visited)[:K], dtype=np.int64)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[cluster] = False
            y = bestX.copy()
            # sequential max-min reinsertion from many random candidates
            # choose Ncand based on remaining time
            rem = remaining()
            if rem > 80:
                Ncand = 50000
            elif rem > 40:
                Ncand = 30000
            else:
                Ncand = 15000
            chunk = 10000
            nchunk = max(1, Ncand // chunk)
            Xa = y[keep_mask].copy()  # (n-K, d)
            # reinsert in random order to diversify
            order = cluster[rng.permutation(len(cluster))] if len(cluster) > 1 else cluster
            # map: we need to update y progressively
            new_list = []
            for k in order:
                if remaining() < 8:
                    break
                best_worst = np.inf
                best_pt = None
                for _c in range(nchunk):
                    cand = rng.standard_normal(size=(chunk, d))
                    cn = np.linalg.norm(cand, axis=1, keepdims=True)
                    cn[cn == 0] = 1.0
                    cand = cand / cn
                    dots = cand @ Xa.T
                    worst = dots.max(axis=1)
                    bi = int(np.argmin(worst))
                    bw = float(worst[bi])
                    if bw < best_worst:
                        best_worst = bw
                        best_pt = cand[bi].copy()
                y[k] = best_pt
                Xa = np.vstack([Xa, best_pt[None, :]])
            pre = _max_cos(y)
            # frozen-core optimisation then short all-point relaxation
            # fast smooth phases (no python pair loops)
            try:
                y = _relax(y, cluster, steps=120, lr0=0.05, temp=0.015)
                y = _relax(y, all_idx, steps=80, lr0=0.02, temp=0.010)
            except Exception:
                pass
            post = _max_cos(y)
            # keep candidates whose pre-LP max below ~0.670
            if post < 0.670:
                lp_sec = 4.0
                if remaining() < 15:
                    lp_sec = min(2.0, max(0.5, remaining() - 8))
                if lp_sec > 0.4 and post < bestV + 0.004:
                    try:
                        y = _lp_polish(y, seconds=lp_sec)
                    except Exception:
                        pass
                    post = _max_cos(y)
            if post < bestV and np.all(np.isfinite(y)):
                # normalise strictly
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                post = _max_cos(y)
                if post < bestV:
                    bestV = post
                    bestX = y.copy()
                    # refresh contact graph next outer loop quickly: break to recompute hubs
                    # continue trying other hubs from improved base without full outer recompute occasionally
                    if hi % 3 == 2:
                        break
            hub_cycle += 1
            if remaining() < 10:
                break
        if remaining() < 10:
            break

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX

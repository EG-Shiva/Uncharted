import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _gd300(X, alpha=100.0, lr=0.008, steps=300):
    x = X.copy()
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        Grad = E @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt /= nrm
        x = x - lr * Gt
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _lp_polish(X, seconds, radius_init=1e-3, cap=3000):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        mval = float(c_all.max())
        thresh = mval - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = cap
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    SEARCH_CAP = 130.0
    TOTAL_BUDGET = 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            if _max_cos(xr) < _max_cos(xb):
                xb = xr.copy()
    except Exception:
        pass

    best_val = _max_cos(xb)
    best_x = xb.copy()
    screened_best_x = best_x.copy()
    screened_best_val = best_val

    sizes = [12, 18, 24]
    outer = 0

    # precompute helper for BFS needs current incumbent graph
    while True:
        if time.time() - t0 > SEARCH_CAP:
            break
        outer += 1
        # rebuild contact graph from current global best (screened or incumbent)
        # use best_x (lowest true max found) for topology
        ref = best_x
        G = ref @ ref.T
        np.fill_diagonal(G, -2.0)
        mval = float(np.max(G))
        thresh = mval - 1e-5
        # degrees
        mask = G > thresh
        deg = mask.sum(axis=1)
        order = np.argsort(-deg)
        K = sizes[(outer - 1) % 3]
        seed_rank = ((outer - 1) // 3) % 15
        seed = int(order[seed_rank % 176])
        # adjacency lists
        neigh_list = [np.where(mask[i])[0] for i in range(176)]
        # BFS community
        visited = set()
        visited.add(seed)
        queue = [seed]
        qpos = 0
        # order neighbours by degree descending for determinism
        while len(visited) < K and qpos < len(queue):
            u = queue[qpos]
            qpos += 1
            nb = neigh_list[u]
            if len(nb) > 1:
                nb = nb[np.argsort(-deg[nb])]
            for v in nb:
                v = int(v)
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
                    if len(visited) >= K:
                        break
        if len(visited) < K:
            for idx in order:
                idx = int(idx)
                if idx not in visited:
                    visited.add(idx)
                    if len(visited) >= K:
                        break
        del_idx = np.array(sorted(visited), dtype=int)[:K]
        frozen_idx = np.array([i for i in range(176) if i not in visited], dtype=int)
        F0 = ref[frozen_idx].copy()

        # reinsert one at a time
        inserted = []
        cur_ok = True
        for kk in range(K):
            if time.time() - t0 > SEARCH_CAP:
                cur_ok = False
                break
            if len(inserted) == 0:
                Cur = F0
            else:
                Cur = np.vstack([F0] + inserted)
            # 20000 random candidates
            trials = rng.standard_normal((20000, 9))
            trials /= np.linalg.norm(trials, axis=1, keepdims=True)
            dots = trials @ Cur.T
            maxc = dots.max(axis=1)
            # best 20 by maxc
            part = np.argpartition(maxc, 20)[:20]
            Y = trials[part].copy()
            # local tangent refinement of best 20
            # 40 steps vectorized
            for _ in range(40):
                C = Y @ Cur.T
                Mr = C.max(axis=1, keepdims=True)
                W = np.exp(80.0 * (C - Mr))
                Grad = W @ Cur
                dot2 = np.sum(Grad * Y, axis=1, keepdims=True)
                Gt = Grad - dot2 * Y
                s = np.sum(W, axis=1, keepdims=True) + 1e-12
                Gt /= s
                Y = Y - 0.08 * Gt
                nr = np.linalg.norm(Y, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                Y /= nr
            Cf = Y @ Cur.T
            maxf = Cf.max(axis=1)
            # tie-break centered sum
            tie = np.sum(np.exp(80.0 * (Cf - maxf[:, None])), axis=1)
            # lexsort: primary maxf, secondary tie
            o = np.lexsort((tie, maxf))
            best_y = Y[o[0]]
            inserted.append(best_y[None, :].copy())
        if not cur_ok:
            break
        # assemble full code
        Xnew = np.empty((176, 9))
        Xnew[frozen_idx] = F0
        ins_mat = np.vstack(inserted)
        Xnew[del_idx] = ins_mat
        # 300 steps normalised repulsion GD (no LP)
        Xnew = _gd300(Xnew, alpha=100.0, lr=0.008, steps=300)
        v = _max_cos(Xnew)
        if v < screened_best_val:
            screened_best_val = v
            screened_best_x = Xnew.copy()
        if v < best_val - 1e-12:
            best_val = v
            best_x = Xnew.copy()

    # LP polish only single best screened candidate for remaining time
    rem = TOTAL_BUDGET - (time.time() - t0)
    rem = float(min(max(rem - 0.5, 0.0), 38.0))
    final_x = screened_best_x.copy()
    # if screened never beat incumbent, polishing incumbent is same; keep screened
    if rem > 1.0:
        try:
            final_x = _lp_polish(screened_best_x, seconds=rem, radius_init=1e-3)
        except Exception:
            final_x = screened_best_x
    # choose best among final, best_x, screened
    cand_list = [(final_x, _max_cos(final_x)), (screened_best_x, screened_best_val), (best_x, best_val)]
    cand_list.sort(key=lambda t: t[1])
    out = cand_list[0][0]
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _degrees(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - tol).sum(axis=1)
    return m, cnt, G

def _anneal_free(Y, free_mask, iters=900, beta0=50.0, beta1=500.0, step0=0.03, step1=0.002):
    n, d = Y.shape
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ls0, ls1 = np.log(step0), np.log(step1)
    free_idx = np.where(free_mask)[0]
    if len(free_idx) == 0:
        return Y
    for it in range(iters):
        frac = it / max(1, iters - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        step = float(np.exp(ls0 + (ls1 - ls0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        # max over pairs involving free points
        m = float(G[free_mask, :].max())
        W = np.exp(beta * (G - m))
        W[W < 1e-7] = 0.0
        np.fill_diagonal(W, 0.0)
        WX = W @ Y
        s = (W * G).sum(axis=1, keepdims=True)
        grad = WX - s * Y
        gfree = grad[free_mask]
        gn = float(np.linalg.norm(gfree, axis=1).max()) if len(gfree) else 0.0
        if not np.isfinite(gn) or gn == 0:
            continue
        Y[free_idx] = Y[free_idx] - (step / gn) * gfree
        Y[free_idx] /= np.linalg.norm(Y[free_idx], axis=1, keepdims=True)
    return Y

def _lp_polish_mask(X, seconds, fix=None, radius0=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    if fix is None:
        fix = np.zeros(n, dtype=bool)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            part = np.argpartition(cfull[idx], len(idx) - 3000)[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[part]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]; c = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bnds = []
        for i in range(n):
            if fix[i]:
                bnds.extend([(0.0, 0.0)] * d)
            else:
                bnds.extend([(-radius, radius)] * d)
        bnds.append((None, None))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _bfs_cluster(adj, seed, k, Gfull):
    n = len(adj)
    seen = set([int(seed)])
    q = [int(seed)]
    qi = 0
    while qi < len(q) and len(seen) < k:
        u = q[qi]; qi += 1
        for v in adj[u]:
            if v not in seen:
                seen.add(int(v))
                q.append(int(v))
                if len(seen) >= k:
                    break
    # if component too small, fill by nearest dots to cluster
    if len(seen) < k:
        members = list(seen)
        # score each outside node by max dot to members
        rem = np.array([i for i in range(n) if i not in seen])
        if len(rem) > 0:
            sub = Gfull[np.ix_(rem, members)].max(axis=1)
            order = np.argsort(-sub)
            need = k - len(seen)
            for t in range(min(need, len(rem))):
                seen.add(int(rem[order[t]]))
    return np.array(sorted(seen), dtype=int)

def _greedy_rebuild(Xfrozen, k, rng, n_cand=100000):
    d = Xfrozen.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    if Xfrozen.shape[0] > 0:
        sf = (C @ Xfrozen.T).max(axis=1)
    else:
        sf = np.full(n_cand, -2.0)
    s_self = np.full(n_cand, -2.0)
    chosen = []
    # for diversity, randomly permute tie-breaking: add tiny noise to sf
    # (deterministic via rng)
    jitter = rng.normal(size=n_cand) * 1e-12
    for _ in range(k):
        tot = np.maximum(sf, s_self) + jitter * 0.0
        # add small random tie break only first iteration to diversify
        idx = int(np.argmin(tot + (jitter if len(chosen) == 0 else 0.0)))
        pt = C[idx].copy()
        chosen.append(pt)
        upd = C @ pt
        np.maximum(s_self, upd, out=s_self)
        s_self[idx] = 2.0
        sf[idx] = 2.0
    return np.stack(chosen, axis=0)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(20261101)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    def time_left():
        return deadline - time.time()
    # elite population of 3 distinct
    elites = [(best_val, best.copy())]
    trial = 0
    sizes = [20, 25, 30, 35, 40, 48]
    while time_left() > 13.0:
        trial += 1
        # pick base: 70% best, else random elite
        elites_sorted = sorted(elites, key=lambda z: z[0])
        if len(elites_sorted) > 1 and rng.random() < 0.3:
            bi = int(rng.integers(len(elites_sorted)))
            base_val, base = elites_sorted[bi][0], elites_sorted[bi][1].copy()
        else:
            base_val, base = elites_sorted[0][0], elites_sorted[0][1].copy()
        m, cnt, Gfull = _degrees(base, tol=1e-9)
        # adjacency for BFS
        Gc = Gfull.copy()
        np.fill_diagonal(Gc, -2.0)
        adj = [np.where(Gc[i] >= m - 1e-9)[0].tolist() for i in range(n)]
        # seed pool
        rats = np.where(cnt <= 0)[0]
        if len(rats) > 0:
            r = int(rats[0])
            # nearest points to rattler
            dots_r = Gfull[r].copy()
            dots_r[r] = -2.0
            pool = np.argsort(-dots_r)[:12]
        else:
            hi = np.where(cnt >= 11)[0]
            if len(hi) == 0:
                pool = np.argsort(-cnt)[:12]
            else:
                pool = hi
        seed = int(pool[int(rng.integers(len(pool)))])
        k = int(sizes[int(rng.integers(len(sizes)))])
        # vary k slightly
        k = int(np.clip(k + int(rng.integers(-3, 4)), 20, 50))
        cluster = _bfs_cluster(adj, seed, k, Gfull)
        free_mask = np.zeros(n, dtype=bool)
        free_mask[cluster] = True
        frozen_idx = np.where(~free_mask)[0]
        Xfrozen = base[frozen_idx]
        # rebuild
        nc = 100000 if time_left() > 40 else 60000
        try:
            new_pts = _greedy_rebuild(Xfrozen, len(cluster), rng, n_cand=nc)
        except Exception:
            continue
        Y = base.copy()
        Y[cluster] = new_pts
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # free-only anneal beta 50->500
        if trial % 3 == 0:
            Y = _anneal_free(Y, free_mask, iters=1000, beta0=50.0, beta1=500.0, step0=0.035, step1=0.002)
        elif trial % 3 == 1:
            Y = _anneal_free(Y, free_mask, iters=750, beta0=50.0, beta1=500.0, step0=0.05, step1=0.003)
        else:
            Y = _anneal_free(Y, free_mask, iters=1200, beta0=40.0, beta1=600.0, step0=0.03, step1=0.0015)
        v_free = _max_cos(Y)
        # frozen LP polish (hold frozen)
        rem = time_left()
        if rem < 13.0:
            break
        fix = ~free_mask
        lp1 = 0.7 if rem > 50 else 0.45
        if time_left() < lp1 + 11.5:
            continue
        Y1 = _lp_polish_mask(Y, seconds=lp1, fix=fix, radius0=2e-3)
        v1 = _max_cos(Y1)
        # unfreeze full polish if promising
        # accept if within tolerance to explore new topology
        thresh_full = elites_sorted[0][0] + (0.004 if trial % 2 == 0 else 0.001)
        if v1 < thresh_full or v_free < thresh_full:
            lp2 = 0.9 if rem > 50 else 0.55
            if time_left() < lp1 + lp2 + 10.5:
                # keep best free result anyway
                cand_val, cand = (v1, Y1) if v1 < v_free else (v_free, Y)
            else:
                Y2 = _lp_polish_mask(Y1, seconds=lp2, fix=None, radius0=1e-3)
                v2 = _max_cos(Y2)
                if v2 <= v1:
                    cand_val, cand = v2, Y2
                else:
                    cand_val, cand = v1, Y1
            # update elites: keep up to 3 distinct best
            # distinct if val differs by >1e-12 or config differs
            worst = max(z[0] for z in elites)
            if cand_val < worst + 1e-12 or len(elites) < 3:
                # avoid exact duplicates
                dup = any(abs(cand_val - z[0]) < 1e-12 for z in elites)
                if not dup:
                    elites.append((cand_val, cand.copy()))
                    elites = sorted(elites, key=lambda z: z[0])[:3]
                elif cand_val < elites_sorted[0][0]:
                    elites = sorted(elites, key=lambda z: z[0])
                    elites[0] = (cand_val, cand.copy())
        # occasional light global shake of worst elite to avoid stagnation: handled by next trials
    elites_sorted = sorted(elites, key=lambda z: z[0])
    best_val, best = elites_sorted[0][0], elites_sorted[0][1].copy()
    rem = deadline - time.time()
    if rem > 2.0:
        best = _lp_polish_mask(best, seconds=max(1.0, rem - 1.0), fix=None, radius0=1e-3)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

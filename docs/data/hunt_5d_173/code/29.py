import time
import numpy as np
from collections import deque
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, free_idx=None, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = radius0
    start = time.time()
    iu = np.triu_indices(n, 1)
    free_mask = None
    if free_idx is not None:
        free_set = set(int(v) for v in free_idx)
        free_mask = np.zeros(n, dtype=bool)
        free_mask[list(free_set)] = True
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = iu_c > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(iu_c[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        if free_mask is None:
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        else:
            bounds = []
            for i in range(n):
                if free_mask[i]:
                    bounds.extend([(-radius, radius)] * d)
                else:
                    bounds.extend([(0.0, 0.0)] * d)
            bounds.append((None, None))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _anneal_free(Y, F, steps_per=150, betas=(50.0, 150.0, 300.0, 500.0)):
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    K = Y.shape[0]
    Nf = F.shape[0]
    for beta in betas:
        lr = 0.02 * (50.0 / beta) ** 0.3
        for _ in range(steps_per):
            Cf = Y @ F.T  # K x Nf
            G = Y @ Y.T
            cmax = Cf.max() if Nf > 0 else -2.0
            if K > 1:
                # max off-diagonal
                tmp = G.copy()
                np.fill_diagonal(tmp, -2.0)
                gm = float(tmp.max())
                if gm > cmax:
                    cmax = gm
            # weights
            Wf = np.exp(beta * (Cf - cmax))  # K x Nf
            # symmetric free-free weights
            W = np.exp(beta * (G - cmax))
            np.fill_diagonal(W, 0.0)
            Z = Wf.sum() + W.sum() * 0.5
            if not np.isfinite(Z) or Z <= 0:
                break
            grad = (Wf @ F + W @ Y) / Z
            # tangent projection
            dot = (grad * Y).sum(axis=1, keepdims=True)
            grad = grad - dot * Y
            Y = Y - lr * grad
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
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
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    pool = [(best_val, best.copy())]

    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        mask = G >= m - 1e-9
        cnt = mask.sum(axis=1)
        adj = mask
        return m, cnt, G, adj

    sizes_cycle = [22, 28, 34, 40, 46, 50, 25, 30, 35, 45, 20, 38]
    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        # pick base from pool
        if len(pool) > 1 and (trial % 3 == 2):
            bi = int(rng.integers(0, len(pool)))
            base_val, base = pool[bi]
            base = base.copy()
        else:
            base_val, base = pool[0][0], pool[0][1].copy()

        m, cnt, Gfull, adj = contact_info(base)
        # cluster size
        sz0 = sizes_cycle[trial % len(sizes_cycle)]
        K = int(sz0 + rng.integers(-3, 4))
        K = max(20, min(50, K))
        K = min(K, n - 5)
        # seed choice
        ratt = np.where(cnt == 0)[0]
        high = np.where(cnt >= 11)[0]
        seed = None
        if (trial % 2 == 0) and len(ratt) > 0:
            r = int(ratt[trial % len(ratt)])
            sims = base[r] @ base.T
            sims[r] = -2.0
            order = np.argsort(-sims)
            top = order[:5]
            seed = int(top[int(rng.integers(0, len(top)))])
        else:
            if len(high) > 0:
                seed = int(high[int(rng.integers(0, len(high)))])
            else:
                seed = int(np.argmax(cnt))
        # BFS cluster
        visited = set([seed])
        q = deque([seed])
        # precompute neighbor lists
        while q and len(visited) < K:
            cur = q.popleft()
            nbrs = np.where(adj[cur])[0].tolist()
            if len(nbrs) > 1:
                # shuffle to vary shape
                nbrs = [nbrs[i] for i in rng.permutation(len(nbrs))]
            for nb in nbrs:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
                    if len(visited) >= K:
                        break
        # if component too small, grow by nearest to cluster
        C = list(visited)
        if len(C) < K:
            Cset = set(C)
            # iteratively add closest outside point
            while len(C) < K:
                # max dot to cluster
                sub = base[C]  # c x d
                rest = [i for i in range(n) if i not in Cset]
                R = base[rest]  # r x d
                dots = R @ sub.T
                md = dots.max(axis=1)
                pick = rest[int(np.argmax(md))]
                C.append(pick)
                Cset.add(pick)
        else:
            C = C[:K]
        C = np.array(sorted(C), dtype=int)
        Fidx = np.array([i for i in range(n) if i not in set(C.tolist())], dtype=int)
        frozen = base[Fidx].copy()

        # greedy rebuild sampling
        Ncand = 70000
        if rem < 40:
            Ncand = 40000
        cand = rng.normal(size=(Ncand, d))
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        # max to frozen
        Cf = cand @ frozen.T
        maxf = Cf.max(axis=1)
        # prefilter to best M
        M = 8000
        if M < Ncand:
            part = np.argpartition(maxf, M)[:M]
            cand_f = cand[part]
            maxf_f = maxf[part]
        else:
            cand_f = cand
            maxf_f = maxf
        Nf2 = len(cand_f)
        # greedy select K
        cur = np.full(Nf2, -2.0)
        chosen = []
        chosen_vecs = []
        avail = np.ones(Nf2, dtype=bool)
        # for speed, keep indices
        for k in range(K):
            tot = np.maximum(maxf_f, cur)
            tot[~avail] = 1e9
            j = int(np.argmin(tot))
            chosen.append(j)
            avail[j] = False
            v = cand_f[j]
            chosen_vecs.append(v)
            # update cur with dot to v
            dots = cand_f @ v
            # only for avail
            cur = np.maximum(cur, dots)
        Y0 = np.stack(chosen_vecs, axis=0)

        # anneal only free points
        Yopt = _anneal_free(Y0, frozen, steps_per=130,
                            betas=(50.0, 150.0, 300.0, 500.0))
        Xt = base.copy()
        Xt[C] = Yopt

        # LP with frozen fixed, then full
        rem2 = deadline - time.time()
        if rem2 < 8.0:
            break
        t_froz = 1.0
        t_full = 1.3
        if rem2 < 25:
            t_froz = 0.7
            t_full = 0.8
        X1 = _lp_polish(Xt, t_froz, free_idx=C)
        X2 = _lp_polish(X1, t_full, free_idx=None)
        v2 = _max_cos(X2)
        if v2 < pool[0][0] - 1e-12:
            pool.insert(0, (v2, X2.copy()))
            pool = pool[:4]
            best_val = v2
            best = X2.copy()
        elif len(pool) < 4:
            # distinctness check
            distinct = True
            for (vv, xx) in pool:
                diff = float(np.linalg.norm(np.sort((X2 @ X2.T).ravel()) - np.sort((xx @ xx.T).ravel())) / n)
                if diff < 1e-4 and abs(vv - v2) < 1e-9:
                    distinct = False
                    break
            if distinct and v2 < pool[0][0] + 5e-4:
                pool.append((v2, X2.copy()))
                # keep sorted by val
                pool = sorted(pool, key=lambda t: t[0])[:4]
        trial += 1

    # final polish on global best
    best = pool[0][1].copy()
    rem = deadline - time.time()
    if rem > 1.5:
        best = _lp_polish(best, rem - 0.7, free_idx=None)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

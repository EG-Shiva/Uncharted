import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds):
    X = _norm(np.asarray(X, dtype=np.float64).copy())
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0:
        return X
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if not np.any(sel):
            break
        I = iu0[sel]; J = iu1[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        cols_I = I[:, None] * d + np.arange(d)
        cols_J = J[:, None] * d + np.arange(d)
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = cols_I
        cols_mat[:, d:2 * d] = cols_J
        cols_mat[:, 2 * d] = n * d
        vals_mat = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_mat[:, :d] = X[J]
        vals_mat[:, d:2 * d] = X[I]
        vals_mat[:, 2 * d] = -1.0
        rows_mat = np.repeat(np.arange(p)[:, None], 2 * d + 1, axis=1)
        A_ub = coo_matrix((vals_mat.ravel(), (rows_mat.ravel(), cols_mat.ravel())), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _smooth_opt(X0, beta0, beta1, iters, lr, rng=None):
    X = np.asarray(X0, dtype=np.float64).copy()
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    M = np.zeros_like(X); V = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    for t in range(1, iters + 1):
        frac = (t - 1) / max(iters - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        cur_lr = lr * (1.0 - 0.5 * frac)
        G = X @ X.T
        gm = float(np.max(G[iu0, iu1]))
        B = np.exp(beta * (G - gm))
        np.fill_diagonal(B, 0.0)
        Grad = B @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        mhat = M / (1.0 - (b1 ** t))
        vhat = V / (1.0 - (b2 ** t))
        X = X - cur_lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm(X)
    return X

def _contacts_info(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - thr, axis=1)
    return m, cnt, G

def _greedy_maximin(rem, k, rng, n_cand=20000):
    d = rem.shape[1]
    C = rng.standard_normal((n_cand, d))
    C = _norm(C)
    # max cosine to rem
    S = C @ rem.T
    smax = np.max(S, axis=1)
    cur = rem
    newpts = []
    for _ in range(k):
        # pick best (minimal smax); diversify among top 40 randomly
        idx_sorted = np.argpartition(smax, 40)[:40]
        # choose the absolute best most of time, random among top sometimes
        if rng.random() < 0.7:
            pick = int(np.argmin(smax))
        else:
            # random among top 40
            pick = int(idx_sorted[rng.integers(0, len(idx_sorted))])
        pt = C[pick].copy()
        # tiny jitter to avoid exact duplicates
        pt = pt + 0.002 * rng.standard_normal(d)
        pt = pt / np.linalg.norm(pt)
        newpts.append(pt)
        upd = C @ pt
        np.maximum(smax, upd, out=smax)
        smax[pick] = 2.0
        cur = np.vstack([cur, pt[None, :]])
        # refresh candidate pool halfway to avoid depletion
        if len(newpts) == k // 2 + 1 and k > 4:
            pass
    return np.vstack([rem] + [p[None, :] for p in newpts]) if newpts else rem.copy()

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    HARD = 174.0
    n, d = 173, 5
    X_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (n, d):
                continue
            A = _norm(A)
            v = _max_cos(A)
            if v < best_val:
                best_val = v
                X_best = A.copy()
        except Exception:
            continue
    if X_best is None:
        X_best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(X_best)
    # initial quick polish of loaded best to ensure machine optimum baseline
    # (skip heavy polish, keep time for exploration)

    explore_end = t0 + HARD - 26.0
    it = 0
    while time.time() < explore_end:
        it += 1
        if time.time() >= explore_end:
            break
        X0 = X_best.copy()
        try:
            m0, cnt, G0 = _contacts_info(X0, thr=1e-7)
        except Exception:
            cnt = np.zeros(n)
            G0 = X0 @ X0.T
            m0 = best_val
        rattlers = np.where(cnt == 0)[0]
        order_hi = np.argsort(-cnt)  # descending contacts
        order_lo = np.argsort(cnt)
        # worst pairs info
        mode = it % 7
        try:
            if mode == 0:
                sigma = 0.02
                Xp = _norm(X0 + sigma * rng.standard_normal((n, d)))
                Xs = _smooth_opt(Xp, 50.0, 160.0, 900, 0.012)
            elif mode == 1:
                sigma = 0.06
                Xp = _norm(X0 + sigma * rng.standard_normal((n, d)))
                Xs = _smooth_opt(Xp, 40.0, 140.0, 1100, 0.015)
            elif mode == 2:
                # rattler-activated patch excision
                hub = int(order_hi[0])
                # neighbors of hub by cosine
                sims_hub = G0[hub].copy()
                sims_hub[hub] = -2
                nb_hub = np.argsort(-sims_hub)[:6]
                drops = set([hub])
                for q in nb_hub[:3]:
                    drops.add(int(q))
                if len(rattlers) > 0:
                    drops.add(int(rattlers[0]))
                    # also neighbors closest to rattler (to open its void)
                    r = int(rattlers[0])
                    sims_r = (X0 @ X0[r])
                    sims_r[r] = -2
                    nbr = np.argsort(-sims_r)[:3]
                    for q in nbr[:2]:
                        drops.add(int(q))
                # fill to k=6..9
                k_target = 6 + int(rng.integers(0, 4))
                pi = 0
                while len(drops) < k_target:
                    drops.add(int(order_hi[pi % 20]))
                    pi += 1
                drops = sorted(list(drops))[:k_target]
                mask = np.ones(n, dtype=bool); mask[np.array(drops)] = False
                rem = X0[mask]
                Xp = _greedy_maximin(rem, len(drops), rng, n_cand=20000)
                Xp = _norm(Xp + 0.008 * rng.standard_normal((n, d)))
                Xs = _smooth_opt(Xp, 40.0, 150.0, 1300, 0.015)
            elif mode == 3:
                # overload cap excision: remove spherical cap
                u = rng.standard_normal(d); u /= np.linalg.norm(u)
                proj = X0 @ u
                k = 8 + int(rng.integers(0, 6))  # 8..13
                drop = np.argsort(-proj)[:k]
                mask = np.ones(n, dtype=bool); mask[drop] = False
                rem = X0[mask]
                Xp = _greedy_maximin(rem, k, rng, n_cand=15000)
                Xp = _norm(Xp + 0.01 * rng.standard_normal((n, d)))
                Xs = _smooth_opt(Xp, 30.0, 130.0, 1300, 0.016)
            elif mode == 4:
                # random connected contact-component excision
                # build contact adjacency at tight threshold
                thr = 1e-6
                # start from hub, BFS
                adj = (G0 > m0 - 1e-7)
                np.fill_diagonal(adj, False)
                start_node = int(order_hi[int(rng.integers(0, 5))])
                comp = set([start_node])
                frontier = [start_node]
                while frontier and len(comp) < 10:
                    cur_nd = frontier.pop()
                    nbrs = np.where(adj[cur_nd])[0].tolist()
                    rng.shuffle(nbrs)
                    for nb in nbrs:
                        if nb not in comp:
                            comp.add(int(nb))
                            frontier.append(int(nb))
                            break
                    if len(frontier) == 0 and len(comp) < 8:
                        # add random high-contact node
                        for cand in order_hi:
                            if int(cand) not in comp:
                                frontier.append(int(cand))
                                break
                k = max(5, len(comp))
                drops = sorted(list(comp))[:k]
                if len(rattlers) > 0 and rng.random() < 0.5:
                    if int(rattlers[0]) not in drops:
                        drops = drops[:-1] + [int(rattlers[0])]
                mask = np.ones(n, dtype=bool); mask[np.array(drops)] = False
                rem = X0[mask]
                Xp = _greedy_maximin(rem, len(drops), rng, n_cand=20000)
                Xp = _norm(Xp + 0.008 * rng.standard_normal((n, d)))
                Xs = _smooth_opt(Xp, 45.0, 170.0, 1400, 0.014)
            elif mode == 5:
                # large shake + partial interpolation to random
                sigma = 0.12 + 0.10 * rng.random()
                Xp = _norm(X0 + sigma * rng.standard_normal((n, d)))
                # interpolate a few points to random positions
                k = int(rng.integers(4, 10))
                sel = rng.choice(n, size=k, replace=False)
                R = _norm(rng.standard_normal((k, d)))
                Xp[sel] = R
                Xp = _norm(Xp)
                Xs = _smooth_opt(Xp, 30.0, 110.0, 1500, 0.018)
            else:
                # drop lowest-contact + highest-contact mix
                k = 5 + int(rng.integers(0, 4))
                drops = set()
                drops.add(int(order_hi[int(rng.integers(0, 3))]))
                drops.add(int(order_hi[int(rng.integers(3, 15))]))
                if len(rattlers) > 0:
                    drops.add(int(rattlers[int(rng.integers(0, len(rattlers)))]))
                while len(drops) < k:
                    drops.add(int(rng.integers(0, n)))
                drops = sorted(list(drops))[:k]
                mask = np.ones(n, dtype=bool); mask[np.array(drops)] = False
                rem = X0[mask]
                Xp = _greedy_maximin(rem, len(drops), rng, n_cand=20000)
                Xp = _norm(Xp + 0.012 * rng.standard_normal((n, d)))
                Xs = _smooth_opt(Xp, 35.0, 140.0, 1200, 0.015)
        except Exception:
            continue
        try:
            v_sm = _max_cos(Xs)
        except Exception:
            continue
        # accept for LP if within window
        if v_sm < best_val + 0.005:
            remain = explore_end - time.time()
            if remain < 0.8:
                if v_sm < best_val:
                    X_best = Xs; best_val = v_sm
                continue
            budget = min(3.5, remain)
            if v_sm > best_val + 0.002:
                budget = min(1.5, remain)
            try:
                Xlp = _lp_polish(Xs, budget)
                v_lp = _max_cos(Xlp)
            except Exception:
                continue
            if v_lp < best_val:
                X_best = Xlp; best_val = v_lp
        else:
            if v_sm < best_val:
                X_best = Xs; best_val = v_sm
        if time.time() >= explore_end:
            break
    elapsed = time.time() - t0
    rem = HARD - elapsed
    rem = float(np.clip(rem, 3.0, 28.0))
    if elapsed < 177.0:
        try:
            X_best = _lp_polish(X_best, min(rem, 177.0 - elapsed))
        except Exception:
            pass
    X_best = _norm(X_best)
    return X_best

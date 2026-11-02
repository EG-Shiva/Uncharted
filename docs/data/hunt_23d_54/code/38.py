import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _greedy_clique_partition(G, maxv, thr, rng, nblock=6, bsize=9):
    n = G.shape[0]
    # non-contact adjacency: clearly below max
    A = (G < (maxv - thr))
    np.fill_diagonal(A, False)
    A = A & A.T
    remaining = list(range(n))
    assign = np.full(n, -1, dtype=int)
    for b in range(nblock):
        if len(remaining) <= bsize or b == nblock - 1:
            for v in remaining:
                assign[v] = b
            break
        # try several seeds, greedy expansion to clique-like set
        best_set = None
        best_score = -1
        # candidate seeds: sample up to 20
        seeds = remaining if len(remaining) <= 20 else list(rng.choice(remaining, 20, replace=False))
        for s in seeds:
            cur = [s]
            curset = {s}
            # candidates connected to all cur
            while len(cur) < bsize:
                # find node in remaining\cur maximizing connections to cur
                best_c = -1
                best_cdeg = -1
                for c in remaining:
                    if c in curset:
                        continue
                    ok = True
                    # count connections to cur (must be fully connected to keep clique)
                    conn = 0
                    for u in cur:
                        if A[c, u]:
                            conn += 1
                    # require fully connected for clique; if none, allow best partial
                    if conn == len(cur):
                        # tie-break by degree to remaining
                        deg = int(np.sum(A[c, remaining]))
                        if deg > best_cdeg:
                            best_cdeg = deg
                            best_c = c
                if best_c == -1:
                    break
                cur.append(best_c)
                curset.add(best_c)
            # score: size then internal density
            sz = len(cur)
            if sz > best_score:
                best_score = sz
                best_set = list(cur)
                if sz == bsize:
                    break
        if best_set is None or len(best_set) == 0:
            best_set = remaining[:bsize]
        # if clique smaller than bsize, fill with nodes least connected to other blocks greedily
        if len(best_set) < bsize:
            # fill by minimal mean dot to best_set
            cand = [v for v in remaining if v not in set(best_set)]
            # sort by mean G to best_set ascending
            means = [(np.mean(G[v, best_set]), v) for v in cand]
            means.sort()
            need = bsize - len(best_set)
            for _, v in means[:need]:
                best_set.append(v)
        for v in best_set:
            assign[v] = b
        remaining = [v for v in remaining if assign[v] == -1]
    # repair sizes to exactly 9 (in case of rounding)
    # enforce by moving
    for b in range(nblock):
        while int(np.sum(assign == b)) > bsize:
            idx = np.where(assign == b)[0]
            # move node with highest mean intra dot
            scores = []
            for v in idx:
                mates = [u for u in idx if u != v]
                m = float(np.mean(G[v, mates])) if len(mates) else 0.0
                scores.append((m, v))
            scores.sort(reverse=True)
            vmove = scores[0][1]
            # target block with space and minimal mean dot
            best_t = -1
            best_tm = 1e9
            for t in range(nblock):
                if int(np.sum(assign == t)) >= bsize:
                    continue
                mates = np.where(assign == t)[0]
                m = float(np.mean(G[vmove, mates])) if len(mates) else 0.0
                if m < best_tm:
                    best_tm = m
                    best_t = t
            if best_t == -1:
                break
            assign[vmove] = best_t
    # any unassigned
    for v in range(n):
        if assign[v] == -1:
            # put in smallest block
            sizes = [int(np.sum(assign == b)) for b in range(nblock)]
            assign[v] = int(np.argmin(sizes))
    return assign


def _rebalance(assign, G, nblock=6, bsize=9):
    n = len(assign)
    # one pass: fix overfull
    for _ in range(10):
        sizes = np.array([int(np.sum(assign == b)) for b in range(nblock)])
        if np.all(sizes == bsize):
            break
        # find overfull
        over = [b for b in range(nblock) if sizes[b] > bsize]
        under = [b for b in range(nblock) if sizes[b] < bsize]
        if not over or not under:
            break
        b = over[0]
        idx = np.where(assign == b)[0]
        # pick worst member (highest mean to block)
        best_v = -1
        best_s = -1e9
        for v in idx:
            mates = idx[idx != v]
            m = float(np.mean(G[v, mates])) if len(mates) else 0.0
            if m > best_s:
                best_s = m
                best_v = v
        # best under block
        best_t = under[0]
        best_tm = 1e9
        for t in under:
            mates = np.where(assign == t)[0]
            m = float(np.mean(G[best_v, mates])) if len(mates) else 0.0
            if m < best_tm:
                best_tm = m
                best_t = t
        assign[best_v] = best_t
    return assign


def _reassign_by_neighborhood(x, assign, nblock=6, bsize=9):
    n = x.shape[0]
    G = x @ x.T
    newa = np.full(n, -1, dtype=int)
    # mean dot to each block
    means = np.zeros((n, nblock))
    for b in range(nblock):
        members = np.where(assign == b)[0]
        if len(members) == 0:
            means[:, b] = 0.0
        else:
            means[:, b] = G[:, members].mean(axis=1)
    # greedy assignment in order of confidence: smallest margin
    # initial wish
    wish = np.argmin(means, axis=1)
    # assign respecting capacity by sorting by advantage
    # advantage = smallest - second smallest (more negative = stronger)
    order = []
    for i in range(n):
        s = np.sort(means[i])
        adv = s[0] - s[1]  # <=0, more negative stronger
        order.append((adv, i))
    order.sort()
    sizes = np.zeros(nblock, dtype=int)
    for _, i in order:
        b = wish[i]
        if sizes[b] < bsize:
            newa[i] = b
            sizes[b] += 1
        else:
            # pick best available
            ob = np.argsort(means[i])
            placed = False
            for bb in ob:
                if sizes[bb] < bsize:
                    newa[i] = bb
                    sizes[bb] += 1
                    placed = True
                    break
            if not placed:
                newa[i] = b
    return newa


def _block_anneal(x0, assign0, steps, k0, k1, lr0, lr1, d0, d1, eq0, eq1, win, seed, t0, deadline, reass_every=400):
    rng = np.random.RandomState(seed)
    x = x0.copy()
    n, d = x.shape
    nblock = int(assign0.max()) + 1
    assign = assign0.copy()
    iu = np.triu_indices(n, 1)
    # masks depend on assign; rebuild on reassign
    best_local = _max_cos(x)
    bx = x.copy()
    ba = assign.copy()
    v = np.zeros_like(x)
    mu = 0.85
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if time.time() > deadline:
            break
        f = s / max(steps - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        delta = d0 + (d1 - d0) * f
        eqw = eq0 + (eq1 - eq0) * f
        if s % reass_every == 0 and s > 0:
            assign = _reassign_by_neighborhood(x, assign)
            # check improvement potential: keep
        G = x @ x.T
        # inter/intra masks
        same = (assign[:, None] == assign[None, :])
        # current inter max
        Gm = G.copy()
        # mask intra to -inf for inter max
        Gm[same] = -2.0
        np.fill_diagonal(Gm, -2.0)
        m_inter = float(np.max(Gm))
        # weights
        W = np.zeros_like(G)
        # inter: near-max window
        sel_inter = (~same) & (G > (m_inter - win))
        np.fill_diagonal(sel_inter, False)
        if np.any(sel_inter):
            W[sel_inter] = np.exp(k * (G[sel_inter] - m_inter))
        # intra penalty above m_inter - delta
        thr_intra = m_inter - delta
        sel_intra = same & (G > thr_intra)
        np.fill_diagonal(sel_intra, False)
        if np.any(sel_intra):
            lam = 8.0  # fixed strong push
            W[sel_intra] += lam * (G[sel_intra] - thr_intra) * 50.0
        # equality penalty: for inter in window, pull toward window mean
        if eqw > 1e-12 and np.any(sel_inter):
            vals = G[sel_inter]
            muw = float(np.mean(vals))
            # weight = 2*eqw*(G-muw), scaled
            Weq = 2.0 * eqw * (G[sel_inter] - muw) * 20.0
            W[sel_inter] += Weq
            # keep weights non-catastrophic: clip very negative
            # negative means attraction; allow but limit
            # clip W overall later via force norm
        # symmetrize (W already symmetric if masks symmetric)
        # force
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        # safety clip
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        # track best feasible (overall max)
        if s % 25 == 0 or s == steps - 1:
            cur = _max_cos(x)
            if cur < best_local:
                best_local = cur
                bx = x.copy()
                ba = assign.copy()
    return bx, best_local, ba


def _lp_polish(x, seconds, t0, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds:
        if (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        it += 1
        if it > 40:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:900]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr() if False else coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            x = bx.copy()
            if radius < 5e-4:
                break
    return bx


def solve():
    t0 = time.time()
    BUDGET = 165.0
    LP_TIME = 18.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    G0 = x_best @ x_best.T
    # build partitions at several thresholds, pick best (lowest max intra)
    best_assign = None
    best_intra = 1e9
    for thr in [1e-3, 3e-3, 1e-2]:
        a = _greedy_clique_partition(G0, best, thr, np.random.RandomState(int(thr * 1e6) % 2**31))
        # refine once by reassignment
        # evaluate max intra
        same = (a[:, None] == a[None, :])
        Gm = G0.copy()
        np.fill_diagonal(Gm, -2.0)
        # max intra
        intra_vals = Gm[same]
        mx_intra = float(np.max(intra_vals)) if intra_vals.size else 1.0
        if mx_intra < best_intra:
            best_intra = mx_intra
            best_assign = a.copy()
    if best_assign is None:
        best_assign = np.repeat(np.arange(6), 9)
        rng.shuffle(best_assign)
    assign0 = best_assign.copy()

    smooth_deadline = t0 + BUDGET - LP_TIME - 2.0
    # a few anneal runs with different margins
    configs = [
        dict(steps=1400, k0=120, k1=1600, lr0=0.020, lr1=0.0025, d0=0.008, d1=0.0, eq0=0.6, eq1=0.0, win=0.025, seed=11),
        dict(steps=1400, k0=150, k1=2200, lr0=0.022, lr1=0.0020, d0=0.004, d1=0.0, eq0=0.3, eq1=0.0, win=0.020, seed=22),
        dict(steps=1200, k0=100, k1=1200, lr0=0.018, lr1=0.0030, d0=0.010, d1=0.0, eq0=0.8, eq1=0.0, win=0.030, seed=33),
    ]
    cur_x = x_best.copy()
    cur_a = assign0.copy()
    for cfg in configs:
        if time.time() > smooth_deadline:
            break
        # small perturbation to escape basin, keep block ansatz
        x_init = cur_x + rng.randn(n, d) * 0.006
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        bx, bv, ba = _block_anneal(x_init, cur_a, reass_every=350, t0=t0, deadline=smooth_deadline, **cfg)
        if bv < best - 1e-12:
            best = bv
            x_best = bx.copy()
            cur_x = bx.copy()
            cur_a = ba.copy()
        else:
            # even if not better, continue from bx if close (diversify) occasionally
            if bv < best + 0.004 and time.time() < smooth_deadline - 20:
                cur_x = bx.copy()
                cur_a = ba.copy()
        # also try from global best with fresh assign periodically
        if time.time() < smooth_deadline - 10 and rng.rand() < 0.4:
            cur_x = x_best.copy() + rng.randn(n, d) * 0.010
            cur_x /= np.linalg.norm(cur_x, axis=1, keepdims=True)
            cur_a = _reassign_by_neighborhood(cur_x, cur_a)

    # final LP polish (at most LP_TIME)
    remain = BUDGET - (time.time() - t0) - 1.0
    ft = min(LP_TIME, max(2.0, remain))
    if ft > 0.5:
        try:
            xp = _lp_polish(x_best, seconds=ft, t0=t0, budget=BUDGET - 0.5)
            vp = _max_cos(xp)
            if vp < best:
                x_best = xp.copy()
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

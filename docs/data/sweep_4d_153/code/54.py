import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_mask(x, tol=1e-9):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    iu = np.triu_indices(n, 1)
    c = g[iu]
    sel = c > m - tol
    b = np.zeros(c.shape, dtype=bool)
    b[sel] = True
    return b, m


def contact_counts(x, tol=1e-9):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cm = g > m - tol
    cnt = cm.sum(axis=1)
    return cnt, m, cm


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        gfull = x @ x.T
        c = gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start > seconds:
            break
    return x


def anneal_movable(x, mov, iters, k0, k1, lr0, lr1, mu, noise, rng):
    # x : (n,d) normalized, mov : 1d int array
    m = len(mov)
    v = np.zeros((m, x.shape[1]))
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        # avoid overflow: G-moff <= 0
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        Wm = W[mov]  # (m,n)
        rs = Wm.sum(axis=1, keepdims=True) + 1e-12
        grad = (Wm @ x) / rs
        Xm = x[mov]
        dot = np.sum(grad * Xm, axis=1, keepdims=True)
        gt = grad - dot * Xm
        v = mu * v + gt
        if mu > 0:
            vc = v / (1.0 - mu ** (t + 1))
        else:
            vc = v
        Xn = Xm - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(Xn.shape)
            pd = np.sum(pert * Xn, axis=1, keepdims=True)
            pert -= pd * Xn
            Xn = Xn + amp * pert
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        x[mov] = Xn
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        base0 = np.load("best.npy")
    except Exception:
        try:
            base0 = np.load("record.npy")
        except Exception:
            base0 = rng.standard_normal((n, d))
    base0 = np.asarray(base0, dtype=float)
    if base0.shape != (n, d):
        base0 = rng.standard_normal((n, d))
    base0 = base0 / np.linalg.norm(base0, axis=1, keepdims=True)
    best_val = max_cos(base0)
    incumbent = base0.copy()
    inc_val = best_val
    inc_mask, _ = contact_mask(incumbent)

    # diversity pool: list of (val, X)
    pool = []

    # current start (for temperature moves)
    cur = incumbent.copy()
    cur_val = inc_val

    SEARCH_END = t0 + 130.0

    trial = 0
    NCAND = 8192
    while time.time() < SEARCH_END:
        trial += 1
        # pick base: mostly incumbent/cur, sometimes pool
        r = rng.random()
        if r < 0.55:
            base = incumbent
        elif r < 0.80:
            base = cur
        elif len(pool) > 0:
            base = pool[int(rng.integers(len(pool)))][1]
        else:
            base = incumbent
        x = base.copy()

        # analyse contacts of base
        try:
            cnt, m0, cm = contact_counts(x)
        except Exception:
            cnt = np.ones(n) * 5
            cm = None

        cs = [8, 12, 16][int(rng.integers(3))]
        # rattlers and low-contact points
        order = np.argsort(cnt, kind="stable")
        ratt = np.where(cnt == 0)[0].tolist()
        low = np.where(cnt <= 4)[0].tolist()
        # neighbours of rattlers/low via contact matrix
        neigh = set()
        if cm is not None:
            for idx in (ratt + low):
                nb = np.where(cm[idx])[0].tolist()
                for u in nb:
                    neigh.add(int(u))
        # build candidate pool of indices
        pool_idx = set(ratt) | set(low) | neigh
        # always include rattlers
        cluster = set(ratt)
        # add shuffled low + neighbours
        cand_list = list((set(low) | neigh) - cluster)
        rng.shuffle(cand_list)
        for u in cand_list:
            if len(cluster) >= cs:
                break
            cluster.add(int(u))
        # if still short, add lowest-count points
        if len(cluster) < cs:
            for u in order:
                if int(u) not in cluster:
                    cluster.add(int(u))
                if len(cluster) >= cs:
                    break
        # if too big (many rattlers? not), trim randomly keeping rattlers
        cluster = list(cluster)
        if len(cluster) > cs:
            # keep rattlers, sample rest
            must = set(ratt)
            rest = [u for u in cluster if u not in must]
            rng.shuffle(rest)
            need = cs - len(must)
            cluster = list(must) + rest[:max(0, need)]
            # if still short due to few rattlers logic, just cut
            if len(cluster) > cs:
                cluster = cluster[:cs]
        mov = np.array(sorted(cluster), dtype=int)
        frozen_mask = np.ones(n, dtype=bool)
        frozen_mask[mov] = False
        core_idx = np.where(frozen_mask)[0]

        # greedy reinsertion: random order of mov
        seq = mov.copy()
        rng.shuffle(seq)
        # pre-generate candidates
        cand = rng.standard_normal((NCAND, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # we will place one by one, evaluating against frozen core + already placed
        placed = []
        # core points matrix
        Xcore = x[core_idx]
        # for speed: compute cand @ Xcore.T once (NCAND x ncore)
        try:
            core_dots = cand @ Xcore.T  # (NCAND, ncore)
            core_max = core_dots.max(axis=1)  # (NCAND,)
        except Exception:
            core_max = np.zeros(NCAND)
        # keep track of dots to already-placed for each cand: running max
        run_max = core_max.copy()
        # we need to update run_max as we place points? But placed points not known before.
        # Instead iterative: for each s in seq, pick best cand given current placed set,
        # then update run_max with chosen point for next iterations.
        # To include diversity, occasionally pick among top-5.
        chosen_pts = {}
        # reset run_max
        run_max = core_max.copy()
        for s in seq:
            # run_max currently = max(core_max, dots to previously placed)
            # find minimal run_max
            if rng.random() < 0.85:
                pick = int(np.argmin(run_max))
            else:
                # pick among 5 best randomly
                part = np.argpartition(run_max, 5)[:5]
                pick = int(part[int(rng.integers(len(part)))])
            pt = cand[pick]
            chosen_pts[int(s)] = pt
            # update run_max with this pt
            dd = cand @ pt  # (NCAND,)
            run_max = np.maximum(run_max, dd)
        for s in seq:
            x[int(s)] = chosen_pts[int(s)]

        # anneal only movable ~1000 steps
        k0 = float(rng.choice(np.array([12.0, 18.0, 25.0])))
        k1 = float(rng.choice(np.array([200.0, 300.0, 450.0])))
        lr0 = float(rng.choice(np.array([0.04, 0.06, 0.08])))
        lr1 = float(rng.choice(np.array([0.0015, 0.003, 0.005])))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
        noise = float(rng.choice(np.array([0.0, 0.0, 0.004, 0.008])))
        iters = 1000
        # adjust iters if time low
        if time.time() > SEARCH_END - 15:
            iters = 400
        try:
            x = anneal_movable(x, mov, iters, k0, k1, lr0, lr1, mu, noise, rng)
        except Exception:
            continue
        v_pre = max_cos(x)
        # quick reject: if far worse than incumbent + 5e-4 skip LP to save time
        if v_pre > inc_val + 5e-4 and rng.random() < 0.7:
            # still maybe accept as cur with temp? skip
            continue
        # short LP polish unfrozen
        budget = 0.9
        if time.time() > SEARCH_END - 10:
            budget = 0.4
        try:
            xp = lp_polish(x, budget)
        except Exception:
            xp = x
        v = max_cos(xp)
        # contact edge change vs incumbent
        try:
            bmask, _ = contact_mask(xp)
            inter = float(np.logical_and(bmask, inc_mask).sum())
            union = float(np.logical_or(bmask, inc_mask).sum()) + 1e-12
            jdist = 1.0 - inter / union
        except Exception:
            jdist = 0.0

        improved = v < inc_val - 1e-12
        diverse = (v <= inc_val + 2e-4) and (jdist >= 0.20)
        if improved:
            inc_val = v
            incumbent = xp.copy()
            inc_mask = bmask.copy() if 'bmask' in locals() else inc_mask
            cur = xp.copy()
            cur_val = v
            pool.append((v, xp.copy()))
            # keep pool small, best first
            pool = sorted(pool, key=lambda z: z[0])[:6]
        elif diverse:
            # keep diverse candidate
            pool.append((v, xp.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:8]
            # temperature move: sometimes step cur to diverse point
            # acceptance temperature cycles
            temp = float(rng.choice(np.array([5e-05, 1e-04, 2e-04])))
            if v <= cur_val + temp or rng.random() < 0.15:
                cur = xp.copy()
                cur_val = v
        else:
            # simulated-annealing style acceptance for exploration
            # allow small worsening to become cur
            temp2 = 5e-05 * (1.0 + 0.5 * np.sin(trial))
            if v < cur_val or (v < cur_val + temp2 and rng.random() < 0.1):
                cur = xp.copy()
                cur_val = v
            # occasionally reset cur to incumbent
            if rng.random() < 0.05:
                cur = incumbent.copy()
                cur_val = inc_val
        if time.time() >= SEARCH_END:
            break

    # final long LP polish of best found
    time_left = 175.0 - (time.time() - t0)
    if time_left < 5:
        time_left = 5
    time_left = min(time_left, 45.0)
    # polish incumbent and best base, pick best
    bestX = incumbent.copy()
    bestV = inc_val
    # also consider base0 if better (shouldn't)
    if best_val < bestV:
        bestX = base0.copy()
        bestV = best_val
    # if pool has good points, polish top 2
    cands = [(bestV, bestX)]
    for (vv, xx) in pool[:2]:
        # avoid duplicates of incumbent
        cands.append((vv, xx))
    # allocate time
    per = time_left / len(cands)
    final_best = bestX
    final_val = bestV
    for (vv, xx) in cands:
        b = min(per, 25.0)
        try:
            xp = lp_polish(xx, b)
        except Exception:
            xp = xx
        v = max_cos(xp)
        if v < final_val:
            final_val = v
            final_best = xp
        if time.time() - t0 > 176:
            break
    final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
    # safety: return better of final and original record
    if max_cos(base0) < max_cos(final_best):
        return base0
    return final_best

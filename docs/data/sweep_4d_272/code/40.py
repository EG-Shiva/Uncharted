import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _maxcos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, slack):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    adj = g > (m - slack)
    deg = adj.sum(axis=1).astype(int)
    return deg, m, g, adj

def _bfs_patch(adj, start, k, rng):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    order = [int(start)]
    frontier = [int(start)]
    nbrs = [np.where(adj[i])[0] for i in range(n)]
    # shuffle neighbor order deterministically via rng for diversity
    while len(order) < k and len(frontier) > 0:
        nxt = []
        # randomize frontier order slightly
        rng.shuffle(frontier)
        for u in frontier:
            nb = nbrs[u].copy()
            rng.shuffle(nb)
            for v in nb:
                v = int(v)
                if not seen[v]:
                    seen[v] = True
                    order.append(v)
                    nxt.append(v)
                    if len(order) >= k:
                        break
            if len(order) >= k:
                break
        frontier = nxt
        if len(frontier) == 0 and len(order) < k:
            # disconnected: add random unseen node closest in dot (fallback random)
            unseen = np.where(~seen)[0]
            if len(unseen) == 0:
                break
            pick = int(unseen[rng.integers(len(unseen))])
            seen[pick] = True
            order.append(pick)
            frontier = [pick]
    return np.array(order[:k], dtype=int)

def _best_sample(others, rng, n_cand=20000):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    if others.shape[0] == 0:
        return cand[0]
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    return cand[bi].copy(), float(scores[bi])

def _local_opt_point(b, others, rng, iters=160):
    b = b / np.linalg.norm(b)
    if others.shape[0] > 0:
        bs = float((others @ b).max())
    else:
        bs = -2.0
    step = 0.06
    for it in range(iters):
        prop = b + step * rng.standard_normal((48, 4))
        prop = prop / np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1) if others.shape[0] > 0 else np.zeros(48)
        j = int(np.argmin(sc))
        if sc[j] < bs - 1e-12:
            b = prop[j].copy()
            bs = float(sc[j])
            step *= 1.04
        else:
            step *= 0.96
        if step < 2e-5:
            # refresh step a bit
            step = max(step, 1e-4)
    return b, bs

def _soft_relax(y, betas, steps_list, lr, t0, deadline):
    n = y.shape[0]
    for beta, steps in zip(betas, steps_list):
        if time.time() > deadline:
            break
        for s in range(steps):
            if time.time() > deadline:
                break
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # avoid overflow: exp(beta*(G-m)), G-m<=0
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            W[W < 1e-12] = 0.0
            grad = W @ y
            proj = (grad * y).sum(axis=1, keepdims=True)
            gt = grad - proj * y
            nrm = float(np.mean(np.linalg.norm(gt, axis=1)))
            if not np.isfinite(nrm) or nrm < 1e-14:
                break
            # step
            y = y - (lr / max(nrm, 1e-12)) * gt
            y = _norm(y)
    return y

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _maxcos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SURG_END = t0 + 118.0
    rng = np.random.default_rng(20261031)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = _norm(np.asarray(xb, dtype=np.float64))
    if xb.shape != (272, 4):
        xb = _norm(rng.standard_normal((272, 4)))
    base_val = _maxcos(xb)
    best_x = xb.copy()
    best_val = base_val

    # contact analysis at slack 5e-7 (within 1e-7..1e-6)
    slack = 5e-7
    deg, mval, G, adj = _contacts(best_x, slack)
    # rattlers at 1e-7
    deg7, _, _, _ = _contacts(best_x, 1e-7)
    rattlers = np.where(deg7 == 0)[0]
    if len(rattlers) == 0:
        # fallback: 3 lowest degree
        rattlers = np.argsort(deg7)[:3]
    order_desc = np.argsort(-deg7)
    hubs = order_desc[:12].copy()

    patch_sizes = [28, 36, 24, 32, 40, 30, 26, 34]
    candidates = []

    for trial in range(len(patch_sizes)):
        if time.time() > SURG_END or time.time() > HARD_END - 55:
            break
        k = int(patch_sizes[trial])
        hub = int(hubs[trial % len(hubs)])
        # recompute degrees on current best every 2 trials to diversify
        if trial % 3 == 2:
            degt, _, _, adjt = _contacts(best_x, slack)
            ordert = np.argsort(-degt)
            hub = int(ordert[(trial * 5) % 20])
            adj_use = adjt
            ratt_use = np.where(_contacts(best_x, 1e-7)[0] == 0)[0]
            if len(ratt_use) == 0:
                ratt_use = np.argsort(_contacts(best_x, 1e-7)[0])[:3]
        else:
            adj_use = adj
            ratt_use = rattlers
        patch = _bfs_patch(adj_use, hub, k, rng)
        # delete whole patch + rattlers (union)
        doomed = set(patch.tolist()) | set(ratt_use.tolist())
        # if patch accidentally includes almost all? no
        doomed = np.array(sorted(doomed), dtype=int)
        keep = np.array(sorted(set(range(272)) - set(doomed.tolist())), dtype=int)
        core = best_x[keep].copy()  # frozen
        n_ins = len(doomed)
        # sequential refill
        inserted = []
        cur_all = core.copy()
        for ii in range(n_ins):
            if time.time() > SURG_END:
                break
            b0, _ = _best_sample(cur_all, rng, n_cand=20000)
            b1, _ = _local_opt_point(b0, cur_all, rng, iters=120)
            inserted.append(b1.copy())
            cur_all = np.vstack([cur_all, b1.reshape(1, 4)])
        if len(inserted) != n_ins:
            continue
        y = np.zeros((272, 4))
        y[keep] = core
        y[doomed] = np.vstack(inserted)
        y = _norm(y)
        v_pre = _maxcos(y)
        # short high-temperature log-sum-exp phase (unfrozen)
        # betas correspond to tau ~ 0.03, 0.012
        deadline_soft = min(SURG_END, time.time() + 4.0)
        y2 = _soft_relax(y, betas=[30.0, 80.0], steps_list=[120, 120], lr=0.03, t0=t0, deadline=deadline_soft)
        v_post = _maxcos(y2)
        candidates.append((v_post, y2.copy(), v_pre, tuple(sorted(doomed.tolist()))))
        # acceptance based partly on graph change: keep diverse even if worse
        # update best if improved
        if v_post < best_val:
            best_val = float(v_post)
            best_x = y2.copy()
            # refresh contact info
            deg, mval, G, adj = _contacts(best_x, slack)
            deg7, _, _, _ = _contacts(best_x, 1e-7)
            rattlers = np.where(deg7 == 0)[0]
            if len(rattlers) == 0:
                rattlers = np.argsort(deg7)[:3]
            order_desc = np.argsort(-deg7)
            hubs = order_desc[:12].copy()
        elif v_post < base_val + 0.004:
            # accept as alternate basin seed with some probability: keep for polish anyway
            pass

    # sort candidates by post-anneal value
    candidates.sort(key=lambda t: t[0])
    # also include original best
    pool = [(best_val, best_x.copy())] + [(c[0], c[1]) for c in candidates[:6]]

    # deduplicate pool by value proximity? keep top 4
    pool = sorted(pool, key=lambda kv: kv[0])[:4]

    polished_best_x = best_x.copy()
    polished_best_val = best_val
    # if no candidates, pool has best only
    for v, xc in pool:
        if time.time() > HARD_END - 6:
            break
        # give each a time slice
        remaining = HARD_END - time.time()
        # allocate fairly
        alloc = max(8.0, remaining / (len(pool) + 1))
        deadline = min(HARD_END, time.time() + alloc)
        # quick extra soft at lower temp before LP if time allows and candidate is not already polished
        # one short sharpening
        if time.time() + 2 < deadline and v > polished_best_val - 0.001:
            xc = _soft_relax(xc.copy(), betas=[150.0], steps_list=[60], lr=0.02, t0=t0, deadline=min(deadline - 1, time.time() + 2.0))
        xp, vp = _lp_polish(xc, deadline)
        if vp < polished_best_val:
            polished_best_val = float(vp)
            polished_best_x = xp.copy()

    out = _norm(polished_best_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    out = out / nrm
    return out

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _lp_polish(x, seconds, t0, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - t0 > deadline:
            break
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -2500)[-2500:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[part]] = True
            sel = sel2
            cnt = 2500
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        crow = cols.reshape(p, 2 * d + 1)
        vrow = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        crow[:, :d] = I[:, None] * d + ar
        crow[:, d:2 * d] = J[:, None] * d + ar
        crow[:, 2 * d] = n * d
        vrow[:, :d] = x[J]
        vrow[:, d:2 * d] = x[I]
        vrow[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best


def _relax(x, movable, iters, tau, step0):
    # movable: bool (n,)
    x = x.copy()
    step = step0
    for _ in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - 8.0 * tau
        # weights
        # avoid huge exp: only where > thr
        W = np.zeros_like(G)
        # use mask
        mask = G > thr
        # exp in place
        W[mask] = np.exp((G[mask] - m) / tau)
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        Gt[~movable] = 0.0
        # clip large steps
        x = x - step * Gt
        x = _norm(x)
        step *= 0.995
    return x


def _bfs_cluster(adj, start, k, rng):
    # adj: list of lists
    seen = set([int(start)])
    queue = [int(start)]
    order = [int(start)]
    # BFS expanding in (shuffled) adjacency order for variety
    qi = 0
    while len(seen) < k and qi < len(queue):
        u = queue[qi]
        qi += 1
        nbrs = adj[u]
        if len(nbrs) > 1:
            perm = rng.permutation(len(nbrs))
            nbrs = [nbrs[i] for i in perm]
        for w in nbrs:
            if w not in seen:
                seen.add(w)
                queue.append(w)
                order.append(w)
                if len(seen) >= k:
                    break
    # if disconnected component too small, fill randomly
    n = len(adj)
    while len(seen) < k:
        c = int(rng.integers(0, n))
        if c not in seen:
            seen.add(c)
            order.append(c)
    return np.array(order[:k], dtype=np.int64)


def solve():
    t0 = time.time()
    DEADLINE = 165.0  # hard internal deadline
    rng = np.random.default_rng(12345)
    n_d, dim = 323, 7

    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n_d, dim):
                a = _norm(a)
                if not np.all(np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < b0:
                    b0 = v
                    x0 = a.copy()
        except Exception:
            continue
    if x0 is None:
        x0 = rng.standard_normal(size=(n_d, dim))
        x0 = _norm(x0)
        b0 = _max_cos(x0)

    best_x = x0.copy()
    best_val = b0

    # build contact graph helper
    def build_adj(x, thr_offset=1e-7):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - thr_offset
        I, J = np.where(np.triu(G > thr, 1))
        adj = [[] for _ in range(n_d)]
        for a, b in zip(I.tolist(), J.tolist()):
            adj[a].append(b)
            adj[b].append(a)
        degs = np.array([len(a) for a in adj], dtype=np.int64)
        return adj, degs, m

    adj, degs, mval = build_adj(best_x, 1e-7)
    hub_order = np.argsort(-degs)

    # trial loop over hubs + random clusters
    # predefine cluster sizes 12..18 cycling
    n_trials = 40
    # keep top hubs
    top_hubs = hub_order[:12].tolist()

    for trial in range(n_trials):
        if time.time() - t0 > DEADLINE - 25.0:
            break
        k = 12 + (trial % 7)  # 12..18
        # select start hub
        if trial < len(top_hubs) * 2:
            hub = top_hubs[trial % len(top_hubs)]
            C = _bfs_cluster(adj, hub, k, rng)
        else:
            s = int(rng.integers(0, n_d))
            C = _bfs_cluster(adj, s, k, rng)
        C = np.unique(C)
        # ensure size k (unique may shrink, refill)
        while len(C) < k:
            extra = int(rng.integers(0, n_d))
            if extra not in set(C.tolist()):
                C = np.append(C, extra)
        mask_c = np.zeros(n_d, dtype=bool)
        mask_c[C] = True
        R = np.where(~mask_c)[0]
        core = best_x[R]  # (n-k, d)

        # 50k candidates
        try:
            cand = rng.standard_normal(size=(50000, dim))
            cand = _norm(cand)
        except Exception:
            continue
        inserted = []
        # sequential max-min
        # maintain existing matrix incrementally
        existing = core.copy()
        for s_idx in range(len(C)):
            if time.time() - t0 > DEADLINE - 20.0:
                break
            # scores: max cosine to existing
            # chunk to limit memory: 50000x~310 -> 15.5M floats ~124MB; chunk in 2
            best_idx = -1
            best_score = np.inf
            # process in chunks of 10000
            for cs in range(0, 50000, 10000):
                ce = min(cs + 10000, 50000)
                S = cand[cs:ce] @ existing.T  # (chunk, m)
                mx = S.max(axis=1)
                li = int(np.argmin(mx))
                if mx[li] < best_score:
                    best_score = float(mx[li])
                    best_idx = cs + li
            chosen = cand[best_idx].copy()
            inserted.append(chosen)
            existing = np.vstack([existing, chosen[None, :]])
        if len(inserted) != len(C):
            continue
        x_trial = best_x.copy()
        x_trial[C] = np.array(inserted)
        x_trial = _norm(x_trial)

        # frozen-core optimisation then all-point short relaxation
        movable_frozen = np.zeros(n_d, dtype=bool)
        movable_frozen[C] = True
        try:
            x_trial = _relax(x_trial, movable_frozen, iters=120, tau=0.01, step0=0.05)
            if time.time() - t0 > DEADLINE - 20.0:
                # still evaluate
                pass
            else:
                movable_all = np.ones(n_d, dtype=bool)
                x_trial = _relax(x_trial, movable_all, iters=120, tau=0.007, step0=0.02)
        except Exception:
            continue

        try:
            v_pre = _max_cos(x_trial)
        except Exception:
            continue
        if not np.isfinite(v_pre):
            continue
        # keep if better even without LP
        if v_pre < best_val:
            # promising: LP polish short
            if v_pre < 0.670:
                secs = 6.0
                # don't exceed deadline
                remain = DEADLINE - (time.time() - t0) - 12.0
                if remain < 2.0:
                    # no time for LP, just take it
                    best_val = v_pre
                    best_x = x_trial.copy()
                    adj, degs, mval = build_adj(best_x, 1e-7)
                    hub_order = np.argsort(-degs)
                    top_hubs = hub_order[:12].tolist()
                    continue
                secs = min(secs, max(2.0, remain))
                try:
                    x_lp, v_lp = _lp_polish(x_trial, seconds=secs, t0=t0, deadline=DEADLINE - 8.0)
                except Exception:
                    x_lp, v_lp = x_trial, v_pre
                if v_lp < best_val and np.all(np.isfinite(x_lp)):
                    best_val = float(v_lp)
                    best_x = x_lp.copy()
                    adj, degs, mval = build_adj(best_x, 1e-7)
                    hub_order = np.argsort(-degs)
                    top_hubs = hub_order[:12].tolist()
                elif v_pre < best_val:
                    # LP didn't improve but pre was better than best? update to LP best or pre?
                    # if LP worse, keep pre if it beats best
                    if v_pre < best_val:
                        best_val = v_pre
                        best_x = x_trial.copy()
                        adj, degs, mval = build_adj(best_x, 1e-7)
                        hub_order = np.argsort(-degs)
                        top_hubs = hub_order[:12].tolist()
            else:
                # not promising for LP but if beats best (rare since best~0.665), take
                if v_pre < best_val:
                    best_val = v_pre
                    best_x = x_trial.copy()
                    adj, degs, mval = build_adj(best_x, 1e-7)
                    hub_order = np.argsort(-degs)
                    top_hubs = hub_order[:12].tolist()
        else:
            # not better than best, but if below 0.670 maybe LP could jump below best?
            # only try occasionally to save time: try if within 0.002 of best
            if v_pre < 0.670 and v_pre < best_val + 0.002 and (trial % 3 == 0):
                remain = DEADLINE - (time.time() - t0) - 12.0
                if remain > 3.0:
                    try:
                        x_lp, v_lp = _lp_polish(x_trial, seconds=min(4.0, remain), t0=t0, deadline=DEADLINE - 8.0)
                    except Exception:
                        continue
                    if v_lp < best_val and np.all(np.isfinite(x_lp)):
                        best_val = float(v_lp)
                        best_x = x_lp.copy()
                        adj, degs, mval = build_adj(best_x, 1e-7)
                        hub_order = np.argsort(-degs)
                        top_hubs = hub_order[:12].tolist()

    # final short polish of best with remaining time
    remain = DEADLINE - (time.time() - t0)
    if remain > 3.0:
        try:
            x_lp, v_lp = _lp_polish(best_x, seconds=min(remain - 1.0, 20.0), t0=t0, deadline=DEADLINE)
            if v_lp < best_val and np.all(np.isfinite(x_lp)):
                best_val = float(v_lp)
                best_x = x_lp.copy()
        except Exception:
            pass

    best_x = _norm(best_x)
    # safety: if somehow worse than initial, return initial best valid
    try:
        vf = _max_cos(best_x)
        if not np.isfinite(vf) or vf > b0 + 1e-12:
            # return the better of the two
            if vf > b0:
                best_x = x0.copy()
    except Exception:
        best_x = x0.copy()
    best_x = _norm(best_x)
    return best_x

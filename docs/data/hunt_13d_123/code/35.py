import time
import numpy as np
from scipy.optimize import nnls, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=1400):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1.2e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and radius > 1e-14 and itc < 30:
        itc += 1
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(25 * radius, 2e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)), (J[:, None] * d + np.arange(d)), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            remain = seconds - (time.time() - start)
            if remain <= 0.05:
                break
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.2, remain)})
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
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return best_x

def _stress_weights(X, mval, tol=1e-6, cap=3000, lam=1.0):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    thr = mval - tol
    sel = np.where(c > thr)[0]
    if len(sel) > cap:
        part = np.argpartition(-c[sel], cap - 1)[:cap]
        sel = sel[part]
    # neighbor lists
    neigh = [[] for _ in range(n)]
    cval = {}
    for k in sel:
        i = int(iu0[k]); j = int(iu1[k])
        cc = float(c[k])
        neigh[i].append((j, cc))
        neigh[j].append((i, cc))
    totals = np.zeros(n)
    resids = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        lst = neigh[i]
        k = len(lst)
        counts[i] = k
        if k == 0:
            totals[i] = 0.0; resids[i] = 0.0
            continue
        T = np.empty((d, k))
        xi = X[i]
        for t, (j, cc) in enumerate(lst):
            T[:, t] = X[j] - cc * xi
        A = np.vstack([T, lam * np.ones((1, k))])
        b = np.zeros(d + 1); b[d] = lam
        try:
            w, _ = nnls(A, b)
        except Exception:
            w = np.ones(k) / k
        totals[i] = float(w.sum())
        r = T @ w
        resids[i] = float(np.linalg.norm(r))
    return totals, resids, counts, G

def _repack_patch(Xb, free_idx, rng, budget):
    n, d = Xb.shape
    t_end = time.time() + budget
    trial = Xb.copy()
    frozen_mask = np.ones(n, dtype=bool)
    frozen_mask[np.asarray(free_idx, dtype=int)] = False
    # greedy hole refill sequentially for free points
    for pos, idx in enumerate(list(free_idx)):
        if time.time() >= t_end - 0.5:
            break
        # sample candidates
        Y = rng.normal(size=(2500, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # max dot to all other current points (exclude idx)
        # compute efficiently
        S = Y @ trial.T  # 2500 x 123
        S[:, idx] = -10.0
        md = S.max(axis=1)
        bi = int(np.argmin(md))
        v = Y[bi] + 0.002 * rng.normal(size=(d,))
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            v = Y[bi]
            nv = np.linalg.norm(v)
        trial[idx] = v / nv
    # annealed penalized softmax descent on free subset only
    free = ~frozen_mask
    free_idx_arr = np.asarray(free_idx, dtype=int)
    # schedule
    steps = 1200
    T0, T1 = 0.015, 0.0022
    lr0, lr1 = 0.45, 0.04
    # penalty weight for max (softmax already); add small L2 to keep moving? not needed
    for s in range(steps):
        if time.time() >= t_end:
            break
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = trial @ trial.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ trial) / row
        dot = (Avg * trial).sum(axis=1, keepdims=True)
        Gt = Avg - dot * trial
        trial[free] = trial[free] - lr * Gt[free]
        nn = np.linalg.norm(trial[free], axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        trial[free] /= nn
    # renormalize all
    trial /= np.linalg.norm(trial, axis=1, keepdims=True)
    return trial

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    LIM = 175.0
    rng = _np.random.default_rng(12345)
    n, d = 123, 13
    X0 = None; bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn); a = _np.asarray(a, float)
            if a.shape == (n, d):
                a /= _np.linalg.norm(a, axis=1, keepdims=True)
                if _np.any(~_np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < bv:
                    bv = v; X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d)); X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        bv = _max_cos(X0)
    best = X0.copy(); best_val = float(bv)
    # stress analysis on best
    try:
        totals, resids, counts, Gfull = _stress_weights(best, best_val, tol=1e-6)
    except Exception:
        totals = _np.zeros(n); resids = _np.zeros(n); counts = _np.zeros(n, dtype=int)
        Gfull = best @ best.T
        _np.fill_diagonal(Gfull, -2.0)
    # identify rattler: min total (or min counts)
    order_stress = _np.argsort(totals)  # ascending: lowest stress first
    order_resid = _np.argsort(-resids)  # descending residual
    order_over = _np.argsort(-totals)  # overloaded first
    rattler = int(order_stress[0])
    # nearest neighbours of rattler by angle
    gr = Gfull[rattler].copy()
    nn_r = _np.argsort(-gr)  # closest first (excluding self? self is -2 so last)
    # build up to 3 patches
    patches = []
    hub_choices = []
    # hub0: lowest non-zero stress (second after rattler if rattler zero)
    # hub1: largest residual
    # hub2: largest overload (max totals)
    cands = []
    # pick distinct hubs
    h0 = int(order_stress[1]) if n > 1 else int(order_stress[0])
    h1 = int(order_resid[0])
    h2 = int(order_over[0])
    for h in (h0, h1, h2):
        if h not in hub_choices:
            hub_choices.append(h)
        if len(hub_choices) >= 3:
            break
    # ensure 3 entries even with duplicates by adding next choices
    k = 1
    while len(hub_choices) < 3:
        for arr in (order_stress, order_resid, order_over):
            for cand in arr:
                cand = int(cand)
                if cand != rattler and cand not in hub_choices:
                    hub_choices.append(cand)
                    break
            if len(hub_choices) >= 3:
                break
        k += 1
        if k > 5:
            break
    for hi, hub in enumerate(hub_choices[:3]):
        s = set()
        s.add(rattler)
        # 5 nearest neighbours of rattler
        c = 0
        for q in nn_r:
            q = int(q)
            if q == rattler:
                continue
            s.add(q); c += 1
            if c >= 5:
                break
        # hub neighbourhood: hub + up to 7 nearest
        s.add(hub)
        gh = Gfull[hub].copy()
        nn_h = _np.argsort(-gh)
        c = 0
        for q in nn_h:
            q = int(q)
            if q == hub or q in s:
                continue
            s.add(q); c += 1
            if c >= 7 or len(s) >= 16:
                break
        # size adjust to 10-18
        lst = list(s)
        if len(lst) > 18:
            # keep rattler, hub + closest
            # trim farthest from hub/rattler
            lst = lst[:18]
        if len(lst) < 10:
            # pad with lowest-stress points
            for q in order_stress:
                q = int(q)
                if q not in s:
                    s.add(q)
                if len(s) >= 12:
                    break
            lst = list(s)
        patches.append(lst)
    # iterate patches: repack (<=15s) + LP polish (<=5s), accept only if improves
    for pi, patch in enumerate(patches):
        remain = LIM - (_time.time() - t0)
        if remain < 8.0:
            break
        repack_budget = min(15.0, remain - 6.0)
        if repack_budget < 1.0:
            break
        # cap patch size 10-18
        if len(patch) < 8 or len(patch) > 20:
            # fix
            patch = patch[:16]
        trial = _repack_patch(best, patch, rng, repack_budget)
        # time check after repack stage
        if _time.time() - t0 >= LIM - 6.0:
            v = _max_cos(trial)
            if v < best_val - 1e-14:
                best = trial; best_val = v
            break
        remain2 = LIM - (_time.time() - t0)
        lp_sec = min(5.0, max(0.6, remain2 - 1.0))
        # strictly at most 5s
        lp_sec = min(lp_sec, 5.0)
        try:
            pol = _lp_polish(trial, lp_sec)
        except Exception:
            pol = trial
        # time check after LP stage
        v = _max_cos(pol)
        if v < best_val - 1e-14 and _np.all(_np.isfinite(pol)):
            best = pol.copy(); best_val = v
            # refresh stress for remaining patches? keep original G for simplicity but update Gfull
            try:
                totals, resids, counts, Gfull = _stress_weights(best, best_val, tol=1e-6)
            except Exception:
                pass
        # else discard
        if _time.time() - t0 >= LIM - 2.0:
            break
    best /= _np.linalg.norm(best, axis=1, keepdims=True)
    return best

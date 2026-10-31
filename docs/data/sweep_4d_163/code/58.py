import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _degrees(x, mc, eps=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mask = (mc - G) < eps
    # count per point (symmetric, count upper+lower)
    # each pair contributes to two points; use row-wise
    deg = np.count_nonzero(mask, axis=1)
    # mask includes both triangles; diagonal is False since mc-(-2)>eps
    return deg, G

def _refine_voids(V, C, iters=60, T0=0.02, T1=0.002, lr0=0.12, lr1=0.02):
    # V (Nv,4), C (K,4)
    V = _norm(V)
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        D = V @ C.T
        m = D.max(axis=1, keepdims=True)
        W = np.exp((D - m) / T)
        rs = W.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-12)
        grad = (W @ C) / rs
        dot = np.sum(grad * V, axis=1, keepdims=True)
        gt = grad - dot * V
        sn = np.linalg.norm(gt, axis=1, keepdims=True)
        # normalize step length to <=1
        gt = gt / np.maximum(sn, 1.0)
        V = V - lr * gt
        V = _norm(V)
    return V

def _smooth_mobile(X, mobile, iters, T0, T1, lr0, lr1):
    # X (n,4) normalized, mobile boolean mask
    midx = np.where(mobile)[0]
    if midx.size == 0:
        return X
    n = X.shape[0]
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        X[midx] = X[midx] - lr * step[midx]
        X[midx] = _norm(X[midx])
    return X

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=2000):
    global _IU
    x = _norm(x)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 500:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.04)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = 150.0
    DEAD = 170.0
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (163, 4):
                a = _norm(a)
                # filter non-finite
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            continue
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        a = rng0.normal(size=(163, 4))
        a = _norm(a)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best = float(vals[bi])

    n = 163
    # contact degrees of incumbent
    mc = best
    deg, _ = _degrees(x_best, mc, eps=1e-9)
    order = np.argsort(deg, kind="stable")
    # weak pool: degree <=4
    weak_pool = [int(i) for i in order if deg[i] <= 4]
    if len(weak_pool) == 0:
        weak_pool = [int(i) for i in order[:8]]
    # ensure rattler (min degree) first
    # build deletion sets of size 1,2,4,6
    del_sets = []
    # size 1: rattler
    del_sets.append([int(order[0])])
    # size 2: two weakest
    if len(order) >= 2:
        del_sets.append([int(order[0]), int(order[1])])
    # size 4: four weakest
    if len(order) >= 4:
        del_sets.append([int(order[i]) for i in range(4)])
    # size 6: six weakest
    if len(order) >= 6:
        del_sets.append([int(order[i]) for i in range(6)])
    # two extra random draws from weak pool for diversity (deterministic)
    srng = np.random.default_rng(20261031)
    if len(weak_pool) >= 4:
        # random 4 distinct from weak pool
        perm = srng.permutation(len(weak_pool))
        s = sorted([weak_pool[int(perm[i])] for i in range(min(4, len(weak_pool)))])
        if s not in del_sets:
            del_sets.append(s)
    if len(weak_pool) >= 6:
        perm2 = srng.permutation(len(weak_pool))
        s = sorted([weak_pool[int(perm2[i])] for i in range(min(6, len(weak_pool)))])
        if s not in del_sets:
            del_sets.append(s)
    # limit to at most 6 sets
    del_sets = del_sets[:6]

    for si, dset in enumerate(del_sets):
        if time.time() - t0 > HARD:
            break
        D = len(dset)
        dset_sorted = sorted(dset)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[dset_sorted] = False
        core = x_best[keep_mask]  # (K,4)
        K = core.shape[0]
        # generate 50000 random unit quaternions
        grng = np.random.default_rng(5000 + si * 977)
        cand = grng.normal(size=(50000, 4))
        cand = _norm(cand)
        # score by max cosine to core
        # chunk to save memory
        best_scores = np.empty(50000, dtype=float)
        CH = 10000
        for a in range(0, 50000, CH):
            b = min(a + CH, 50000)
            dots = cand[a:b] @ core.T
            best_scores[a:b] = dots.max(axis=1)
        # take best 200 void candidates (lowest max cosine)
        idx200 = np.argpartition(best_scores, 200)[:200]
        V = cand[idx200].copy()
        # locally refine with log-sum-exp L-BFGS-like smooth steps (batched)
        # use two-stage anneal for speed
        V = _refine_voids(V, core, iters=50, T0=0.02, T1=0.002, lr0=0.10, lr1=0.015)
        # rescore after refine
        core_scores = (V @ core.T).max(axis=1)
        # greedy reinsertion with mutual compatibility
        Gc = V @ V.T
        selected = []
        # first: minimal core score
        first = int(np.argmin(core_scores))
        selected.append(first)
        for _ in range(1, D):
            best_j = -1
            best_v = 1e9
            sel_arr = np.array(selected, dtype=int)
            for j in range(V.shape[0]):
                if j in selected:
                    continue
                mut = float(np.max(Gc[j, sel_arr])) if len(sel_arr) > 0 else -2.0
                v = float(max(core_scores[j], mut))
                if v < best_v:
                    best_v = v
                    best_j = j
            if best_j < 0:
                break
            selected.append(int(best_j))
        new_pts = V[np.array(selected, dtype=int)]
        # build full config
        Xnew = x_best.copy()
        for k, gi in enumerate(dset_sorted):
            Xnew[gi] = new_pts[k]
        Xnew = _norm(Xnew)
        v_new = _max_cos(Xnew)
        # mobile set: reinserted + core within slack 1e-3 of them
        # compute max dot to new points for each core row
        # need mapping from full index to core; simpler: compute dots of Xnew to new pts
        # use current global max
        mcur = v_new
        # dots from all points to new points
        # new indices = dset_sorted
        # compute G full once
        Gf = Xnew @ Xnew.T
        np.fill_diagonal(Gf, -2.0)
        # for each i, max to any reinserted
        # columns of dset
        max_to_new = Gf[:, dset_sorted].max(axis=1) if D > 0 else np.full(n, -2.0)
        mobile = np.zeros(n, dtype=bool)
        mobile[dset_sorted] = True
        # core within 1e-3 slack: max_to_new > mcur - 1e-3
        near = (max_to_new > (mcur - 1e-3))
        mobile = mobile | near
        # freeze core otherwise; reoptimize mobile with annealed smooth-max descent
        # annealed schedule
        Xnew = _smooth_mobile(Xnew, mobile, iters=350, T0=0.010, T1=0.0009, lr0=0.05, lr1=0.007)
        v_mob = _max_cos(Xnew)
        # only do short LP polish if promising (within 2e-3 of best or better) and time allows
        # spec: at most one short LP polish per surgery
        improved_smooth = (v_mob < best - 1e-9)
        # if smooth alone already improves, do a quick LP to lock it; else try LP only if close
        do_lp = False
        if improved_smooth:
            do_lp = True
        elif (v_mob < best + 2e-3) and (time.time() - t0 < HARD - 12):
            # only try LP for first two such cases to save time
            do_lp = (si < 3)
        if do_lp:
            remain_hard = HARD - (time.time() - t0)
            lp_sec = float(min(5.0, max(1.5, remain_hard - 4.0)))
            if lp_sec >= 1.5:
                Xlp = _lp_polish(Xnew, lp_sec, r0=1.5e-3)
                v_lp = _max_cos(Xlp)
                Xnew, v_mob = Xlp, v_lp
        if v_mob < best - 1e-9:
            x_best = _norm(Xnew).copy()
            best = float(v_mob)
            # refresh degrees/order/weak pool for subsequent surgeries
            deg, _ = _degrees(x_best, best, eps=1e-9)
            order = np.argsort(deg, kind="stable")
        # time guard
        if DEAD - (time.time() - t0) < 8:
            break

    # final safety: ensure finite, normalized
    x_best = _norm(x_best)
    # shrink by tiny margin not needed; return
    return x_best

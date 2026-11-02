import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    tight = g > m - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return m, cnt

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _subset_anneal(x, free_idx, t_target, steps, beta0, beta1, lr0, lr1, deadline=None):
    n, d = x.shape
    free_mask = np.zeros(n, dtype=bool)
    free_mask[np.asarray(free_idx)] = True
    # involvement mask for pairs
    inv = (free_mask[:, None] | free_mask[None, :])
    np.fill_diagonal(inv, False)
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb0 = np.log(beta1 / beta0) / max(steps - 1, 1)
    lrl0 = np.log(lr1 / lr0) / max(steps - 1, 1)
    for s in range(steps):
        if deadline is not None and (s % 100 == 0) and time.time() > deadline:
            break
        beta = beta0 * np.exp(lrb0 * s)
        lr = lr0 * np.exp(lrl0 * s)
        G = x @ x.T
        Arg = beta * (G - t_target)
        np.clip(Arg, -50.0, 50.0, out=Arg)
        E = np.exp(Arg, out=Arg)
        E[~inv] = 0.0
        # gradient
        grad = E @ x
        # tangent projection
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        gt[~free_mask] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        # keep frozen exactly
        # renormalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _refine_void(c0, cur, steps=220, beta=60.0, lr0=0.06, lr1=0.002):
    c = c0 / max(np.linalg.norm(c0), 1e-12)
    lrl = np.log(lr1 / lr0) / max(steps - 1, 1)
    for s in range(steps):
        lr = lr0 * np.exp(lrl * s)
        dots = cur @ c
        mx = dots.max()
        e = np.exp(beta * (dots - mx))
        w = e / e.sum()
        g = w @ cur
        gd = float(g @ c)
        gt = g - gd * c
        c = c - lr * gt
        nrm = np.linalg.norm(c)
        if nrm < 1e-12:
            break
        c /= nrm
    return c

def solve():
    t0 = time.time()
    HARD = 170.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    RECORD = 0.84910235828
    patch_sizes = [8, 12, 16]
    deltas = [1e-4, 3e-4, 1e-3]
    trial = 0
    # main patch loop, reserve ~18s for final polish
    while time.time() - t0 < HARD - 22:
        if time.time() - t0 > HARD - 30 and trial > 2:
            break
        m_cur, cnt = _contacts(x_best, tol=1e-7)
        rattlers = np.where(cnt == 0)[0]
        order_hub = np.argsort(-cnt)
        # choose centre
        if trial == 0:
            centre = int(rattlers[0]) if len(rattlers) > 0 else int(np.argmin(cnt))
        elif trial % 3 == 1:
            centre = int(rng.integers(n))
        elif trial % 3 == 2:
            # highest-contact hub, random among top 8
            top = order_hub[:8]
            centre = int(top[rng.integers(len(top))])
        else:
            if len(rattlers) > 0 and (trial % 2 == 0):
                centre = int(rattlers[rng.integers(len(rattlers))])
            else:
                centre = int(rng.integers(n))
        k = patch_sizes[trial % len(patch_sizes)]
        delta = deltas[trial % len(deltas)]
        t_target = min(best_val, RECORD) - delta
        # choose reference for neighbours
        if rng.random() < 0.65:
            ref = centre
        else:
            # randomly chosen hub point
            hubs = np.where(cnt >= 6)[0]
            if len(hubs) > 0:
                ref = int(hubs[rng.integers(len(hubs))])
            else:
                ref = int(order_hub[0])
        # neighbours of ref by cosine
        dots_ref = x_best @ x_best[ref]
        dots_ref[ref] = -2.0
        # if ref != centre, ensure centre included
        neigh_order = np.argsort(-dots_ref)
        del_set = set()
        del_set.add(int(centre))
        for idx in neigh_order:
            if len(del_set) >= k:
                break
            if int(idx) == int(ref) and int(ref) not in del_set and len(del_set) < k:
                # allow ref itself as part of patch
                pass
            del_set.add(int(idx))
        # if ref not yet included and we want patch around ref, add ref
        if ref != centre and ref not in del_set and len(del_set) < k:
            del_set.add(int(ref))
        del_idx = np.array(sorted(del_set), dtype=int)
        # safety: ensure size k (may be k or k due to logic)
        if len(del_idx) > k:
            # keep centre + closest k-1
            keep_rest = [i for i in del_idx if i != centre][:k-1]
            del_idx = np.array([centre] + keep_rest, dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        base = x_best[keep_mask]
        cur_list = [base[i] for i in range(base.shape[0])]
        cur_mat = base.copy()
        new_pts = []
        # random order of insertion
        ins_order = rng.permutation(len(del_idx))
        for _ii in range(len(del_idx)):
            # 20000 random candidates
            cand = rng.normal(size=(20000, 4))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # max dot to current set
            # chunk to save memory
            best_cand = None
            best_score = 1e9
            # process in 2 chunks of 10000
            for a in (0, 10000):
                cc = cand[a:a+10000]
                D = cc @ cur_mat.T
                mx = D.max(axis=1)
                j = int(np.argmin(mx))
                if float(mx[j]) < best_score:
                    best_score = float(mx[j])
                    best_cand = cc[j].copy()
            # local refinement
            best_cand = _refine_void(best_cand, cur_mat, steps=220, beta=60.0, lr0=0.06, lr1=0.002)
            new_pts.append(best_cand)
            cur_mat = np.vstack([cur_mat, best_cand[None, :]])
            if time.time() - t0 > HARD - 22:
                break
        y = np.empty_like(x_best)
        y[keep_mask] = base
        # map new_pts to del_idx in shuffled order? just in order
        # use ins_order to assign (doesn't matter)
        for pos, pt in zip(del_idx, new_pts):
            y[pos] = pt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # subset anneal on moved points
        dl = HARD - (time.time() - t0)
        if dl < 8:
            break
        y = _subset_anneal(y, del_idx, t_target, steps=650, beta0=30.0, beta1=260.0, lr0=0.03, lr1=0.003)
        # short all-free anneal
        y = _subset_anneal(y, np.arange(n), t_target, steps=320, beta0=40.0, beta1=220.0, lr0=0.012, lr1=0.002)
        # LP finish
        remain = HARD - (time.time() - t0)
        lp_budget = min(4.0, max(1.2, remain - 6.0))
        if lp_budget < 0.5:
            break
        y_lp, v_lp = _lp_polish(y, lp_budget)
        if v_lp < best_val - 1e-12:
            x_best = y_lp
            best_val = v_lp
        elif v_lp < best_val + 0.002:
            # extra all-free anneal since contact graph may have changed
            y2 = _subset_anneal(y_lp, np.arange(n), t_target - 1e-4, steps=700, beta0=35.0, beta1=300.0, lr0=0.010, lr1=0.0015)
            remain2 = HARD - (time.time() - t0)
            if remain2 > 5:
                y2_lp, v2_lp = _lp_polish(y2, min(4.0, remain2 - 3.0))
                if v2_lp < best_val:
                    x_best = y2_lp
                    best_val = v2_lp
        trial += 1
        # early exit if clearly improved well below record? continue to improve
        if time.time() - t0 > HARD - 22:
            break
    # final polish of incumbent with remaining time
    try:
        remain = HARD - (time.time() - t0)
        if remain > 2.0:
            x_best, best_val = _lp_polish(x_best, min(remain - 1.0, 12.0))
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

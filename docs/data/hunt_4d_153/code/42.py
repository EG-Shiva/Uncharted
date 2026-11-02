import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _counts(x, thr=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = float(g.max())
    tight = g > m - thr
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return m, cnt, iu, g

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
        if time.time() - start >= budget:
            break
    return x, best

def _full_smooth(x0, steps, beta0, beta1, lr0, lr1):
    n, d = x0.shape
    x = x0.copy()
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    la0, la1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * f)
        lr = np.exp(la0 + (la1 - la0) * f)
        G = x @ x.T
        g = G[Iu, Ju]
        gm = g.max()
        e = np.exp(beta * (g - gm))
        w = e / e.sum()
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1 ** (s + 1))
        vhat = v1 / (1 - b2 ** (s + 1))
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _subset_target_smooth(x0, moved, t_target, steps, beta0, beta1, lr0, lr1):
    n, d = x0.shape
    x = x0.copy()
    iu = np.triu_indices(n, 1)
    selm = moved[iu[0]] | moved[iu[1]]
    Ip = iu[0][selm]; Jp = iu[1][selm]
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    la0, la1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * f)
        lr = np.exp(la0 + (la1 - la0) * f)
        g = (x[Ip] * x[Jp]).sum(axis=1)
        gm = g.max()
        e = np.exp(beta * (g - gm))
        Z = e.sum()
        w = e / Z
        excess = g - t_target
        pos = excess > 0
        if not np.any(pos):
            # all below target: gentle spreading with small weight
            weight = w * 0.15
        else:
            weight = w * pos.astype(float)
            # renormalize to keep step stable
            sw = weight.sum()
            if sw > 1e-12:
                weight = weight / sw
        grad = np.zeros_like(x)
        np.add.at(grad, Ip, weight[:, None] * x[Jp])
        np.add.at(grad, Jp, weight[:, None] * x[Ip])
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        # only update moved rows in adam moments
        m1[moved] = b1 * m1[moved] + (1 - b1) * gt[moved]
        v1[moved] = b2 * v1[moved] + (1 - b2) * (gt[moved] * gt[moved])
        mhat = m1[moved] / (1 - b1 ** (s + 1))
        vhat = v1[moved] / (1 - b2 ** (s + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        x[moved] = x[moved] - lr * step
        x[moved] /= np.maximum(np.linalg.norm(x[moved], axis=1, keepdims=True), 1e-12)
    return x

def _largest_void(x_keep, rng, n_cand=20000):
    cand = rng.normal(size=(n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_keep.T
    md = dots.max(axis=1)
    bi = int(np.argmin(md))
    v = cand[bi].copy()
    best_md = float(md[bi])
    # local refinement: batched hill climb
    sig = 0.03
    for it in range(160):
        prop = v[None, :] + rng.normal(0, sig, size=(32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        mds = (prop @ x_keep.T).max(axis=1)
        k = int(np.argmin(mds))
        if mds[k] < best_md:
            v = prop[k].copy()
            best_md = float(mds[k])
        sig *= 0.992
        if it % 40 == 39:
            sig = max(sig, 0.004)
    return v

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val
    n = 153
    patch_ks = [7, 11, 15]
    deltas = [1e-4, 3e-4, 1e-3]
    it = 0
    while time.time() - t0 < 138.0:
        remain = HARD - (time.time() - t0)
        if remain < 30:
            break
        it += 1
        # analyse current walker
        _, cnt, _, _ = _counts(cur_x, thr=1e-7)
        rattlers = np.where(cnt == 0)[0]
        order_hi = np.argsort(-cnt)  # hubs first
        # choose centre: rattler first, then hubs / random
        if len(rattlers) > 0 and (it % 5 == 1 or it <= 3):
            centre = int(rattlers[rng.integers(len(rattlers))])
        elif it % 3 == 0:
            centre = int(order_hi[rng.integers(min(8, n))])
        else:
            centre = int(rng.integers(n))
        k = patch_ks[(it - 1) % len(patch_ks)]
        # occasionally randomize k
        if rng.random() < 0.2:
            k = int(rng.choice(patch_ks))
        delta = deltas[(it - 1) % len(deltas)]
        t_target = 0.849102 - delta
        # victims: centre + k largest cosine to centre
        dots_c = cur_x @ cur_x[centre]
        dots_c[centre] = -2.0
        neigh = np.argsort(-dots_c)[:k]
        victims = np.unique(np.concatenate([np.array([centre]), neigh]))
        nv = len(victims)
        is_victim = np.zeros(n, dtype=bool)
        is_victim[victims] = True
        # freeze rest, reinsert one at a time at largest voids
        y = cur_x.copy()
        # random insertion order
        ins_order = victims.copy()
        rng.shuffle(ins_order)
        # current kept set grows
        kept_idx = np.where(~is_victim)[0]
        x_keep = (y[kept_idx] / np.linalg.norm(y[kept_idx], axis=1, keepdims=True)).copy()
        # we will maintain list of points
        cur_set = x_keep.tolist()
        import math as _m
        new_pts = {}
        for idx in ins_order:
            xa = np.array(cur_set)
            v = _largest_void(xa, rng, n_cand=20000)
            # tiny jitter to avoid exact symmetry
            v = v + rng.normal(0, 0.004, size=4)
            v /= np.linalg.norm(v)
            new_pts[int(idx)] = v
            cur_set.append(v.tolist())
        for idx, v in new_pts.items():
            y[idx] = v
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # optimize only reinserted points with target threshold
        moved = np.zeros(n, dtype=bool)
        moved[victims] = True
        # anneal subset: steps depend on patch size
        y = _subset_target_smooth(y, moved, t_target, steps=450, beta0=60.0, beta1=1600.0, lr0=0.006, lr1=0.0004)
        v_sub = _max_cos(y)
        # unfreeze: short all-free softmax anneal
        # skip if terrible
        if v_sub < min(cur_val + 0.006, best_val + 0.004):
            y = _full_smooth(y, steps=320, beta0=60.0, beta1=1300.0, lr0=0.003, lr1=0.0004)
            v_s = _max_cos(y)
        else:
            v_s = v_sub
        thresh = min(cur_val + 0.004, best_val + 0.0025)
        do_lp = (v_s < thresh) or (it % 12 == 0 and v_s < cur_val + 0.008)
        if do_lp:
            bud = 1.6 if v_s > best_val else 2.4
            if remain - bud < 26:
                bud = max(0.5, remain - 26)
            y_lp, v_lp = _lp_polish(y, bud)
        else:
            y_lp, v_lp = y, v_s
        improved = False
        if v_lp < best_val - 1e-12:
            x_best = y_lp.copy(); best_val = float(v_lp)
            cur_x = y_lp.copy(); cur_val = float(v_lp)
            improved = True
        else:
            # useful event is changed contact graph: keep within 0.002 for extra anneal
            if v_lp < best_val + 0.002:
                # one extra all-free anneal + LP
                if remain > 34:
                    y2 = _full_smooth(y_lp, steps=500, beta0=80.0, beta1=2000.0, lr0=0.0025, lr1=0.0002)
                    y2, v2 = _lp_polish(y2, 1.4)
                    if v2 < best_val - 1e-12:
                        x_best = y2.copy(); best_val = float(v2)
                        cur_x = y2.copy(); cur_val = float(v2)
                        improved = True
                    elif v2 < cur_val - 1e-12:
                        cur_x = y2.copy(); cur_val = float(v2)
                    elif v_lp < cur_val + 0.0015 and rng.random() < 0.4:
                        cur_x = y_lp.copy(); cur_val = float(v_lp)
                else:
                    if v_lp < cur_val and rng.random() < 0.7:
                        cur_x = y_lp.copy(); cur_val = float(v_lp)
            else:
                if v_lp < cur_val - 1e-12:
                    cur_x = y_lp.copy(); cur_val = float(v_lp)
                elif v_lp < cur_val + 0.002 and rng.random() < 0.18:
                    cur_x = y_lp.copy(); cur_val = float(v_lp)
        if it % 28 == 0:
            # restart walker near best with small kick to avoid drift
            cur_x = x_best.copy()
            cur_x += rng.normal(0, 0.02, size=cur_x.shape)
            cur_x /= np.linalg.norm(cur_x, axis=1, keepdims=True)
            cur_val = _max_cos(cur_x)
    fb = max(4.0, HARD - (time.time() - t0) - 2.0)
    fb = min(fb, 22.0)
    if fb > 0.5:
        x_best, best_val = _lp_polish(x_best, fb)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

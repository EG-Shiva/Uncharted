import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def contacts_info(x, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    cnt = np.sum(g >= m - tol, axis=1)
    row_max = np.max(g, axis=1)
    return m, cnt, row_max, g


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iu0_0 = iu0[0]
    iu0_1 = iu0[1]
    while True:
        if time.time() - start > seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0_0, iu0_1]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            break
        if nnz > 2500:
            # keep top 2500
            vals = c_all[sel]
            kth = vals.size - 2500
            thr = np.partition(vals, kth)[kth]
            sel = c_all >= thr
            # if still too many, subsample deterministically by stride
            nnz2 = int(np.count_nonzero(sel))
            if nnz2 > 3200:
                idx = np.where(sel)[0][::2]
                msk = np.zeros_like(sel)
                msk[idx] = True
                sel = msk
        I = iu0_0[sel]
        J = iu0_1[sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = np.arange(d)
        cols = np.concatenate([I[:, None] * d + ci, J[:, None] * d + ci, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_polish(x, iters=500, T0=0.015, T1=0.0015, lr0=0.01, lr1=0.002):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = np.sum(W, axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def largest_holes(x_fixed, rng, n_holes=4, n_sample=9000, n_refine=60):
    # find points on S^3 far from x_fixed (minimize max dot)
    cand = rng.normal(size=(n_sample, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_fixed.T
    worst = np.max(dots, axis=1)
    idx = np.argsort(worst)[:n_holes * 3]
    picks = cand[idx]
    # greedy farthest selection among candidates
    chosen = []
    rem = list(range(len(picks)))
    # start from very best
    chosen.append(0)
    rem.remove(0)
    while len(chosen) < n_holes and rem:
        # maximize min distance to already chosen + min dot to fixed set
        best_i = -1
        best_s = 1e9
        for i in rem:
            d_to_chosen = min(float(picks[i] @ picks[j]) for j in chosen)
            s = worst[idx[i]] + 0.3 * d_to_chosen
            if s < best_s:
                best_s = s
                best_i = i
        chosen.append(best_i)
        rem.remove(best_i)
    holes = picks[chosen]
    # local refine each hole: tangential repulsion ascent
    for h_i in range(len(holes)):
        h = holes[h_i]
        for _ in range(n_refine):
            d = x_fixed @ h  # (m,)
            # weights focused on nearest neighbours
            T = 0.02
            m0 = float(np.max(d))
            w = np.exp((d - m0) / T)
            g = w @ x_fixed  # pull toward neighbours
            # move away
            gt = g - float(g @ h) * h
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-12:
                break
            gt /= nrm
            h = h - 0.02 * gt
            h /= np.linalg.norm(h)
        holes[h_i] = h
    return holes


def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is None:
            continue
        c = np.asarray(c, dtype=float)
        if c.shape != (163, 4):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        if np.any(~np.isfinite(c)):
            continue
        cands.append(c)
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    n, d = 163, 4

    # quick initial LP to ensure local optimum baseline (short)
    if time.time() - t0 < 150:
        best = lp_polish(best, 4.0)
        best_val = max_cos(best)

    trial = 0
    # reserve ~25s for final polish
    while time.time() - t0 < 125 and trial < 80:
        trial += 1
        m, cnt, row_max, G = contacts_info(best, tol=1e-9)
        # identify worst points: involved in near-max pairs
        tight = G >= m - 1e-7
        np.fill_diagonal(tight, False)
        deg_tight = np.sum(tight, axis=1)
        order_worst = np.argsort(-deg_tight - 0.001 * (row_max))
        rattlers = np.where(cnt == 0)[0]
        overloaded = np.where(cnt >= 8)[0]
        y = best.copy()
        mode = trial % 6
        if mode == 0 and len(rattlers) > 0:
            # relocate rattlers to largest holes computed w.r.t. non-rattlers
            keep = np.ones(n, dtype=bool)
            keep[rattlers] = False
            # if many rattlers, keep at least n-6
            if np.count_nonzero(keep) < n - 6:
                keep[:] = True
                keep[rattlers[:2]] = False
            x_fixed = y[keep]
            nh = int(np.count_nonzero(~keep))
            nh = max(nh, len(rattlers))
            nh = min(nh, 6)
            holes = largest_holes(x_fixed, rng, n_holes=nh)
            rr = rattlers[:nh]
            for k, ridx in enumerate(rr):
                y[ridx] = holes[k % len(holes)]
            # small shake to neighbours of holes to open space
            y += rng.normal(scale=0.004, size=y.shape)
        elif mode == 1:
            # shake worst points strongly
            k = 8 + (trial % 5)
            idx = order_worst[:k]
            y[idx] += rng.normal(scale=0.055, size=(len(idx), d))
            # tiny global jitter
            y += rng.normal(scale=0.003, size=y.shape)
        elif mode == 2:
            # cap-localized shake
            u = rng.normal(size=4)
            u /= np.linalg.norm(u)
            dots = y @ u
            thr = 0.70 + 0.05 * ((trial // 6) % 4)
            mask = dots > thr
            if np.count_nonzero(mask) < 5:
                idx = np.argsort(dots)[-12:]
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True
            nm = int(np.count_nonzero(mask))
            y[mask] += rng.normal(scale=0.05, size=(nm, d))
            y += rng.normal(scale=0.002, size=y.shape)
        elif mode == 3:
            # overloaded resample: move 1-2 most loaded to a hole
            if len(overloaded) > 0:
                o = overloaded[np.argsort(-cnt[overloaded])][:2]
                keep = np.ones(n, dtype=bool)
                keep[o] = False
                holes = largest_holes(y[keep], rng, n_holes=len(o))
                for k, ridx in enumerate(o):
                    # keep 50% blend to preserve some memory
                    y[ridx] = holes[k] + 0.15 * rng.normal(size=4)
            else:
                y += rng.normal(scale=0.012, size=y.shape)
        elif mode == 4:
            # global moderate shake with decaying sigma
            sigma = [0.012, 0.025, 0.045, 0.008][(trial // 6) % 4]
            y += rng.normal(scale=sigma, size=y.shape)
        else:
            # antipodal-ish kick: pick a tight pair and push them apart + jitter
            iu = np.triu_indices(n, 1)
            Gm = y @ y.T
            np.fill_diagonal(Gm, -2.0)
            flat = Gm[iu]
            j = int(np.argmax(flat))
            a, b = int(iu[0][j]), int(iu[1][j])
            # move a and b apart along their difference projected
            diff = y[a] - y[b]
            diff /= (np.linalg.norm(diff) + 1e-12)
            y[a] += 0.04 * diff + rng.normal(scale=0.01, size=4)
            y[b] -= 0.04 * diff + rng.normal(scale=0.01, size=4)
            y += rng.normal(scale=0.004, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth
        y = smooth_polish(y, iters=450, T0=0.014, T1=0.0012, lr0=0.011, lr1=0.002)
        # short LP
        remain = 125 - (time.time() - t0)
        if remain < 1.0:
            break
        y = lp_polish(y, min(2.5, remain * 0.4))
        v = max_cos(y)
        if v < best_val - 1e-12:
            best = y
            best_val = v
        # sideways acceptance very rarely to diversify rattler count: if equal but fewer contacts
        elif abs(v - best_val) <= 5e-13 and trial % 10 == 0:
            _, cnt_y, _, _ = contacts_info(y, tol=1e-9)
            if int(np.sum(cnt_y)) < int(np.sum(cnt)):
                best = y
                best_val = v
    # final polish with remaining time (leave 5s margin)
    remain = 175 - (time.time() - t0)
    if remain > 3:
        best = smooth_polish(best, iters=300, T0=0.008, T1=0.0008, lr0=0.006, lr1=0.001)
        best = lp_polish(best, min(remain - 1.0, 28.0))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best

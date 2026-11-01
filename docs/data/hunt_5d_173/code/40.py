import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _contacts(x, tol=1e-9):
    rm, G = _row_max(x)
    m = float(rm.max())
    # contacts per point
    C = (m - G) < tol
    np.fill_diagonal(C, False)
    return C.sum(axis=1), m, G, rm

def _lp_standard(x, deadline, radius0=1e-3, max_pairs=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        if np.count_nonzero(sel) > max_pairs:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:max_pairs]
            msel = np.zeros_like(sel, dtype=bool)
            msel[idx[order]] = True
            sel = msel
        I, J = iu0[sel], iu1[sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _lp_custom_r(x, deadline, r, radius0=2e-3, slack=5e-3, k_near=15):
    # include all pairs within slack of max + nearest k_near pairs of r
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        sel = (m - c_all) < slack
        # nearest k of r
        # find pairs involving r with largest c
        # c_all index for pairs with r: mask
        mask_r = (iu0 == r) | (iu1 == r)
        idx_r = np.where(mask_r)[0]
        if len(idx_r) > 0:
            order = np.argsort(c_all[idx_r])[::-1][:k_near]
            sel[idx_r[order]] = True
        if np.count_nonzero(sel) > 3500:
            idx = np.where(sel)[0]
            # always keep r-nearest
            must = set(idx_r[np.argsort(c_all[idx_r])[::-1][:k_near]].tolist())
            # rank rest by tightness
            order = np.argsort(c_all[idx])[::-1][:3500]
            msel = np.zeros_like(sel, dtype=bool)
            msel[idx[order]] = True
            for mm in must:
                msel[mm] = True
            sel = msel
        I, J = iu0[sel], iu1[sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _forcing_step(x, cap0, r, radius):
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iu0, iu1]
    # non-r pairs near cap0
    thresh = cap0 - max(20.0 * radius, 1e-7)
    mask_nonr = (iu0 != r) & (iu1 != r)
    sel_nonr = mask_nonr & (c_all > thresh)
    mask_r = ~mask_nonr  # pairs involving r
    # include all r pairs (172 rows) - cheap
    sel = sel_nonr | mask_r
    I, J = iu0[sel], iu1[sel]
    csel = c_all[sel]
    is_r_pair = (I == r) | (J == r)
    p = len(I)
    nv = n * d + 1
    # Build A_ub: for non-r: <xJ,dxI>+<xI,dxJ> <= cap0 - c
    # for r: <xJ,dxI>+<xI,dxJ> - t_r <= -c
    rows_list = []
    cols_list = []
    vals_list = []
    b_ub = np.empty(p)
    for k in range(p):
        i, j = int(I[k]), int(J[k])
        base = k * (2 * d + 1)
        # dxI block
        for a in range(d):
            rows_list.append(k); cols_list.append(i * d + a); vals_list.append(x[j, a])
        for a in range(d):
            rows_list.append(k); cols_list.append(j * d + a); vals_list.append(x[i, a])
        if is_r_pair[k]:
            rows_list.append(k); cols_list.append(n * d); vals_list.append(-1.0)
            b_ub[k] = -csel[k]
        else:
            rows_list.append(k); cols_list.append(n * d); vals_list.append(0.0)
            b_ub[k] = cap0 - csel[k]
    A_ub = coo_matrix((np.array(vals_list), (np.array(rows_list), np.array(cols_list))), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    bounds = [(-radius, radius)] * (n * d) + [(None, None)]
    # freeze r: dx_r = 0
    for a in range(d):
        bounds[r * d + a] = (0.0, 0.0)
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                  bounds=bounds, method="highs")
    return res, I, J, is_r_pair

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (173, 5):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((173, 5))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    x_best = None
    best = 2.0
    for a in cands:
        v = _max_cos(a)
        if v < best:
            best = v
            x_best = a.copy()
    # ensure local optimum briefly
    if BUDGET - (time.time() - t0) > 12:
        x_best, best = _lp_standard(x_best, time.time() + 3.0, radius0=1e-3)
    cap0 = best
    # identify rattler r (zero-contact point)
    cnt, m, G, rm = _contacts(x_best, tol=1e-9)
    r = int(np.argmin(cnt))
    # if no rattler, pick loosest row max
    if cnt[r] > 0:
        # pick minimal row max (loosest)
        r = int(np.argmin(rm))
    r_max0 = float(rm[r])
    x_cur = x_best.copy()
    # ---- Phase A: freeze r, push others away, cap non-r at cap0, minimize r-max ----
    eps_cycle = [1e-4, 3e-4, 1e-3, 3e-3]
    radius = 0.015
    t_phaseA_end = t0 + 55.0
    # current r max
    _, _, _, rm_cur = _contacts(x_cur, tol=1e-9)
    cur_rmax = float(rm_cur[r])
    # also track non-r max
    itA = 0
    while time.time() < t_phaseA_end and radius > 1e-6:
        itA += 1
        eps = eps_cycle[itA % len(eps_cycle)]
        try:
            res, I, J, is_r = _forcing_step(x_cur, cap0, r, radius)
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(173, 5)
        # enforce freeze exactly
        dx[r, :] = 0.0
        t_pred = float(res.x[-1])
        y = x_cur + dx
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        # evaluate
        Gy = y @ y.T
        np.fill_diagonal(Gy, -2.0)
        # non-r max
        mask = np.ones((173, 173), dtype=bool)
        mask[r, :] = False; mask[:, r] = False
        max_nonr = float(Gy[mask].max())
        rmax_new = float(Gy[r, :].max())
        # accept if non-r stays <= cap0 (+tiny) and r improves
        if max_nonr <= cap0 + 2e-9 and rmax_new < cur_rmax - 5e-9:
            x_cur = y
            cur_rmax = rmax_new
            radius = min(radius * 1.25, 0.04)
            # if we achieved record-eps for current eps, keep going to larger eps
            if cur_rmax <= cap0 - 3e-3:
                # hole large enough
                pass
        else:
            # check if feasible but no improvement: shrink
            # if prediction says no progress, shrink
            if t_pred >= cur_rmax - 1e-12:
                radius *= 0.6
            else:
                radius *= 0.7
        if radius < 2e-4 and itA > 30:
            # restart radius a bit to try next epsilon
            radius = 0.008
        if itA > 120:
            break
        if time.time() > t_phaseA_end:
            break
    x_forced = x_cur.copy()
    forced_rmax = cur_rmax
    # ---- Phase B: unfreeze r, joint polish with r-biased set, accept >=3 contacts ----
    # track balanced candidate (may be worse temporarily)
    x_bal = x_forced.copy()
    # evaluate balanced max
    bal_max = _max_cos(x_bal)
    cnt_bal, _, _, _ = _contacts(x_bal, tol=1e-7)
    r_cnt = int(cnt_bal[r])
    # if forcing didn't help, still try activation from best
    t_phaseB_end = t0 + 125.0
    sigmas = [0.004, 0.01, 0.025, 0.05, 0.008, 0.015]
    trial = 0
    # keep best balanced with >=3 contacts even if worse (within 5e-3)
    x_bal_best = None
    bal_best_val = 2.0
    # also consider forced itself if it already has contacts (unlikely)
    while time.time() < t_phaseB_end:
        trial += 1
        remain = t_phaseB_end - time.time()
        if remain < 3.0:
            break
        sigma = sigmas[trial % len(sigmas)]
        x_try = x_forced.copy()
        # perturb rattler + small global jitter to change topology
        x_try[r] = x_try[r] + sigma * rng.standard_normal(5)
        x_try[r] /= np.linalg.norm(x_try[r])
        # occasionally jitter its nearest neighbours too
        if trial % 3 == 0:
            Gt = x_try @ x_try.T
            np.fill_diagonal(Gt, -2.0)
            nbrs = np.argsort(Gt[r])[::-1][:6]
            x_try[nbrs] += 0.01 * rng.standard_normal((len(nbrs), 5))
            xn = np.linalg.norm(x_try[nbrs], axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x_try[nbrs] /= xn
        # occasionally relocate rattler toward edge of its hole to force contacts:
        # move it partway toward its nearest neighbour
        if trial % 4 == 0:
            Gt2 = x_forced @ x_forced.T
            np.fill_diagonal(Gt2, -2.0)
            nn = int(np.argmax(Gt2[r]))
            # slerp partway
            alpha = 0.05 + 0.05 * rng.random()
            v = x_try[nn] - (x_try[nn] @ x_try[r]) * x_try[r]
            nv_ = np.linalg.norm(v)
            if nv_ > 1e-12:
                v /= nv_
                x_try[r] = np.cos(alpha) * x_try[r] + np.sin(alpha) * v
                x_try[r] /= np.linalg.norm(x_try[r])
        lp_time = min(1.6, remain - 0.5)
        y, v = _lp_custom_r(x_try, time.time() + lp_time, r, radius0=3e-3, slack=5e-3, k_near=15)
        cnt_y, _, _, rm_y = _contacts(y, tol=1e-7)
        rc = int(cnt_y[r])
        # acceptance: if rattler has >=3 contacts, keep as balanced candidate even if worse
        if rc >= 3:
            # allow up to cap0+5e-3 temporarily
            if v < bal_best_val and v <= cap0 + 5e-3:
                bal_best_val = v
                x_bal_best = y.copy()
            # also update x_forced to this balanced point to continue search (sideways move)
            # keep exploring around balanced region
            if v <= cap0 + 5e-3:
                x_forced = y.copy()
                # re-evaluate rattler? keep original r for forcing inclusion
                pass
        if v < best - 1e-12:
            x_best, best = y.copy(), v
            cap0 = best  # tighten cap for further forcing? keep original? update
        # sideways: if close to best and balanced, diversify
        if trial % 9 == 0 and v < best + 2e-7 and rc >= 2:
            # brief standard polish to squeeze
            y2, v2 = _lp_standard(y, time.time() + 1.0, radius0=5e-4)
            if v2 < best - 1e-12:
                x_best, best = y2, v2
                cap0 = best
            if rc >= 3 and v2 <= cap0 + 5e-3 and v2 < bal_best_val:
                bal_best_val = v2
                x_bal_best = y2.copy()
        if trial > 60:
            break
    # if we found a balanced candidate better than or close to best, polish it hard
    remain = BUDGET - (time.time() - t0)
    if x_bal_best is not None and remain > 5.0:
        # polish balanced candidate with standard + custom alternating
        y, v = _lp_custom_r(x_bal_best, time.time() + min(3.0, remain / 3), r, radius0=1e-3, slack=5e-3, k_near=15)
        if v < best - 1e-12:
            x_best, best = y, v
        # final standard squeeze
        remain2 = BUDGET - (time.time() - t0)
        if remain2 > 3.0:
            y2, v2 = _lp_standard(y, time.time() + remain2 - 1.0, radius0=5e-4)
            if v2 < best:
                x_best, best = y2, v2
            # if balanced and within noise, prefer balanced (load sharing)
            cntb, _, _, _ = _contacts(y2, tol=1e-7)
            if int(cntb[r]) >= 3 and v2 <= best + 1e-9:
                x_best, best = y2, v2
    # ---- Phase C: final squeeze of global best with remaining time ----
    remain = BUDGET - (time.time() - t0)
    if remain > 2.5:
        x_best, best = _lp_standard(x_best, time.time() + remain - 0.5, radius0=5e-4)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best

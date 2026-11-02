import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _build_full_A(p, I, J, x, nv, d):
    Cols = np.empty((p, 2 * d + 1), dtype=np.int64)
    Cols[:, :d] = I[:, None] * d + np.arange(d)
    Cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
    Cols[:, 2 * d] = nv - 1
    Vals = np.empty((p, 2 * d + 1), dtype=np.float64)
    Vals[:, :d] = x[J]
    Vals[:, d:2 * d] = x[I]
    Vals[:, 2 * d] = -1.0
    rows = np.repeat(np.arange(p), 2 * d + 1)
    A_ub = coo_matrix((Vals.ravel(), (rows, Cols.ravel())), shape=(p, nv)).tocsr()
    return A_ub

def _lp_full(x, deadline, r0=0.012, rmax=0.05, cap=1500):
    n, d = x.shape
    x = _norm(x.copy())
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    nv = n * d + 1
    iu0, iu1 = np.triu_indices(n, 1)
    erows = np.repeat(np.arange(n), d)
    ecols = np.arange(n * d)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size > cap:
            part = np.argpartition(c_all, -cap)[-cap:]
            sel = part
        c_sel = c_all[sel]
        I = iu0[sel]; J = iu1[sel]
        p = sel.size
        A_ub = _build_full_A(p, I, J, x, nv, d)
        A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x, best

def _lp_subset(x, free, deadline, r0=0.04, rmax=0.08, cap=1500):
    n, d = x.shape
    free = np.asarray(free, dtype=np.int64)
    k = free.size
    if k == 0:
        return x.copy(), _max_cos(x)
    x = _norm(x.copy())
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    nv = k * d + 1
    pos = np.full(n, -1, dtype=np.int64)
    pos[free] = np.arange(k)
    free_mask = np.zeros(n, dtype=bool); free_mask[free] = True
    iu0, iu1 = np.triu_indices(n, 1)
    # A_eq data (fixed x restricted to free)
    xf = x[free]
    erows = np.repeat(np.arange(k), d)
    ecols = np.arange(k * d)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size > cap:
            part = np.argpartition(c_all, -cap)[-cap:]
            sel = part
        I = iu0[sel]; J = iu1[sel]
        c_sel = c_all[sel]
        p = sel.size
        both = free_mask[I] & free_mask[J]
        ionly = free_mask[I] & (~free_mask[J])
        jonly = (~free_mask[I]) & free_mask[J]
        # count entries
        n_both = int(both.sum()); n_io = int(ionly.sum()); n_jo = int(jonly.sum())
        # build row/col/val lists via vectorized blocks
        rows_list = []; cols_list = []; vals_list = []
        # helper to append group
        # both: 2d+1 per row
        idx_b = np.where(both)[0]
        if idx_b.size > 0:
            Ib = I[idx_b]; Jb = J[idx_b]
            pb = idx_b.size
            Cb = np.empty((pb, 2 * d + 1), dtype=np.int64)
            Cb[:, :d] = pos[Ib][:, None] * d + np.arange(d) if False else (pos[Ib][:, None] * 1 * d + np.arange(d))
            # pos mapping: local index * d + dim
            Cb[:, :d] = pos[Ib][:, None] * d + np.arange(d)
            Cb[:, d:2 * d] = pos[Jb][:, None] * d + np.arange(d)
            Cb[:, 2 * d] = nv - 1
            Vb = np.empty((pb, 2 * d + 1), dtype=np.float64)
            Vb[:, :d] = x[Jb]
            Vb[:, d:2 * d] = x[Ib]
            Vb[:, 2 * d] = -1.0
            rb = np.repeat(idx_b, 2 * d + 1)
            rows_list.append(rb); cols_list.append(Cb.ravel()); vals_list.append(Vb.ravel())
        idx_i = np.where(ionly)[0]
        if idx_i.size > 0:
            Ii = I[idx_i]; Ji = J[idx_i]
            pi = idx_i.size
            Ci = np.empty((pi, d + 1), dtype=np.int64)
            Ci[:, :d] = pos[Ii][:, None] * d + np.arange(d)
            Ci[:, d] = nv - 1
            Vi = np.empty((pi, d + 1), dtype=np.float64)
            Vi[:, :d] = x[Ji]
            Vi[:, d] = -1.0
            ri = np.repeat(idx_i, d + 1)
            rows_list.append(ri); cols_list.append(Ci.ravel()); vals_list.append(Vi.ravel())
        idx_j = np.where(jonly)[0]
        if idx_j.size > 0:
            Ij = I[idx_j]; Jj = J[idx_j]
            pj = idx_j.size
            Cj = np.empty((pj, d + 1), dtype=np.int64)
            Cj[:, :d] = pos[Jj][:, None] * d + np.arange(d)
            Cj[:, d] = nv - 1
            Vj = np.empty((pj, d + 1), dtype=np.float64)
            Vj[:, :d] = x[Ij]
            Vj[:, d] = -1.0
            rj = np.repeat(idx_j, d + 1)
            rows_list.append(rj); cols_list.append(Cj.ravel()); vals_list.append(Vj.ravel())
        # fixed-fixed: only t column
        none_mask = ~(both | ionly | jonly)
        idx_n = np.where(none_mask)[0]
        if idx_n.size > 0:
            rows_list.append(idx_n)
            cols_list.append(np.full(idx_n.size, nv - 1, dtype=np.int64))
            vals_list.append(np.full(idx_n.size, -1.0, dtype=np.float64))
        if rows_list:
            R = np.concatenate(rows_list); C = np.concatenate(cols_list); V = np.concatenate(vals_list)
        else:
            R = np.empty(0, dtype=np.int64); C = np.empty(0, dtype=np.int64); V = np.empty(0, dtype=np.float64)
        A_ub = coo_matrix((V, (R, C)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x[free].ravel(), (erows, ecols)), shape=(k, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bounds = [(-radius, radius)] * (k * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(k),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
            continue
        dx = np.zeros((n, d))
        dx[free] = res.x[:-1].reshape(k, d)
        y = _norm(x + dx)
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x, best

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_HARD = t0 + 176.0
    T_EXPLORE = t0 + 160.0
    rng = _np.random.default_rng(11)
    try:
        x0 = _np.load("best.npy").astype(float)
    except Exception:
        x0 = _np.load("record.npy").astype(float)
    try:
        xr = _np.load("record.npy").astype(float)
    except Exception:
        xr = x0
    x0 = _norm(x0); xr = _norm(xr)
    v0 = _max_cos(x0); vr = _max_cos(xr)
    if vr < v0:
        x_best, best = xr, vr
    else:
        x_best, best = x0, v0
    n, d = x_best.shape
    # initial quick full polish
    if _time.time() < T_EXPLORE:
        xb, vb = _lp_full(x_best, min(_time.time() + 3.0, T_EXPLORE), r0=0.004, rmax=0.02)
        if vb < best:
            x_best, best = xb, vb
    iu0, iu1 = _np.triu_indices(n, 1)
    it = 0
    while _time.time() < T_EXPLORE:
        it += 1
        now = _time.time()
        if now >= T_EXPLORE or now > T_HARD - 14:
            break
        G = x_best @ x_best.T
        _np.fill_diagonal(G, -2.0)
        thr = best - 0.008
        deg = (G > thr).sum(axis=1)
        order = _np.argsort(deg)  # ascending loose first
        mode = it % 6
        y = x_best.copy()
        if mode == 1:
            k = 5
            free = order[:k]
            for idx in free:
                v = rng.standard_normal(size=(d,)); v /= _np.linalg.norm(v)
                y[int(idx)] = v
            dl = min(now + 0.9, T_HARD - 14)
            yp, vp_sub = _lp_subset(y, free, dl, r0=0.05, rmax=0.09)
            # check true value
            vp = _max_cos(yp)
            if vp < best - 2e-6:
                dl2 = min(_time.time() + 2.0, T_HARD - 12)
                yf, vf = _lp_full(yp, dl2, r0=0.010, rmax=0.04)
                if vf < best:
                    x_best, best = yf, vf
                    continue
            if vp < best:
                # full polish opportunistically
                dl2 = min(_time.time() + 1.5, T_HARD - 12)
                yf, vf = _lp_full(yp, dl2, r0=0.008, rmax=0.03)
                if vf < best:
                    x_best, best = yf, vf
            continue
        elif mode == 2:
            k = 3
            free = rng.choice(order[:12], size=k, replace=False)
            for idx in free:
                # void opposite centroid of close neighbours
                ii = int(idx)
                row = (x_best @ x_best.T)[ii]
                nbrs = _np.where(row > best - 0.01)[0]
                if len(nbrs) > 0:
                    cent = x_best[nbrs].mean(axis=0)
                    nn = _np.linalg.norm(cent)
                    if nn > 1e-12:
                        cent /= nn
                        v = -cent + 0.4 * rng.standard_normal(size=(d,))
                        v /= _np.linalg.norm(v)
                        y[ii] = v
                    else:
                        v = rng.standard_normal(size=(d,)); v /= _np.linalg.norm(v)
                        y[ii] = v
                else:
                    v = rng.standard_normal(size=(d,)); v /= _np.linalg.norm(v)
                    y[ii] = v
            dl = min(now + 0.9, T_HARD - 14)
            yp, _ = _lp_subset(y, free, dl, r0=0.05, rmax=0.09)
            vp = _max_cos(yp)
            if vp < best - 2e-6:
                dl2 = min(_time.time() + 2.0, T_HARD - 12)
                yf, vf = _lp_full(yp, dl2, r0=0.010, rmax=0.04)
                if vf < best:
                    x_best, best = yf, vf
            elif vp < best:
                dl2 = min(_time.time() + 1.2, T_HARD - 12)
                yf, vf = _lp_full(yp, dl2, r0=0.008, rmax=0.03)
                if vf < best:
                    x_best, best = yf, vf
            continue
        elif mode == 3:
            # single valley reinsert + neighbours free
            idx = int(order[rng.integers(0, 6)])
            v = rng.standard_normal(size=(d,)); v /= _np.linalg.norm(v)
            y[idx] = v
            # free = idx + 2 other loose
            others = rng.choice(order[:12], size=2, replace=False)
            free = _np.unique(_np.concatenate(([idx], others)))
            dl = min(now + 0.8, T_HARD - 14)
            yp, _ = _lp_subset(y, free, dl, r0=0.05, rmax=0.09)
            vp = _max_cos(yp)
            if vp < best:
                dl2 = min(_time.time() + 1.5, T_HARD - 12)
                yf, vf = _lp_full(yp, dl2, r0=0.008, rmax=0.03)
                if vf < best:
                    x_best, best = yf, vf
            continue
        elif mode == 4:
            # break a max contact involving a loose point
            c = (x_best @ x_best.T)[iu0, iu1]
            # restrict to pairs touching loose set
            loose_set = set(order[:15].tolist())
            # top 60 pairs
            top = _np.argpartition(c, -60)[-60:]
            # prefer those touching loose
            cand = [t for t in top if int(iu0[t]) in loose_set or int(iu1[t]) in loose_set]
            if len(cand) == 0:
                cand = top.tolist()
            pick = int(rng.choice(_np.asarray(cand)))
            i, j = int(iu0[pick]), int(iu1[pick])
            idx = i if rng.random() < 0.5 else j
            y[idx] = y[idx] + 0.35 * rng.standard_normal(size=(d,))
            y[idx] /= _np.linalg.norm(y[idx])
            free = _np.unique(_np.concatenate(([idx], rng.choice(order[:10], size=3, replace=False))))
            dl = min(now + 0.8, T_HARD - 14)
            yp, _ = _lp_subset(y, free, dl, r0=0.04, rmax=0.08)
            vp = _max_cos(yp)
            if vp < best:
                dl2 = min(_time.time() + 1.5, T_HARD - 12)
                yf, vf = _lp_full(yp, dl2, r0=0.008, rmax=0.03)
                if vf < best:
                    x_best, best = yf, vf
            continue
        elif mode == 5:
            # two random reinsertions (global diversification)
            idxs = rng.choice(n, size=2, replace=False)
            for idx in idxs:
                v = rng.standard_normal(size=(d,)); v /= _np.linalg.norm(v)
                y[int(idx)] = v
            free = idxs
            dl = min(now + 0.8, T_HARD - 14)
            yp, _ = _lp_subset(y, free, dl, r0=0.05, rmax=0.09)
            vp = _max_cos(yp)
            if vp < best - 1e-4:
                dl2 = min(_time.time() + 2.0, T_HARD - 12)
                yf, vf = _lp_full(yp, dl2, r0=0.010, rmax=0.04)
                if vf < best:
                    x_best, best = yf, vf
            elif vp < best + 0.005 and vp < best + 0.002:
                # still try full if not too bad and we have time rarely
                if rng.random() < 0.15:
                    dl2 = min(_time.time() + 1.0, T_HARD - 12)
                    yf, vf = _lp_full(yp, dl2, r0=0.008, rmax=0.03)
                    if vf < best:
                        x_best, best = yf, vf
            continue
        else:  # mode 0: rare global nudge + full polish
            sigma = 0.010 + 0.008 * rng.random()
            y = y + sigma * rng.standard_normal(size=y.shape)
            y = _norm(y)
            dl = min(now + 2.0, T_HARD - 12)
            if dl <= now + 0.3:
                break
            yp, vp = _lp_full(y, dl, r0=0.012, rmax=0.05)
            if vp < best:
                x_best, best = yp, vp
            continue
    rem = T_HARD - _time.time()
    if rem > 3:
        xb, vb = _lp_full(x_best, _time.time() + rem - 0.5, r0=0.005, rmax=0.03)
        if vb < best:
            x_best, best = xb, vb
    x_best = _norm(x_best)
    return x_best

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_IU = None

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _top_pairs(c_all, cap):
    n = c_all.shape[0]
    if n <= cap:
        return np.arange(n)
    # indices of largest cap
    part = np.argpartition(-c_all, cap - 1)[:cap]
    return part

def _build_lp(x, sel_idx, iu, radius):
    n, d = x.shape
    I = iu[0][sel_idx]
    J = iu[1][sel_idx]
    c = (x @ x.T)[I, J] if False else None
    return I, J

def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    global _IU
    x = _norm(x)
    n, d = x.shape
    if _IU is None:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0, iu1 = iu
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        cnt = sel.size
        if cnt == 0:
            break
        if cnt > cap:
            # keep largest cap
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 400:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
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
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def _smooth(x, iters, T0, T1, lr0, lr1):
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _dual_order(x, radius=2e-3, cap=1500):
    """Return (sel_I, sel_J, sel_c, order_by_stress). order indexes into sel arrays."""
    global _IU
    n, d = x.shape
    iu0, iu1 = _IU
    G = x @ x.T
    c_all = G[iu0, iu1]
    m = float(c_all.max())
    thr = m - max(20.0 * radius, 1e-9)
    sel = np.where(c_all > thr)[0]
    if sel.size > cap:
        v = c_all[sel]
        kth = np.sort(v)[-cap]
        sel = np.where(c_all >= kth)[0]
        if sel.size > cap + 400:
            sel = sel[::2][:cap]
    I = iu0[sel]; J = iu1[sel]
    c = c_all[sel]
    p = len(I)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cd = np.arange(d)
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
        order = np.argsort(-c)
        return I, J, c, order
    if res.status != 0:
        order = np.argsort(-c)
        return I, J, c, order
    try:
        marg = None
        if hasattr(res, 'ineqlin') and res.ineqlin is not None:
            iq = res.ineqlin
            if hasattr(iq, 'marginals'):
                marg = np.asarray(iq.marginals).ravel()
            elif isinstance(iq, dict) and 'marginals' in iq:
                marg = np.asarray(iq['marginals']).ravel()
            else:
                try:
                    marg = np.asarray(iq['marginals']).ravel()
                except Exception:
                    marg = None
        if marg is None:
            # try lowercase / alternative attr
            marg = np.asarray(getattr(res, 'marginals', None)).ravel()
    except Exception:
        marg = None
    if marg is None or marg.size != p:
        order = np.argsort(-c)
        return I, J, c, order
    stress = -marg  # positive for active
    order = np.argsort(-stress, kind='stable')
    return I, J, c, order

def _forced_run(x, force_I, force_J, target, seconds, r_init=0.012, cap=1200):
    """Constrained reopt: force pairs (force_I[k],force_J[k]) <= target, minimise max of rest."""
    global _IU
    x = _norm(x)
    n, d = x.shape
    iu0, iu1 = _IU
    # map forced pair to linear index for quick lookup (set of encoded)
    fenc = set((int(a) * 1000 + int(b)) for a, b in zip(force_I, force_J))
    # also need c for forced
    G0 = x @ x.T
    fc = G0[force_I, force_J] if len(force_I) > 0 else np.zeros(0)
    radius = r_init
    start = time.time()
    it = 0
    while time.time() - start < seconds and radius > 1e-12 and it < 12:
        it += 1
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        # candidate pool: top cap by cosine
        if c_all.size > cap:
            sel = _top_pairs(c_all, cap)
        else:
            sel = np.arange(c_all.size)
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        f = len(force_I)
        # build combined A_ub: p rows with t, f rows without t
        # rows 0..p-1 : xJ*dxI + xI*dxJ - t <= -c
        # rows p..p+f-1 : xJf*dxIf + xIf*dxJf <= target - fc_current
        # fc_current recomputed from current x
        if f > 0:
            Gf = G[force_I, force_J]
            rhs_f = target - Gf
        else:
            rhs_f = np.zeros(0)
        nv = n * d + 1
        # build sparse
        # first block
        rows1 = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols1 = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                                np.full((p, 1), n * d)], 1).ravel()
        vals1 = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        if f > 0:
            rows2 = np.repeat(p + np.arange(f), 2 * d)
            cols2 = np.concatenate([np.asarray(force_I)[:, None] * d + cd,
                                    np.asarray(force_J)[:, None] * d + cd], 1).ravel()
            vals2 = np.concatenate([x[force_J], x[force_I]], 1).ravel()
            rows = np.concatenate([rows1, rows2])
            cols = np.concatenate([cols1, cols2])
            vals = np.concatenate([vals1, vals2])
            b_ub = np.concatenate([-c, rhs_f])
            nrows = p + f
        else:
            rows, cols, vals = rows1, cols1, vals1
            b_ub = -c
            nrows = p
        A_ub = coo_matrix((vals, (rows, cols)), shape=(nrows, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            # if infeasible due to too-tight forcing, relax radius? keep shrinking
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        # accept even if max increased (temporary), but keep trust region sane
        # check forcing progress
        Gy = y @ y.T
        mf = float(Gy[force_I, force_J].max()) if f > 0 else -2
        # move to y regardless (to escape basin), unless NaN
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        x = y
        # adapt radius: if forced max still above target, keep large; else shrink slightly
        if f > 0 and mf > target + 1e-4:
            radius = min(radius * 1.2, 0.02)
        else:
            radius = min(radius * 1.05, 0.02)
    return x

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    HARD = 168.0
    global _IU
    n, d = 163, 4
    _IU = np.triu_indices(n, 1)
    xb = None; xr = None
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
        if c is not None:
            try:
                c = np.asarray(c, dtype=float)
                if c.shape == (163, 4):
                    c = _norm(np.nan_to_num(c))
                    cands.append(c)
            except Exception:
                pass
    if not cands:
        z = rng.normal(size=(163, 4)); z = _norm(z)
        cands.append(z)
    best = None; best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    base_val = best_val
    # dual-guided stress order at record/best
    try:
        sI, sJ, sC, sOrd = _dual_order(best)
    except Exception:
        G = best @ best.T
        iu0, iu1 = _IU
        c_all = G[iu0, iu1]
        o = np.argsort(-c_all)[:1500]
        sI, sJ = iu0[o], iu1[o]
        sC = c_all[o]
        sOrd = np.argsort(-sC)
    # contact degrees at 1e-7-ish for high-degree cluster
    G = best @ best.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    deg = (G >= m0 - 1e-7).sum(axis=1)
    high_pts = np.argsort(-deg)[:6]
    # build forced clusters (each 20-40 pairs)
    clusters = []
    # A: highest-dual edges
    for sz, dl in [(30, 8e-4), (25, 1.2e-3), (35, 2e-3), (20, 5e-4)]:
        idx = sOrd[:sz]
        clusters.append((sI[idx].copy(), sJ[idx].copy(), dl))
    # B: high-degree points incident edges
    # collect incident active pairs among sel
    # build lookup from pair to stress rank for sorting
    pair_rank = {}
    for r, k in enumerate(sOrd):
        pair_rank[(int(sI[k]), int(sJ[k]))] = r
    for hp_choice in [high_pts[:3], high_pts[3:6], high_pts[:4]]:
        cand = []
        for k in range(len(sI)):
            a, b = int(sI[k]), int(sJ[k])
            if a in hp_choice or b in hp_choice:
                cand.append(k)
        # sort cand by stress order
        cand = sorted(cand, key=lambda k: pair_rank.get((int(sI[k]), int(sJ[k])), 9999))[:32]
        if len(cand) >= 15:
            ci = np.array(cand)
            for dl in [8e-4, 1.5e-3]:
                clusters.append((sI[ci].copy(), sJ[ci].copy(), dl))
                break
    # C: spatially localized patches (3 seeds)
    for s in range(3):
        if time.time() - t0 > 20:
            break
        ctr = int(rng.integers(0, n))
        dots = best[ctr] @ best.T
        # nearest 18
        nn = np.argsort(-dots)[:18]
        nnset = set(int(v) for v in nn)
        cand = []
        for k in range(len(sI)):
            a, b = int(sI[k]), int(sJ[k])
            if a in nnset or b in nnset:
                cand.append(k)
        if len(cand) < 12:
            # fallback: top pairs among patch by current cosine
            sub = best[nn] @ best[nn].T
            # map to global pairs: take all pairs inside nn with high sub value
            tri = np.triu_indices(len(nn), 1)
            vals = sub[tri]
            o = np.argsort(-vals)[:30]
            fi = nn[tri[0][o]]; fj = nn[tri[1][o]]
            clusters.append((fi.copy(), fj.copy(), 1e-3))
        else:
            ci = np.array(cand[:30])
            clusters.append((sI[ci].copy(), sJ[ci].copy(), 1e-3 if s % 2 == 0 else 6e-4))
    # deduplicate / limit to ~10 clusters
    clusters = clusters[:12]
    base_graph = (G >= m0 - 1e-5)
    best_graph = base_graph.copy()
    cur_best = best.copy(); cur_val = best_val
    # surgery loop
    for ci, (fI, fJ, delta) in enumerate(clusters):
        if time.time() - t0 > HARD - 35:
            break
        target = base_val - delta
        # start from best + small shake to break exact jam
        y = cur_best.copy()
        # tiny shake scaled by trial
        y = y + rng.normal(scale=0.004, size=y.shape)
        y = _norm(y)
        # forced reopt ~1.2s
        remain = HARD - 35 - (time.time() - t0)
        if remain < 3:
            break
        y = _forced_run(y, np.asarray(fI), np.asarray(fJ), target, seconds=min(1.4, remain / 3.0))
        # quick smooth to relieve
        y = _smooth(y, iters=220, T0=0.012, T1=0.0018, lr0=0.006, lr1=0.0009)
        # unforced polish ~1.0s
        remain2 = HARD - 35 - (time.time() - t0)
        if remain2 > 0.6:
            y = _lp_polish(y, seconds=min(1.1, remain2 / 2.5), r0=1.5e-3)
        v = _max_cos(y)
        # graph difference at 1e-5
        Gy = y @ y.T
        np.fill_diagonal(Gy, -2.0)
        my = float(Gy.max())
        graph = (Gy >= my - 1e-5)
        diff = int(np.count_nonzero(graph != base_graph) // 2)
        # keep if better, or if diverse and not much worse (to explore new basin)
        if v < cur_val - 1e-12:
            cur_best = y.copy(); cur_val = v
            if v < best_val - 1e-12:
                best = y.copy(); best_val = v
                best_graph = graph.copy()
        else:
            # accept diverse topology as new starting point occasionally
            if diff > 120 and v < cur_val + 2e-3 and (ci % 3 == 0):
                cur_best = y.copy(); cur_val = v
            elif v < best_val - 1e-12:
                best = y.copy(); best_val = v
        # if we improved over base, update base_val reference slightly for next targets?
        # keep original base_val for forcing magnitude
    # final polish best
    elapsed = time.time() - t0
    budget = max(4.0, min(30.0, 172.0 - elapsed))
    t1 = time.time()
    best = _smooth(best, iters=250, T0=0.008, T1=0.0012, lr0=0.004, lr1=0.0007)
    left = budget - (time.time() - t1)
    if left > 1.5:
        best = _lp_polish(best, seconds=left, r0=1.2e-3)
    best = _norm(np.nan_to_num(best))
    return best

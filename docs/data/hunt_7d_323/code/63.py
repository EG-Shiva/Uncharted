import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import solve_triangular

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _contact_info(x, thresh=1e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > m - thresh
    I = iu[0][sel].astype(np.int64)
    J = iu[1][sel].astype(np.int64)
    csel = c[sel]
    # per point counts with same thresh
    cnt = np.sum(G > m - thresh, axis=1)
    return m, I, J, csel, cnt, G

def _build_projector(x, I, J, csel):
    n, d = x.shape
    m = len(I)
    if m == 0:
        return None
    A = x[J] - csel[:, None] * x[I]
    B = x[I] - csel[:, None] * x[J]
    JJt = np.zeros((m, m), dtype=np.float64)
    # accumulate V V^T per point
    for p in range(n):
        # find incident rows
        # vectorized search
        mI = (I == p)
        mJ = (J == p)
        if not (np.any(mI) or np.any(mJ)):
            continue
        idx = np.where(mI | mJ)[0]
        k = len(idx)
        V = np.empty((k, d), dtype=np.float64)
        # map
        # need to know for each idx position whether I or J
        for ii, kk in enumerate(idx):
            if I[kk] == p:
                V[ii] = A[kk]
            else:
                V[ii] = B[kk]
        JJt[np.ix_(idx, idx)] += V @ V.T
    # regularize
    reg = 1e-7
    JJt += reg * np.eye(m)
    try:
        L = np.linalg.cholesky(JJt)
    except Exception:
        JJt += 1e-6 * np.eye(m)
        try:
            L = np.linalg.cholesky(JJt)
        except Exception:
            return None
    return (L, JJt, A, B)

def _project_null(Vr, x, I, J, proj):
    # Vr assumed tangent (n,d)
    if proj is None:
        return Vr
    L, JJt, A, B = proj
    # w = J Vr
    w = np.sum(A * Vr[I], axis=1) + np.sum(B * Vr[J], axis=1)
    try:
        z = solve_triangular(L, w, lower=True, check_finite=False)
        y = solve_triangular(L.T, z, lower=False, check_finite=False)
    except Exception:
        return Vr
    corr = np.zeros_like(Vr)
    # corr = J^T y
    np.add.at(corr, I, A * y[:, None])
    np.add.at(corr, J, B * y[:, None])
    Vn = Vr - corr
    # re-tangentize (remove numerical radial)
    Vn -= (np.sum(Vn * x, axis=1)[:, None]) * x
    return Vn

def _exp_move(x, D):
    # D tangent (n,d), per-point great-circle step
    nr = np.linalg.norm(D, axis=1)
    out = np.empty_like(x)
    # avoid div0
    big = nr > 1e-15
    out[~big] = x[~big]
    cb = np.cos(nr[big])
    sb = np.sin(nr[big]) / nr[big]
    out[big] = cb[:, None] * x[big] + sb[:, None] * D[big]
    return _norm(out)

def _lse_relax(x0, steps=200, tau=0.007):
    x = x0.copy()
    n, d = x.shape
    # step schedule rms per step
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights
        # shift for stability
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        F = E @ x  # (n,d)
        Ft = F - (np.sum(F * x, axis=1)[:, None]) * x
        rms = np.sqrt(np.mean(np.sum(Ft * Ft, axis=1)))
        if rms < 1e-15:
            break
        # schedule 0.008 -> 0.002
        frac = s / max(1, steps - 1)
        sstep = 0.008 * (1 - frac) + 0.002 * frac
        x = x - (sstep / rms) * Ft
        x = _norm(x)
    return x

def _lp_polish_short(x, seconds, t_end):
    # very short trust-region LP, max few iterations
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 3e-3
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end - 0.3 and it < 4:
        if time.time() >= t_end - 0.3:
            break
        if seconds is not None and seconds <= 0:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2500
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cr = cols.reshape(p, 2 * d + 1)
        vr = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        cr[:, :d] = I[:, None] * d + ar
        cr[:, d:2 * d] = J[:, None] * d + ar
        cr[:, 2 * d] = n * d
        vr[:, :d] = x[J]
        vr[:, d:2 * d] = x[I]
        vr[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.5, t_end - time.time() - 0.2)})
        except Exception:
            radius *= 0.5
            it += 1
            continue
        if res.status != 0:
            radius *= 0.5
            it += 1
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; bx = y.copy(); best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        it += 1
        if radius < 1e-12:
            break
    return bx, best

def solve():
    t0 = time.time()
    T_HARD = 150.0
    t_end = t0 + T_HARD
    rng = np.random.default_rng(12345)
    # load incumbent
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        a = rng.standard_normal(size=(323, 7))
        best_x = _norm(a)
        best_v = _max_cos(best_x)
    x_inc = best_x.copy()
    v_inc = best_v
    # initial contact info + projector
    def rebuild(x):
        m, I, J, csel, cnt, G = _contact_info(x, thresh=1e-7)
        proj = None
        # only build if m reasonable
        if 100 < len(I) < 5000:
            try:
                proj = _build_projector(x, I, J, csel)
            except Exception:
                proj = None
        # old mask over triu for diff
        n = x.shape[0]
        iu = np.triu_indices(n, 1)
        G2 = x @ x.T
        np.fill_diagonal(G2, -2.0)
        c2 = G2[iu]
        mask = c2 > m - 1e-7
        return m, I, J, csel, cnt, mask, proj, iu
    m_inc, I_inc, J_inc, csel_inc, cnt_inc, mask_inc, proj_inc, iu_all = rebuild(x_inc)
    iuI, iuJ = iu_all
    n, d = x_inc.shape
    lp_count = 0
    max_lp = 5
    trial = 0
    # weights for low-contact emphasis
    while time.time() < t_end - 4.0:
        trial += 1
        if time.time() >= t_end - 4.0:
            break
        # check remaining; rebuild weights occasionally
        # random step size 0.03-0.20
        s_target = float(rng.uniform(0.03, 0.20))
        K = int(rng.integers(5, 31))
        # per-point boost for rattlers/low contacts
        wgt = 1.0 + 4.0 * np.exp(-cnt_inc.astype(np.float64) / 5.0)
        Dacc = np.zeros((n, d), dtype=np.float64)
        ok = False
        for _ in range(K):
            if time.time() >= t_end - 4.0:
                break
            Vr = rng.standard_normal(size=(n, d))
            Vr -= (np.sum(Vr * x_inc, axis=1)[:, None]) * x_inc
            Vr *= wgt[:, None]
            # normalize Vr fro
            fn = np.linalg.norm(Vr)
            if fn < 1e-12:
                continue
            Vr /= fn
            Vn = _project_null(Vr, x_inc, I_inc, J_inc, proj_inc)
            fn2 = np.linalg.norm(Vn)
            if fn2 < 1e-9:
                # row-space vector, try another
                continue
            Vn /= fn2
            Dacc += float(rng.standard_normal()) * Vn
            ok = True
        if not ok:
            continue
        fd = np.linalg.norm(Dacc)
        if fd < 1e-12:
            continue
        Dacc /= fd
        Dacc *= s_target * np.sqrt(n)
        # large step
        x_try = _exp_move(x_inc, Dacc)
        # cheap 200-step LSE relaxation
        try:
            x_rel = _lse_relax(x_try, steps=200, tau=0.007)
        except Exception:
            continue
        v_rel = _max_cos(x_rel)
        # contact graph diff vs incumbent
        try:
            Gr = x_rel @ x_rel.T
            np.fill_diagonal(Gr, -2.0)
            mr = float(Gr.max())
            cr = Gr[iuI, iuJ]
            mask_r = cr > mr - 1e-7
            diff = int(np.sum(mask_r != mask_inc))
        except Exception:
            continue
        if diff < 50:
            continue
        # only polish if not terrible and we have LP budget and time
        if v_rel > v_inc + 0.01:
            continue
        if lp_count >= max_lp:
            # no LP budget: check direct improvement from relax
            if v_rel < v_inc - 1e-12:
                x_inc = x_rel.copy(); v_inc = v_rel
                m_inc, I_inc, J_inc, csel_inc, cnt_inc, mask_inc, proj_inc, _ = rebuild(x_inc)
            continue
        # one short LP polish, max ~3 sec
        remain = t_end - time.time()
        if remain < 5.0:
            break
        lp_budget = min(3.0, remain - 2.0)
        if lp_budget <= 0.5:
            break
        try:
            x_pol, v_pol = _lp_polish_short(x_rel, lp_budget, time.time() + lp_budget)
        except Exception:
            continue
        lp_count += 1
        if v_pol < v_inc:
            x_inc = x_pol.copy(); v_inc = float(v_pol)
            # rebuild projector for new basin
            try:
                m_inc, I_inc, J_inc, csel_inc, cnt_inc, mask_inc, proj_inc, _ = rebuild(x_inc)
            except Exception:
                pass
            # early exit if big win? continue searching
        # time check
        if time.time() >= t_end - 1.0:
            break
    # always return incumbent if no improvement; ensure normalized
    if v_inc < best_v:
        return _norm(x_inc)
    else:
        return _norm(best_x)

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[I0, J0]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.argpartition(-cf, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            cnt = 3000
        I = I0[sel]; J = J0[sel]
        c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            from scipy.sparse import coo_matrix
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _active_pairs(X, p_cap=900, slack_cap=1e-6):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    # number within slack_cap
    order = np.argsort(-c)
    # count within cap
    within = int(np.sum(c > m - slack_cap))
    p = min(p_cap, within)
    p = max(p, 600)
    p = min(p, len(c))
    sel = order[:p]
    I = iu[0][sel]; J = iu[1][sel]
    return I, J, m, c

def _build_J_sparse(X, I, J):
    p = len(I)
    n, d = X.shape
    N = n * d
    # tangent-projected rows: at i: x_j - c x_i, at j: x_i - c x_j
    Xi = X[I]
    Xj = X[J]
   cdot = np.sum(Xi * Xj, axis=1)
    Vi = Xj - cdot[:, None] * Xi
    Vj = Xi - cdot[:, None] * Xj
    rows = np.repeat(np.arange(p), 2 * d)
    cols_i = (I[:, None] * d + np.arange(d)).ravel()
    cols_j = (J[:, None] * d + np.arange(d)).ravel()
    cols = np.concatenate([cols_i, cols_j]) if False else None
    # interleave per row: first d for I block, next d for J block
    cols = np.empty(p * 2 * d, dtype=np.int64)
    cols[0::2*d] = 0  # placeholder
    # build row by row vectorised
    # rows already repeat; cols order must match rows order (I block then J block per row)
    cI = (I[:, None] * d + np.arange(d))
    cJ = (J[:, None] * d + np.arange(d))
    cols = np.empty((p, 2 * d), dtype=np.int64)
    cols[:, :d] = cI
    cols[:, d:] = cJ
    cols = cols.ravel()
    data = np.concatenate([Vi, Vj], axis=1).ravel()
    Jmat = csr_matrix((data, (rows, cols)), shape=(p, N))
    return Jmat

def _null_direction(X, Jmat, scales, rng):
    n, d = X.shape
    N = n * d
    R = rng.randn(n, d)
    # tangent project
    nd = np.sum(R * X, axis=1, keepdims=True)
    R = R - nd * X
    # bias away from overloaded: scale rows
    R = R * scales[:, None]
    # re-tangent (scales keep tangent)
    r = R.ravel()
    nr = float(np.linalg.norm(r))
    if nr < 1e-14:
        return None
    # solve min ||J^T lam - r|| via LSMR
    try:
        A = Jmat.T.tocsr()
        # damp regularises singular JJ^T
        out = lsmr(A, r, damp=1e-6, atol=1e-7, btol=1e-7, maxiter=250, show=False)
        lam = out[0]
        pred = A.dot(lam)
        dx = r - pred
    except Exception:
        dx = r
    # enforce tangent exactly
    DX = dx.reshape(n, d)
    nd2 = np.sum(DX * X, axis=1, keepdims=True)
    DX = DX - nd2 * X
    nrm = float(np.linalg.norm(DX))
    if not np.isfinite(nrm) or nrm < 1e-14:
        return None
    return DX / nrm

def _step_accept(X, DX, m0, tol=1e-7):
    # largest step with max <= m0+tol
    # candidate grid ascending
    grid = [0.002, 0.005, 0.01, 0.02, 0.035, 0.06, 0.10, 0.16]
    best_t = None
    best_X = None
    best_m = None
    # evaluate in order; keep last acceptable (monotone approx)
    for t in grid:
        Y = _norm(X + t * DX)
        m = _max_cos(Y)
        if m <= m0 + tol:
            best_t = t; best_X = Y; best_m = m
        else:
            # if increase is large, larger t likely worse -> break
            if m > m0 + tol + 5e-6 and t > 0.02:
                break
            # otherwise continue? still record break
            # stop at first violation to keep continuity
            break
    return best_t, best_X, best_m

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    rng = np.random.RandomState(20261103)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape
    # short initial tighten (cheap)
    # main nullspace crawl walks
    walk = 0
    # adaptive p_cap alternates to vary null dim
    p_caps = [850, 900, 950, 800, 700]
    while time.time() < deadline - 35.0:
        walk += 1
        p_cap = p_caps[(walk - 1) % len(p_caps)]
        Xcur = X_best.copy()
        m_start = _max_cos(Xcur)
        Iact, Jact, m0, _ = _active_pairs(Xcur, p_cap=p_cap, slack_cap=1e-6)
        p = len(Iact)
        # contact counts for bias
        cnt = np.zeros(n, dtype=float)
        np.add.at(cnt, Iact, 1); np.add.at(cnt, Jact, 1)
        # bias away from overloaded: downweight high-contact points
        # alternate strength
        beta = 0.12 + 0.08 * ((walk % 3) / 2.0)
        scales = 1.0 / (1.0 + beta * cnt)
        # rattlers get boost
        scales = scales / scales.mean()
        init_set = set(zip(Iact.tolist(), Jact.tolist()))
        # topology thresholds
        enter_thresh = 1e-6
        sep_thresh = 6e-6
        crawled = 0
        max_steps = 30
        # precompute G for enter detection: need full pair list
        for step in range(max_steps):
            if time.time() > deadline - 35.0:
                break
            Jmat = _build_J_sparse(Xcur, Iact, Jact)
            DX = _null_direction(Xcur, Jmat, scales, rng)
            if DX is None:
                # fresh random tangent
                R = rng.randn(n, d)
                R = R - np.sum(R * Xcur, axis=1, keepdims=True) * Xcur
                nrm = np.linalg.norm(R)
                if nrm < 1e-14:
                    break
                DX = R / nrm
            m0cur = _max_cos(Xcur)
            bt, bX, bm = _step_accept(Xcur, DX, m0cur, tol=1e-7)
            if bt is None or bX is None:
                # try smaller step
                Y = _norm(Xcur + 0.0008 * DX)
                if _max_cos(Y) <= m0cur + 1e-7:
                    Xcur = Y; crawled += 1
                    continue
                break
            Xcur = bX
            crawled += 1
            # check topology change against initial manifold
            G = Xcur @ Xcur.T
            # separation of active contacts
            c_act = G[Iact, Jact]
            m_cur = float(np.max(G[np.triu_indices(n, 1)]))
            # material separation?
            if float(np.min(c_act)) < m_cur - sep_thresh:
                break
            # new entrant? check if any non-active pair within enter_thresh of max
            # cheap: find max outside set via top-k
            # find pairs with cosine > m_cur - enter_thresh
            # use full iu (13041 pairs) - cheap
            iu0 = np.triu_indices(n, 1)
            cf = G[iu0]
            mask_close = cf > m_cur - enter_thresh
            # if number of close pairs grew materially or a new pair appeared
            # detect new pair
            # to avoid O(n^2) set check each time, only every few steps
            if step % 3 == 2 or step == max_steps - 1:
                # build quick hash: check top 1200
                order = np.argpartition(-cf, min(1200, len(cf)-1))[:1200]
                found_new = False
                for k in order:
                    if cf[k] > m_cur - enter_thresh:
                        a = int(iu0[0][k]); b = int(iu0[1][k])
                        if a > b:
                            a, b = b, a
                        if (a, b) not in init_set:
                            found_new = True
                            break
                if found_new:
                    break
            # if step size collapsed, stop
            if bt < 0.004 and crawled > 8:
                # try new random direction next iter (continue) but limit
                pass
        if crawled == 0:
            continue
        # LP polish the crawled point
        remain = deadline - time.time() - 4.0
        if remain < 3.0:
            break
        pol_sec = min(5.0, remain)
        # vary polish radius by walk parity via short pre-smooth?
        Xp = _lp_polish(Xcur, seconds=pol_sec, deadline=deadline - 2.0)
        sp = _max_cos(Xp)
        if sp < s_best - 1e-12:
            X_best = Xp.copy(); s_best = sp
            # keep crawling from new best immediately (reset walk counter bias)
        elif sp < s_best + 0.0005 and (walk % 5 == 0):
            # occasional: replace current with diverse elite to avoid re-crawl same basin?
            pass
        # time check
        if time.time() > deadline - 32.0:
            break
    # final intensive polish of best
    remain = deadline - time.time() - 1.5
    if remain > 3.0:
        X_best = _lp_polish(X_best, seconds=min(remain, 18.0), deadline=deadline - 1.0)
    return _norm(X_best)

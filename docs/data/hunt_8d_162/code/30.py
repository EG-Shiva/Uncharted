import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

try:
    from scipy.sparse.linalg import lsmr
    _HAS_LSMR = True
except Exception:
    _HAS_LSMR = False

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, r0=0.004, rmax=0.03):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    t0 = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cf > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.argpartition(-cf, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        I = iuI[sel]; J = iuJ[sel]; c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).ravel(),
                               (J[:, None] * d + np.arange(d)).ravel(),
                               np.full(p, n * d)])
        vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        dx = res.x[:-1].reshape(n, d)
        Y = X + dx
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best - 1e-15:
            X = Y; best = v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
        if time.time() - t0 >= seconds:
            break
    return X

def _build_active(X, thr=1e-6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iuI, iuJ = np.triu_indices(n, 1)
    c = G[iuI, iuJ]
    slack = m - c
    sel = slack < thr
    I = iuI[sel]; J = iuJ[sel]
    return m, G, I, J, sel

def _contact_counts(n, I, J):
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, I, 1)
    np.add.at(cnt, J, 1)
    return cnt

def _null_direction(X, I, J, cnt, rng, damp=1e-4, maxiter=120):
    n, d = X.shape
    N = n * d
    p = len(I)
    # random tangent field, bias away from overloaded
    Z = rng.standard_normal(size=(n, d))
    # tangent projection
    dot = np.sum(Z * X, axis=1, keepdims=True)
    Z = Z - dot * X
    w = 1.0 / (1.0 + cnt.astype(float) / 4.0)
    # keep some floor so overloaded still move a bit
    w = np.maximum(w, 0.08)
    # normalize energy
    w = w * np.sqrt(n / np.sum(w ** 2))
    Z = Z * w[:, None]
    r = Z.ravel()
    nr = np.linalg.norm(r)
    if nr < 1e-14:
        return None
    r = r / nr
    if p == 0 or not _HAS_LSMR:
        V = Z / max(np.linalg.norm(Z, axis=1).max(), 1e-12)
        return V
    # sparse Jacobian p x N
    rows = np.repeat(np.arange(p), 2 * d)
    cI = (I[:, None] * d + np.arange(d)).ravel()
    cJ = (J[:, None] * d + np.arange(d)).ravel()
    cols = np.concatenate([cI, cJ])
    # order: for each row, first d cols I-block then d cols J-block
    # rows repeat already interleaves, need to interleave cols per row
    # rebuild correctly per-row interleaved:
    cols2 = np.empty(p * 2 * d, dtype=np.int64)
    cols2[0::2*d] = 0  # placeholder
    # vectorised interleave
    baseI = (I[:, None] * d + np.arange(d))
    baseJ = (J[:, None] * d + np.arange(d))
    inter = np.concatenate([baseI, baseJ], axis=1).ravel()
    valsX = np.concatenate([X[J], X[I]], axis=1).ravel()
    Jmat = coo_matrix((valsX, (rows, inter)), shape=(p, N)).tocsr()
    try:
        A = Jmat.T.tocsr()  # N x p
        out = lsmr(A, r, damp=damp, atol=1e-7, btol=1e-7, maxiter=maxiter)
        y = out[0]
        v = r - A.dot(y)
    except Exception:
        return None
    V = v.reshape(n, d)
    # re-project to tangent (remove normal drift)
    dot2 = np.sum(V * X, axis=1, keepdims=True)
    V = V - dot2 * X
    mx = np.linalg.norm(V, axis=1).max()
    if not np.isfinite(mx) or mx < 1e-12:
        return None
    # check null residual: want J v small
    try:
        jv = Jmat.dot(v)
        if np.linalg.norm(jv) > 0.5 * np.linalg.norm(r) * 2.0 and False:
            pass
    except Exception:
        pass
    V = V / mx
    # down-weight overloaded direction already in r; also zero tiny?
    return V

def _retract(X, V, t):
    Y = X + t * V
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _largest_step(X, V, cur_max, allow=1e-7, t_init=0.03, t_max=0.12):
    # find largest t with max <= cur_max+allow
    def val(t):
        Y = _retract(X, V, t)
        return _max_cos(Y)
    v0 = val(t_init)
    if v0 > cur_max + allow:
        t_hi = t_init
        t_lo = 0.0
        # shrink
        t = t_init
        while t > 1e-6 and val(t) > cur_max + allow:
            t *= 0.5
        if t <= 1e-6:
            return 0.0
        t_lo = t
        t_hi = t * 2.0
        # bisect lo-hi
        for _ in range(12):
            mid = 0.5 * (t_lo + t_hi)
            if val(mid) <= cur_max + allow:
                t_lo = mid
            else:
                t_hi = mid
        return t_lo
    else:
        t_lo = t_init
        t = t_init
        while t < t_max:
            t2 = min(t * 2.0, t_max)
            if val(t2) <= cur_max + allow:
                t_lo = t2
                t = t2
                if t >= t_max - 1e-12:
                    break
            else:
                break
        t_hi = min(t * 2.0, t_max)
        if t_hi <= t_lo:
            return t_lo
        # if hi is bad, bisect
        if val(t_hi) > cur_max + allow:
            for _ in range(12):
                mid = 0.5 * (t_lo + t_hi)
                if val(mid) <= cur_max + allow:
                    t_lo = mid
                else:
                    t_hi = mid
        else:
            t_lo = t_hi
        return t_lo

def solve():
    t_start = time.time()
    BUDGET = 175.0
    rng = np.random.default_rng(20261103)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (162, 8):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (162, 8):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        a = rng.standard_normal(size=(162, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    X_best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = 162, 8

    # quick initial LP to sync
    if time.time() - t_start < BUDGET - 60:
        X_best = _lp_polish(X_best, 4.0)
        best_val = _max_cos(X_best)

    nwalks = 0
    # several random nullspace walks
    while time.time() - t_start < 128.0:
        nwalks += 1
        # build active set for current best
        m, G, I, J, sel = _build_active(X_best, thr=1e-6)
        cnt = _contact_counts(n, I, J)
        # vary damp / threshold slightly per walk for diversity
        damp = float(rng.choice([3e-5, 1e-4, 3e-4, 1e-3]))
        mit = int(rng.choice([80, 120, 160]))
        # crawl chain
        X_cur = X_best.copy()
        cur_max = _max_cos(X_cur)
        # store old active pair set for topology detection
        old_pairs = set(zip(I.tolist(), J.tolist()))
        crawled = False
        nsteps = 10
        for step in range(nsteps):
            if time.time() - t_start > 130.0:
                break
            m2, _, I2, J2, _ = _build_active(X_cur, thr=1e-6)
            cnt2 = _contact_counts(n, I2, J2)
            V = _null_direction(X_cur, I2, J2, cnt2, rng, damp=damp, maxiter=mit)
            if V is None:
                # random tangent kick biased
                Z = rng.standard_normal(size=(n, d))
                Z -= np.sum(Z * X_cur, axis=1, keepdims=True) * X_cur
                V = Z / max(np.linalg.norm(Z, axis=1).max(), 1e-12)
            # randomize sign? keep as is; also try flipping occasionally
            if rng.random() < 0.1:
                V = -V
            t_star = _largest_step(X_cur, V, cur_max, allow=1e-7, t_init=0.03, t_max=0.10)
            if t_star < 3e-4:
                # try opposite direction
                V2 = -V
                t2 = _largest_step(X_cur, V2, cur_max, allow=1e-7, t_init=0.03, t_max=0.10)
                if t2 > t_star:
                    t_star = t2
                    V = V2
            if t_star < 3e-4:
                break
            X_cur = _retract(X_cur, V, t_star)
            cur_max = _max_cos(X_cur)
            crawled = True
            # topology change? check new active set
            _, _, I3, J3, _ = _build_active(X_cur, thr=1e-6)
            new_pairs = set(zip(I3.tolist(), J3.tolist()))
            entered = len(new_pairs - old_pairs)
            left = len(old_pairs - new_pairs)
            # separation: any old active slack now large?
            # if topology changed materially, break to polish
            if entered >= 1 or left >= 5:
                break
            # also if max drifted up too much, break
            if cur_max > best_val + 5e-7:
                break
            old_pairs = new_pairs
        if not crawled:
            # tiny random tangent nudge to escape exact jam
            Z = rng.standard_normal(size=(n, d))
            Z -= np.sum(Z * X_best, axis=1, keepdims=True) * X_best
            Z /= max(np.linalg.norm(Z, axis=1).max(), 1e-12)
            X_cur = _retract(X_best, Z, 0.004)
            cur_max = _max_cos(X_cur)
        # LP polish the crawled point (short)
        remain = BUDGET - (time.time() - t_start)
        if remain < 12:
            break
        X_pol = _lp_polish(X_cur, min(4.0, remain - 8.0))
        v_pol = _max_cos(X_pol)
        if v_pol < best_val - 1e-12:
            X_best = X_pol
            best_val = v_pol
        else:
            # even if not better, occasionally keep equal-manifold move to continue exploring
            # keep the crawled+polished point if within 1e-7 and topology different
            if v_pol <= best_val + 1e-7 and nwalks % 3 == 0:
                # adopt as new base to continue crawling elsewhere (diversify)
                # only if it changes contacts
                _, _, Ia, Ja, _ = _build_active(X_pol, thr=1e-6)
                _, _, Ib, Jb, _ = _build_active(X_best, thr=1e-6)
                if len(Ia) != len(Ib):
                    X_best = X_pol
                    # best_val unchanged (within tol) but keep exploring; don't update best_val upward
                    cur_max_keep = v_pol
                    if cur_max_keep < best_val + 5e-8:
                        pass
        if time.time() - t_start > BUDGET - 40:
            break

    remain = BUDGET - (time.time() - t_start) - 2.0
    if remain > 4:
        X_best = _lp_polish(X_best, remain)
        best_val = _max_cos(X_best)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    return X_best

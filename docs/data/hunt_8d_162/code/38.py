import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
try:
    from scipy.sparse.linalg import lsmr
    _HAS_LSMR = True
except Exception:
    _HAS_LSMR = False

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iu0]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            idx = np.argpartition(-cfull[sel], 3500)[:3500]
            I = I[idx]; J = J[idx]
            p = len(I)
        c = (X @ X.T)[I, J]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _contact_info(X, thresh=1e-6):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    cf = G[iu]
    mask = cf > m - thresh
    I = iu[0][mask]; J = iu[1][mask]
    cnt = np.zeros(n, dtype=int)
    # counts with tighter threshold for overload sense
    for i, j in zip(I, J):
        cnt[i] += 1
        cnt[j] += 1
    return m, G, I, J, cnt

def _nullspace_dir(X, I, J, cnt, rng, bias_strength=1.0):
    n, d = X.shape
    p = len(I)
    N = n * d
    M = p + n
    # build Ja (M x N)
    # rows 0..p-1 active, rows p..p+n-1 tangent
    r_rows = []
    r_cols = []
    r_vals = []
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        xi = X[i]; xj = X[j]
        for a in range(d):
            r_rows.append(k); r_cols.append(i*d+a); r_vals.append(float(xj[a]))
            r_rows.append(k); r_cols.append(j*d+a); r_vals.append(float(xi[a]))
    for k in range(n):
        r = p + k
        for a in range(d):
            r_rows.append(r); r_cols.append(k*d+a); r_vals.append(float(X[k, a]))
    Ja = coo_matrix((np.array(r_vals), (np.array(r_rows), np.array(r_cols))), shape=(M, N)).tocsr()
    # random tangent field with bias away from overloaded points
    R = rng.randn(n, d)
    nd = (R * X).sum(axis=1, keepdims=True)
    R = R - nd * X
    # bias: move loose points more, overloaded less
    # weight w_i = 1/(1+cnt/mean) ; also boost rattlers
    mean_c = float(cnt.mean()) if cnt.mean() > 0 else 10.0
    w = 1.0 / (1.0 + bias_strength * cnt / max(mean_c, 1.0))
    # extra boost for zero-contact
    w = np.where(cnt == 0, 2.0, w)
    R = R * w[:, None]
    # normalize to avg row norm 1
    f = np.linalg.norm(R) / np.sqrt(n)
    if f < 1e-12:
        return None, None
    R = R / f
    b = R.ravel()
    if not _HAS_LSMR:
        # fallback: just return tangent random
        D = R
        D = D - ((D*X).sum(axis=1, keepdims=True))*X
        return D, Ja
    try:
        # solve min_y ||Ja.T y - b||
        res = lsmr(Ja.T, b, maxiter=min(300, max(80, M//4)), atol=1e-8, btol=1e-8, show=False)
        y = res[0]
        corr = Ja.T.dot(y)
        Dvec = b - corr
        D = Dvec.reshape(n, d)
        # re-project to tangent to clean normal drift
        nd2 = (D * X).sum(axis=1, keepdims=True)
        D = D - nd2 * X
        # normalize: max row norm 1
        rn = np.linalg.norm(D, axis=1).max()
        if rn < 1e-12 or not np.isfinite(rn):
            return None, None
        D = D / rn
        return D, Ja
    except Exception:
        return None, None

def _crawl_step(X, D, m0, tol=1e-7, tmax=0.02):
    # find largest t with maxcos <= m0+tol, exponential search
    def val(t):
        Y = _norm(X + t * D)
        return _max_cos(Y), Y
    t = 2e-4
    best_t = 0.0
    best_Y = None
    # if even smallest fails, try smaller
    v, Y = val(t)
    if v > m0 + tol:
        for t2 in [5e-5, 2e-5, 1e-5]:
            v2, Y2 = val(t2)
            if v2 <= m0 + tol:
                return t2, Y2
        return 0.0, None
    best_t, best_Y = t, Y
    while t < tmax:
        t2 = t * 1.5
        if t2 > tmax:
            t2 = tmax
        v2, Y2 = val(t2)
        if v2 <= m0 + tol:
            best_t, best_Y = t2, Y2
            if t2 >= tmax - 1e-12:
                break
            t = t2
        else:
            break
    return best_t, best_Y

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
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
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape

    # quick LP sync of best (1s) in case best.npy not polished
    # skip if time tight; do tiny polish
    # nullspace walks
    n_walks = 10
    for walk in range(n_walks):
        if time.time() > deadline - 30:
            break
        # start from best with tiny random kick to get different nullspace sample
        # but mostly crawl from best itself
        if walk == 0:
            Xc = X_best.copy()
        else:
            # small jitter then re-project? keep close to stay in basin
            sig = 0.002 if walk % 2 == 0 else 0.005
            Xc = _norm(X_best + rng.randn(n, d) * sig)
            # one quick LP to return near manifold? no, crawl directly
        m0, G0, I0, J0, cnt0 = _contact_info(Xc, thresh=1e-6)
        # record initial active set signature
        init_pairs = set(zip([int(a) for a in I0], [int(b) for b in J0]))
        # bias strength varies per walk
        bias = 0.8 + 0.4 * (walk % 3)
        # active thresh varies
        thresh = 1e-6 if walk % 2 == 0 else 3e-7
        m0, G0, I0, J0, cnt0 = _contact_info(Xc, thresh=thresh)
        init_pairs = set(zip([int(a) for a in I0], [int(b) for b in J0]))
        cur_X = Xc; cur_m = _max_cos(cur_X)
        # cap active size for speed
        if len(I0) > 3500:
            # keep tightest 3500
            Gv = G0[I0, J0]
            idx = np.argpartition(-Gv, 3500)[:3500]
            I0 = I0[idx]; J0 = J0[idx]
        steps_done = 0
        for step in range(18):
            if time.time() > deadline - 28:
                break
            _, _, Ia, Ja_, cnta = _contact_info(cur_X, thresh=thresh)
            if len(Ia) > 3500:
                Gv = (cur_X @ cur_X.T)[Ia, Ja_]
                idx = np.argpartition(-Gv, 3500)[:3500]
                Ia = Ia[idx]; Ja_ = Ja_[idx]
            if len(Ia) < 10:
                break
            D, _ = _nullspace_dir(cur_X, Ia, Ja_, cnta, rng, bias_strength=bias)
            if D is None:
                break
            # random sign already random; line search
            cm = _max_cos(cur_X)
            t, Y = _crawl_step(cur_X, D, cm, tol=1e-7, tmax=0.015)
            if t == 0.0 or Y is None:
                # try opposite direction
                t2, Y2 = _crawl_step(cur_X, -D, cm, tol=1e-7, tmax=0.015)
                if t2 == 0.0 or Y2 is None:
                    break
                t, Y = t2, Y2
            cur_X = Y
            steps_done += 1
            # check topology change: new pair entered or active separated
            _, _, Ib, Jb, _ = _contact_info(cur_X, thresh=thresh)
            new_pairs = set(zip([int(a) for a in Ib], [int(b) for b in Jb]))
            # separation: previously active now slack > 5e-6 ?
            Gc = cur_X @ cur_X.T
            sep = False
            if len(I0) > 0:
                # check a sample of init pairs
                sl = cm - Gc[I0[:min(len(I0),500)], J0[:min(len(J0),500)]] if 'I0' in locals() else None
                if sl is not None and np.any(sl > 5e-6):
                    sep = True
            entered = len(new_pairs - init_pairs) > 0
            # also if max rose, stop? we keep <= +1e-7 so fine
            if entered or sep or steps_done >= 6:
                break
        # LP polish the crawled point briefly
        remain = deadline - time.time()
        if remain < 8:
            break
        budget = min(5.0, remain / 4.0)
        if budget < 1.5:
            budget = 1.5
        Xp = _lp_polish(cur_X, budget, t_deadline=deadline - 18)
        sp = _max_cos(Xp)
        if sp < s_best - 1e-12:
            X_best, s_best = Xp, sp
            # early extended polish on improvement
            if time.time() < deadline - 20:
                Xp2 = _lp_polish(X_best, min(6.0, deadline - time.time() - 14), t_deadline=deadline - 12)
                sp2 = _max_cos(Xp2)
                if sp2 < s_best:
                    X_best, s_best = Xp2, sp2
        # also try rattler-to-hole kick from crawled point if no improvement
        # (cheap, keeps diversity)
        if walk % 3 == 2 and time.time() < deadline - 25:
            try:
                Gc = cur_X @ cur_X.T
                np.fill_diagonal(Gc, -2.0)
                nn = Gc.max(axis=1)
                gm = float(nn.max())
                cntT = (Gc > gm - 1e-9).sum(axis=1)
                # victim: most overloaded
                v = int(np.argmax(cntT))
                mask = np.ones(n, dtype=bool); mask[v] = False
                # largest hole
                C = _norm(rng.randn(8000, d))
                dots = C @ cur_X[mask].T
                hole = C[int(np.argmin(dots.max(axis=1)))]
                Xt = cur_X.copy(); Xt[v] = hole
                Xt = _norm(Xt)
                Xt2 = _lp_polish(Xt, 2.0, t_deadline=deadline - 18)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
            except Exception:
                pass
    # final polish best
    if time.time() < deadline - 4:
        budget = min(12.0, deadline - time.time() - 1.5)
        if budget > 1.5:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 0.5)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best

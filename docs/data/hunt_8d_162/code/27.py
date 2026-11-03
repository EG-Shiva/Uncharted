import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

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
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argpartition(-cfull[sel], 3000)[:3000]
            I = I[idx]; J = J[idx]
            p = len(I)
        c = (X[I] * X[J]).sum(axis=1)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _null_direction(X, I, J, w, rng):
    n, d = X.shape
    p = len(I)
    N = n * d
    if p == 0:
        R = rng.randn(n, d)
        R -= (R * X).sum(axis=1, keepdims=True) * X
        return R, 0.0
    dots = (X[I] * X[J]).sum(axis=1)
    BI = X[J] - dots[:, None] * X[I]
    BJ = X[I] - dots[:, None] * X[J]
    # dense JPt p x N
    JPt = np.zeros((p, N), dtype=float)
    # fill
    for k in range(p):
        ii = int(I[k]); jj = int(J[k])
        JPt[k, ii*d:(ii+1)*d] = BI[k]
        JPt[k, jj*d:(jj+1)*d] = BJ[k]
    R = rng.randn(n, d)
    R -= (R * X).sum(axis=1, keepdims=True) * X
    R *= w[:, None]
    r = R.ravel()
    rn = float(np.linalg.norm(r))
    if rn < 1e-12:
        return R, 0.0
    b = JPt @ r
    M = JPt @ JPt.T
    # damp
    tr = float(np.trace(M) / max(1, p))
    damp = 1e-9 * max(1e-12, tr) + 1e-12
    M.flat[::p+1] += damp
    try:
        y = np.linalg.solve(M, b)
    except Exception:
        try:
            y, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            return R, 0.0
    v = r - JPt.T @ y
    V = v.reshape(n, d)
    # re-project to tangent (remove numerical normal)
    V -= (V * X).sum(axis=1, keepdims=True) * X
    vn = float(np.linalg.norm(v))
    ratio = vn / (rn + 1e-30)
    res_norm = float(np.linalg.norm(JPt @ v) / (float(np.linalg.norm(b)) + 1e-30))
    return V, ratio

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
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
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape
    iu0 = np.triu_indices(n, 1)

    def active_set(X, thr=1e-6, cap=1000):
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        slack = m - c
        idx = np.where(slack < thr)[0]
        if len(idx) > cap:
            # keep tightest
            order = np.argsort(slack[idx])[:cap]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        return m, slack, idx, I, J, G

    # quick contact counts for bias
    nwalks = 10
    for walk in range(nwalks):
        if time.time() > deadline - 40:
            break
        X = X_best.copy()
        m0 = _max_cos(X)
        # compute bias weights from current best active counts
        _, _, _, Ia, Ja, _ = active_set(X, thr=1e-6, cap=1100)
        cnt = np.zeros(n, dtype=float)
        if len(Ia) > 0:
            np.add.at(cnt, Ia, 1); np.add.at(cnt, Ja, 1)
        # bias away from overloaded: smaller weight for high contacts
        # also use LP-dual proxy: contacts normalized
        w = 1.0 / (1.0 + 0.25 * cnt)
        # boost rattlers strongly
        w[cnt <= 1] *= 2.0
        w /= (w.mean() + 1e-12)
        # vary bias per walk: randomize strength + different threshold
        thr = 1e-6 if (walk % 2 == 0) else 3e-7
        cap = 1000 if (walk % 3 != 0) else 850
        # different random scaling per walk
        seed_jitter = rng.randn(n) * 0.2 + 1.0
        w = np.maximum(0.15, w * seed_jitter)
        # remember initial active ids for topology detection
        _, _, idx_init, _, _, _ = active_set(X, thr=1e-6, cap=3000)
        set_init = set(idx_init.tolist())
        crawled = False
        for step in range(25):
            if time.time() > deadline - 38:
                break
            m, slack, idx, I, J, G = active_set(X, thr=thr, cap=cap)
            # refresh weights occasionally from fresh counts
            if step % 8 == 0 and step > 0:
                cc = np.zeros(n, dtype=float)
                if len(I) > 0:
                    np.add.at(cc, I, 1); np.add.at(cc, Ja if False else J, 1)
                w = 1.0 / (1.0 + 0.25 * cc)
                w[cc <= 1] *= 2.0
                w /= (w.mean() + 1e-12)
            V, ratio = _null_direction(X, I, J, w, rng)
            vn = float(np.linalg.norm(V))
            if vn < 1e-12 or ratio < 0.03:
                # jammed: try new random with looser cap
                cap = max(500, cap - 100)
                continue
            # normalize: rms per-point = 1
            rms = np.sqrt((V*V).sum() / n)
            V /= (rms + 1e-30)
            # line search largest alpha with rise <=1e-7
            # exponential search
            best_alpha = 0.0
            # try doubling from small
            alpha = 4e-4
            # first check if smallest feasible
            # evaluate incremental
            feas_alpha = 0.0
            # generate trial alphas doubling
            trial_alphas = [4e-4, 8e-4, 1.6e-3, 3.2e-3, 6.4e-3, 1.2e-2, 2.5e-2]
            for a in trial_alphas:
                if time.time() > deadline - 38:
                    break
                Xt = _norm(X + a * V)
                mv = _max_cos(Xt)
                if mv <= m0 + 1e-7:
                    feas_alpha = a
                else:
                    break
            if feas_alpha == 0:
                # try even smaller
                for a in [1e-4, 5e-5]:
                    Xt = _norm(X + a * V)
                    mv = _max_cos(Xt)
                    if mv <= m0 + 1e-7:
                        feas_alpha = a
                        break
                if feas_alpha == 0:
                    continue
            # refine: try 1.5x larger once
            try:
                Xt2 = _norm(X + feas_alpha*1.5 * V)
                if _max_cos(Xt2) <= m0 + 1e-7:
                    feas_alpha *= 1.5
            except Exception:
                pass
            X = _norm(X + feas_alpha * V)
            crawled = True
            m_new = _max_cos(X)
            # topology change? check active membership
            G2 = X @ X.T
            c2 = G2[iu0]
            slack2 = m_new - c2
            idx2 = set(np.where(slack2 < 1e-6)[0].tolist())
            # separated materially: some init contact now slack >5e-6
            separated = False
            if len(set_init) > 0:
                # sample check: count how many init left
                inter = len(set_init.intersection(idx2))
                if inter < len(set_init) - 3:
                    separated = True
            entered = len(idx2 - set_init) > 3
            if separated or entered:
                break
            # update m0 to allow drift but keep reference to original best?
            # keep m0 as max of original best +1e-7 to prevent runaway
            m0 = min(m0, m_new)
            # if we drifted too far above s_best, stop crawl
            if m_new > s_best + 5e-7:
                break
        if not crawled:
            continue
        # LP polish short
        Xp = _lp_polish(X, seconds=1.5, t_deadline=deadline-30)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best = Xp.copy(); s_best = float(sp)
    # final long polish
    remain = deadline - time.time()
    if remain > 5:
        X_best = _lp_polish(X_best, seconds=min(remain-1.0, 28.0))
        s_best = _max_cos(X_best)
    X_best = _norm(X_best)
    return X_best

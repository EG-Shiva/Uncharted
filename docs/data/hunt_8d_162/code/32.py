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
    iuI, iuJ = iu0[0], iu0[1]
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
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
            sel = mask
        I = iuI[sel]; J = iuJ[sel]; c = cf[sel]
        p = len(I)
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


def _relocate_to_hole(X, rng, k=1, n_samp=20000):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - 1e-7).sum(axis=1)
    order = np.argsort(cnt)
    victims = order[:k]
    keep = np.ones(n, dtype=bool)
    keep[victims] = False
    X_keep = X[keep]
    new_pts = []
    for _ in victims:
        C = rng.randn(n_samp, d)
        C = _norm(C)
        if X_keep.shape[0] > 0 or len(new_pts) > 0:
            if len(new_pts) > 0:
                ref = np.vstack([X_keep] + [q[None, :] for q in new_pts]) if X_keep.shape[0] > 0 else np.vstack([q[None, :] for q in new_pts])
            else:
                ref = X_keep
            dots = C @ ref.T
            mv = dots.max(axis=1)
            best = C[int(np.argmin(mv))]
        else:
            best = C[0]
        new_pts.append(best)
    Xn = X.copy()
    for idx, pt in zip(victims, new_pts):
        Xn[int(idx)] = pt
    return Xn


def _null_crawl(X, rng, steps, step_size, p_active, t_deadline=None):
    n, d = X.shape
    ND = n * d
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    for _ in range(steps):
        if t_deadline is not None and time.time() > t_deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cf = G[iuI, iuJ]
        p = int(min(p_active, cf.size))
        idx = np.argpartition(-cf, p - 1)[:p]
        I = iuI[idx]; J = iuJ[idx]
        # contact counts from this active set for weighting (move overloaded more)
        cnt = np.zeros(n, dtype=float)
        np.add.at(cnt, I, 1)
        np.add.at(cnt, J, 1)
        w = 0.5 + cnt / 8.0
        R = rng.randn(n, d)
        R -= (np.sum(R * X, axis=1, keepdims=True)) * X
        R *= w[:, None]
        r = R.ravel()
        M = p + n
        try:
            A = np.zeros((M, ND), dtype=np.float64)
        except Exception:
            break
        # active rows
        for kk in range(p):
            ii = int(I[kk]); jj = int(J[kk])
            A[kk, ii * d:(ii + 1) * d] = X[jj]
            A[kk, jj * d:(jj + 1) * d] = X[ii]
        for ii in range(n):
            A[p + ii, ii * d:(ii + 1) * d] = X[ii]
        try:
            Ar = A @ r
            Gram = A @ A.T
            # ridge
            Gram.flat[::M + 1] += 1e-9
            try:
                y = np.linalg.solve(Gram, Ar)
            except np.linalg.LinAlgError:
                y, *_ = np.linalg.lstsq(Gram, Ar, rcond=None)
            vflat = r - A.T @ y
        except Exception:
            vflat = r
        V = vflat.reshape(n, d)
        # enforce tangency (remove residual normal due to ridge)
        V -= (np.sum(V * X, axis=1, keepdims=True)) * X
        rms = float(np.sqrt(np.mean(V ** 2)))
        if not np.isfinite(rms) or rms < 1e-12:
            continue
        V /= rms
        Xn = X + step_size * V
        Xn = _norm(Xn)
        X = Xn
    return X


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(20261103)
    X_best = None
    s_best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < s_best:
                    s_best = s
                    X_best = v.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _norm(rng.randn(162, 8))
        s_best = _max_cos(X_best)
    n, d = X_best.shape
    # short baseline polish to lock jammed value
    if time.time() < deadline - 60:
        X_best = _lp_polish(X_best, 4.0, deadline - 50)
        s_best = _max_cos(X_best)
    trial = 0
    p_list = [700, 500, 900, 600, 800, 400]
    s_list = [0.016, 0.022, 0.011, 0.018, 0.008, 0.025]
    while time.time() < deadline - 22 and trial < 200:
        trial += 1
        p_active = p_list[trial % len(p_list)]
        step_size = s_list[(trial * 7) % len(s_list)]
        steps = 12 if (trial % 2 == 0) else 8
        Xc = X_best.copy()
        # pre-relocation on some trials to move rattler/low-contact point
        if trial % 3 == 0:
            try:
                Xc = _relocate_to_hole(Xc, rng, k=1, n_samp=15000)
            except Exception:
                pass
        try:
            Xc = _null_crawl(Xc, rng, steps, step_size, p_active, deadline - 20)
        except Exception:
            continue
        if time.time() > deadline - 20:
            break
        if trial % 4 == 1:
            try:
                # post-crawl single-point hole move for the loosest point
                Xc = _relocate_to_hole(Xc, rng, k=1, n_samp=15000)
            except Exception:
                pass
        try:
            Xp = _lp_polish(Xc, 2.0, deadline - 12)
        except Exception:
            continue
        s = _max_cos(Xp)
        if s < s_best - 1e-15:
            X_best = Xp
            s_best = s
            # extended polish on success
            if time.time() < deadline - 15:
                try:
                    X_best = _lp_polish(X_best, 3.0, deadline - 10)
                    s_best = _max_cos(X_best)
                except Exception:
                    pass
    # final polish of best
    if time.time() < deadline - 5:
        try:
            rem = max(1.0, min(8.0, (deadline - 3) - time.time()))
            X_best = _lp_polish(X_best, rem, deadline - 1)
        except Exception:
            pass
    X_best = _norm(X_best)
    return X_best

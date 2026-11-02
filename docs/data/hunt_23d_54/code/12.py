import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _norm(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm


def _lp_polish_fast(x, deadline, r0=0.015, rmax=0.05, cap=500):
    n, d = x.shape
    x = _norm(x.copy())
    best = _max_cos(x)
    best_x = x.copy()
    radius = float(r0)
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        # trust-region safe threshold, capped for speed
        allow = min(20.0 * radius, 0.04)
        thresh = m - allow
        # select
        if cap is not None and allow >= 0.039:
            # take top-cap only
            if c_all.size > cap:
                part = np.argpartition(c_all, -cap)[-cap:]
                sel = part
                c_sel = c_all[sel]
                I = iu0[sel]
                J = iu1[sel]
            else:
                sel = np.where(c_all > thresh)[0]
                c_sel = c_all[sel]
                I = iu0[sel]
                J = iu1[sel]
        else:
            sel = np.where(c_all > thresh)[0]
            if sel.size > cap:
                # keep tightest cap
                cs = c_all[sel]
                idx = np.argpartition(cs, -cap)[-cap:]
                sel = sel[idx]
            c_sel = c_all[sel]
            I = iu0[sel]
            J = iu1[sel]
        p = sel.size
        if p == 0:
            radius = min(radius * 1.5, rmax)
            continue
        nv = n * d + 1
        # vectorized A_ub: rows_I/J/t, cols, vals
        rows_I = np.repeat(np.arange(p), d)
        rows_J = np.repeat(np.arange(p), d)
        rows_T = np.arange(p)
        rows = np.concatenate([rows_I, rows_J, rows_T])
        col_I = (I[:, None] * d + ar).ravel()
        col_J = (J[:, None] * d + ar).ravel()
        col_T = np.full(p, nv - 1, dtype=np.int64)
        cols = np.concatenate([col_I, col_J, col_T])
        val_I = x[J].ravel()
        val_J = x[I].ravel()
        val_T = -np.ones(p, dtype=np.float64)
        vals = np.concatenate([val_I, val_J, val_T])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        erows = np.repeat(np.arange(n), d)
        ecols = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= deadline:
            break
    return best_x, best


def _best_hole(X, exclude, rng, n_sample=4000):
    n, d = X.shape
    S = rng.standard_normal(size=(n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-16
    if exclude is not None:
        mask = np.ones(n, dtype=bool)
        try:
            mask[np.atleast_1d(exclude).astype(int)] = False
        except Exception:
            pass
        Xe = X[mask]
    else:
        Xe = X
    D = S @ Xe.T
    worst = D.max(axis=1)
    # pick among 5 best holes randomly for diversity
    k = min(5, n_sample)
    idx = np.argpartition(worst, k - 1)[:k]
    # choose the absolute best most of time, sometimes 2nd best
    if rng.random() < 0.8:
        b = idx[np.argmin(worst[idx])]
    else:
        b = rng.choice(idx)
    return S[b] / (np.linalg.norm(S[b]) + 1e-16)


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_HARD = t0 + 176.0
    T_EXPLORE = t0 + 162.0
    rng = _np.random.default_rng(11)

    try:
        x0 = _np.load("best.npy").astype(float)
    except Exception:
        try:
            x0 = _np.load("record.npy").astype(float)
        except Exception:
            x0 = rng.standard_normal(size=(54, 23))
    try:
        xr = _np.load("record.npy").astype(float)
    except Exception:
        xr = x0
    x0 = _norm(x0)
    xr = _norm(xr)
    v0 = _max_cos(x0)
    vr = _max_cos(xr)
    if vr < v0:
        x_best, best = xr.copy(), vr
    else:
        x_best, best = x0.copy(), v0
    n, d = x_best.shape

    # brief initial tighten
    if _time.time() < T_EXPLORE:
        xb, vb = _lp_polish_fast(x_best, min(_time.time() + 2.0, T_EXPLORE), r0=0.003, rmax=0.02, cap=1431)
        if vb < best:
            x_best, best = xb, vb

    it = 0
    while _time.time() < T_EXPLORE:
        it += 1
        now = _time.time()
        if now >= T_EXPLORE:
            break
        G = x_best @ x_best.T
        _np.fill_diagonal(G, -2.0)
        thr_deg = best - 0.008
        deg = (G > thr_deg).sum(axis=1)
        order = _np.argsort(deg)
        loose10 = order[:10]
        tight10 = order[-10:]
        # worst pair
        flat = int(_np.argmax(G))
        wi, wj = flat // n, flat % n

        mode = it % 6
        y = x_best.copy()
        if mode == 0:
            sigma = 0.010 + 0.008 * rng.random()
            y = y + sigma * rng.standard_normal(size=y.shape)
            y = _norm(y)
            budget = 2.0
            r0 = 0.012
        elif mode == 1:
            idx = int(rng.choice(loose10))
            v = _best_hole(x_best, exclude=[idx], rng=rng, n_sample=4000)
            # tiny noise to avoid exact symmetry jam
            v = v + 0.01 * rng.standard_normal(size=(d,))
            v /= _np.linalg.norm(v)
            y[idx] = v
            budget = 2.5
            r0 = 0.02
        elif mode == 2:
            idx = int(rng.choice(loose10))
            v = rng.standard_normal(size=(d,))
            v /= _np.linalg.norm(v)
            y[idx] = v
            budget = 2.5
            r0 = 0.02
        elif mode == 3:
            idx = int(wi) if rng.random() < 0.5 else int(wj)
            v = _best_hole(x_best, exclude=[idx], rng=rng, n_sample=4000)
            v = v + 0.01 * rng.standard_normal(size=(d,))
            v /= _np.linalg.norm(v)
            y[idx] = v
            budget = 2.5
            r0 = 0.02
        elif mode == 4:
            seed = int(rng.choice(tight10))
            dots = x_best @ x_best[seed]
            k = 5 + int(rng.integers(0, 4))  # 5-8
            cl = _np.argpartition(dots, -k)[-k:]
            g_common = rng.standard_normal(size=(d,))
            sig_c = 0.05 + 0.03 * rng.random()
            sig_i = 0.01
            for cidx in cl:
                y[int(cidx)] = y[int(cidx)] + sig_c * g_common + sig_i * rng.standard_normal(size=(d,))
            y = _norm(y)
            budget = 2.5
            r0 = 0.018
        else:
            # double hole reinsert of two loose points
            idxs = rng.choice(loose10, size=2, replace=False)
            for kk, idx in enumerate(idxs):
                v = _best_hole(y, exclude=[int(idx)], rng=rng, n_sample=3000)
                y[int(idx)] = v
            # small global shake to detach
            y = y + 0.005 * rng.standard_normal(size=y.shape)
            y = _norm(y)
            budget = 3.0
            r0 = 0.02

        deadline = min(now + budget, T_EXPLORE + 0.4)
        if deadline > T_HARD - 12:
            deadline = T_HARD - 12
        if deadline <= now + 0.3:
            break
        yp, vp = _lp_polish_fast(y, deadline, r0=r0, rmax=0.05, cap=500)
        if vp < best - 1e-12:
            x_best, best = yp, vp

    rem = T_HARD - _time.time()
    if rem > 3:
        xb, vb = _lp_polish_fast(x_best, _time.time() + rem - 0.5, r0=0.005, rmax=0.03, cap=1431)
        if vb < best:
            x_best, best = xb, vb
    x_best = _norm(x_best)
    return x_best

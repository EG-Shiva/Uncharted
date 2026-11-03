import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _eval_cnt(X, iu, gap=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    c = G[iu]
    m = float(c.max())
    cnt = int(np.count_nonzero(c > m - gap))
    return m, cnt

def _lbfgs_run(X0, fun, maxiter):
    from scipy.optimize import minimize
    try:
        res = minimize(fun, _norm(X0).ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol': 1e-14,
                                'gtol': 1e-7, 'maxcor': 10, 'maxls': 20})
        return _norm(res.x.reshape(X0.shape))
    except Exception:
        return _norm(X0)

def _make_lse(beta, iu):
    def fun(Yf):
        n, d = 138, 8
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        mv = c.max()
        E = np.exp(beta * (c - mv))
        S = E.sum() + 1e-300
        f = float(mv + np.log(S) / beta)
        W = np.zeros((n, n))
        W[iu] = E / S
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        GY = Gt / nr
        return f, GY.ravel()
    return fun

def _make_topk(K, iu):
    def fun(Yf):
        n, d = 138, 8
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        if K >= c.size:
            sel = np.ones_like(c, dtype=bool)
        else:
            part = np.argpartition(-c, K - 1)[:K]
            sel = np.zeros_like(c, dtype=bool)
            sel[part] = True
        f = float(c[sel].sum() / K)
        W = np.zeros((n, n))
        Wij = np.zeros_like(c)
        Wij[sel] = 1.0 / K
        W[iu] = Wij
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        GY = Gt / nr
        return f, GY.ravel()
    return fun

def _make_smoothcount(tau, sharp, iu):
    def fun(Yf):
        n, d = 138, 8
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        z = np.clip(sharp * (c - tau), -50.0, 50.0)
        s = 1.0 / (1.0 + np.exp(-z))
        f = float(s.sum())
        w = s * (1.0 - s) * sharp
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        GY = Gt / (nr * sharp)
        return f / sharp, GY.ravel()
    return fun

def _lp_polish(x, deadline, r0=1e-2, cap=2500, active_gap=1e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - active_gap
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            if radius < 1e-3:
                radius = 1e-3
        else:
            radius *= 0.5
    return x

def solve():
    import time
    rng = np.random.default_rng(1)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    iu = np.triu_indices(n, 1)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_m, inc_cnt = _eval_cnt(incumbent, iu, 1e-7)

    def try_accept(Y, cur_m, cur_cnt, ref_m):
        # accept if cnt drops within tolerance on max rise
        try:
            m2, c2 = _eval_cnt(Y, iu, 1e-7)
        except Exception:
            return cur_m, cur_cnt, None
        if c2 < cur_cnt and m2 < ref_m + 1e-5:
            return m2, c2, Y
        if m2 < cur_m - 2e-9 and c2 <= cur_cnt + 20:
            return m2, c2, Y
        return cur_m, cur_cnt, None

    scales = [0.0, 0.002, 0.005, 0.0015]
    # stage plan: (kind, param, maxiter)
    for restart in range(len(scales)):
        if time.time() > deadline - 25:
            break
        sc = scales[restart]
        if sc == 0.0:
            start = incumbent.copy()
        else:
            start = _norm(incumbent + sc * rng.standard_normal((n, d)))
        cur = start.copy()
        cur_m, cur_cnt = _eval_cnt(cur, iu, 1e-7)
        ref_m = inc_m  # tolerance reference
        # short LSE / topk / smooth sequence
        stages = []
        stages.append(('lse', 200, 120))
        stages.append(('lse', 600, 80))
        stages.append(('topk', 500, 80))
        stages.append(('topk', 200, 80))
        stages.append(('topk', 50, 80))
        stages.append(('lse', 1500, 50))
        # smooth counts use current m as tau base
        for kind, par, it in stages:
            if time.time() > deadline - 20:
                break
            if kind == 'lse':
                fun = _make_lse(float(par), iu)
            else:
                fun = _make_topk(int(par), iu)
            Y = _lbfgs_run(cur, fun, it)
            nm, nc, acc = try_accept(Y, cur_m, cur_cnt, ref_m)
            if acc is not None:
                cur, cur_m, cur_cnt = acc, nm, nc
            else:
                # even without accept, allow m-improvement check
                try:
                    m2, c2 = _eval_cnt(Y, iu, 1e-7)
                    if m2 < cur_m - 5e-9:
                        cur, cur_m, cur_cnt = Y, m2, c2
                except Exception:
                    pass
        # smooth-count stages with taus from current m
        gaps = [(1e-5, 1e5, 60), (1e-6, 5e5, 60), (1e-7, 1.5e6, 60)]
        for gap, sharp, it in gaps:
            if time.time() > deadline - 20:
                break
            tau = cur_m - gap
            fun = _make_smoothcount(tau, float(sharp), iu)
            Y = _lbfgs_run(cur, fun, it)
            nm, nc, acc = try_accept(Y, cur_m, cur_cnt, ref_m)
            if acc is not None:
                cur, cur_m, cur_cnt = acc, nm, nc
        # one more LSE re-jam at moderate beta from de-jammed point
        if time.time() < deadline - 15:
            fun = _make_lse(600.0, iu)
            Y = _lbfgs_run(cur, fun, 80)
            nm, nc, acc = try_accept(Y, cur_m, cur_cnt, ref_m)
            if acc is not None:
                cur, cur_m, cur_cnt = acc, nm, nc
            else:
                try:
                    m2, c2 = _eval_cnt(Y, iu, 1e-7)
                    if m2 < cur_m:
                        cur, cur_m, cur_cnt = Y, m2, c2
                except Exception:
                    pass
        # LP polish this candidate if promising (cnt reduced or m close)
        if cur_m < inc_m + 1e-5:
            pol_end = min(deadline - 5, time.time() + 6.0)
            try:
                Yp = _lp_polish(cur, pol_end, r0=8e-3, cap=2500, active_gap=1e-3)
                mp, cp = _eval_cnt(Yp, iu, 1e-7)
                if mp < inc_m:
                    incumbent = Yp.copy()
                    inc_m, inc_cnt = mp, cp
                    ref_m = inc_m
                elif cp < inc_cnt and mp < inc_m + 2e-6:
                    # keep de-jammed for next restart base? polish again lightly
                    # re-polish with wider gap to exploit slack
                    Yp2 = _lp_polish(Yp, min(deadline - 5, time.time() + 3.0), r0=5e-3, cap=3000, active_gap=5e-4)
                    mp2, cp2 = _eval_cnt(Yp2, iu, 1e-7)
                    if mp2 < inc_m:
                        incumbent = Yp2.copy()
                        inc_m, inc_cnt = mp2, cp2
            except Exception:
                pass
        # also polish incumbent occasionally
        if restart == 1 and time.time() < deadline - 10:
            try:
                Yf = _lp_polish(incumbent, min(deadline - 5, time.time() + 4.0), r0=5e-3, cap=3000, active_gap=1e-3)
                mf, cf = _eval_cnt(Yf, iu, 1e-7)
                if mf < inc_m:
                    incumbent, inc_m, inc_cnt = Yf, mf, cf
            except Exception:
                pass

    # final polish
    try:
        if time.time() < deadline - 2:
            Yf = _lp_polish(incumbent, deadline - 0.5, r0=8e-3, cap=3000, active_gap=1e-3)
            mf, _ = _eval_cnt(Yf, iu, 1e-7)
            if mf < inc_m:
                incumbent = Yf
    except Exception:
        pass
    return _norm(incumbent)
